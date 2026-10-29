"""Beam front end: chooses a sensing beam from tracker and scheduler reports.

For each sensing opportunity the front end

1. predicts the tracker's state to each grant's transmission time (the gap is the scheduler's
   actual timing, not a fixed ``gap_k``);
2. builds the paper's admissible candidates around the grant's serving beam and the predicted
   dominant-scatterer direction;
3. bounds the UE gain loss over the scheduler's UE-direction knowledge (SRS estimate, or the
   grid-beam cell under U1) and keeps beams whose loss fits the grant's measured SINR within the
   permitted MCS drop;
4. predicts P_D with the multi-scatterer model (or the paper's point model) for the grant's REs and
   bandwidth with the UE as clutter. P_D is averaged over the pose uncertainty and, with
   ``rcs_mode="predictive"``, over the tracker's predictive RCS distribution;
5. returns the cheapest reuse whose predicted P_D reaches ``tau``, trying the sector ladder before
   the two-lobe baseline (the paper's sequential policy). Without one it falls back to a dedicated
   sensing beam.

Candidates are evaluated in increasing cost. A cost level is evaluated only when every cheaper
level has no feasible beam, which gives the same choice as evaluating all of them. The decision
carries the executed beam's predicted P_D versus RCS. ``rcs_look`` turns that curve and the sensing
report into the evidence used by ``rcs.RcsEstimator``.

Two P_D notions are kept apart:
- The decision value ``pd_pred`` is E_pose[max over scatterer cells]. This is a conservative
  bound on detecting the target in any cell.
- The evidence curve ``pd_curve`` is the crossing probability of the designated (dominant
  scatterer) cell alone. That cell is the event the back end reports for RCS estimation.

P_D values are compared with ``tau`` after rounding to ``PD_DECIMALS``. Batch-dependent last-bit
noise then cannot change feasibility or tie-breaks between lazy and exhaustive evaluation.

Deployment and calibration (``site=SiteConfig(...)``, see ``site.py``): every public call runs with
the site's parameters applied. P_D is averaged over the RCS nodes combined with the site's SNR
calibration nodes (``snr_offset_db``, ``snr_sigma_db``).

Measurement quality (``SensingRequirement``, via ``TrackReport.requirement`` or ``plan(...,
requirement=)``): P_D >= tau only says some cell is detected. A requirement adds a bound on the
predicted strongest-cell SNR and/or on the expected number of cells above a quality SNR. The
cheapest candidate meeting both is chosen; every decision reports ``pred_cell_snr_db`` and
``exp_quality_cells``. ``plan(..., return_table=True)`` evaluates all admissible candidates and
returns them in ``BeamDecision.candidates`` for reuse/accuracy trade-off studies.

A box with no face visible from the RSU (e.g. a ghost track next to the RSU) is handled as a point
target for that prediction.
"""

import functools

import math
import numpy as np
from scipy.special import ndtri
from . import model as V
from .detection import predictive_pd
from .interfaces import BeamDecision, RcsReport
from .quadrature import UELossCache
from .rcs import RcsLook
from .target import (FP, PoseNodes, box_scatterers, pose_nodes, extended_pd_shifts, cell_snr,
                     array_gain, pd_upper_bound, focus_snr, fixed_cell_pd_shifts, cell_pd)
from .tracker import heading_of, predict_state, polar_jacobian
from .detection import delay_power

SEQUENTIAL = (11, 13)
PD_DECIMALS = 12
PAPER_RCS_PRIOR_DBSM = -1.10319987650635  # main_calibration.json prior_dbsm


def _in_site(fn):
    """Run a public front-end method with the front end's site applied (re-entrant)."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if self.site is None or self._depth:
            return fn(self, *args, **kwargs)
        with self.site.active():
            self._depth += 1
            try:
                return fn(self, *args, **kwargs)
            finally:
                self._depth -= 1

    return wrapper


PD_OFFSETS_DB = np.arange(-20.0, 20.01, 1.0)
OBJECTIVES = ("pd", "coverage", "quality")
# Score resolution of the objectives: differences below it are ties, settled by cost. Detected
# cells saturate at the number of scatterers, so without this a 0.001-cell gain could buy an
# expensive beam.
OBJECTIVE_RESOLUTION = {"coverage": 0.01, "quality": 0.01}
PEAK_U = np.linspace(-1.0, 1.0, 4097)


def curve_sigma():
    lo, hi, step = FP["rcs_curve_db"]
    return np.arange(lo, hi + step / 2, step)


class BeamFrontEnd:
    """Beam front end. ``rsu``/``bore`` may come from ``site`` (``SiteConfig``) instead."""

    def __init__(self, rsu=None, bore=None, target_model="extended", tau=0.9, levels=SEQUENTIAL,
                 ue_vehicle_rcs_dbsm=0.0, rcs_mode=None, lazy=True, site=None,
                 on_unmet="dedicated", evidence_cell="scatterer", objective="pd", coverage_k=1.0,
                 dedicated_covering=False, extent_prior=None):
        if target_model not in ("extended", "point"):
            raise ValueError(target_model)
        if objective not in OBJECTIVES:
            raise ValueError(objective)
        self.objective, self.coverage_k = objective, float(coverage_k)
        self.dedicated_covering, self.extent_prior = bool(dedicated_covering), extent_prior
        self.site, self._depth = site, 0
        if on_unmet not in ("dedicated", "reuse_pd", "best_quality"):
            raise ValueError(on_unmet)
        if evidence_cell not in ("scatterer", "predicted", "event"):
            raise ValueError(evidence_cell)
        self.on_unmet, self.evidence_cell = on_unmet, evidence_cell
        if rsu is None:
            rsu = site.rsu_xy if site is not None else None
        if bore is None:
            bore = site.boresight_rad if site is not None else None
        if rsu is None or bore is None:
            raise ValueError("rsu and bore are required (directly or through site)")
        self.rcs_mode = FP["rcs_mode"] if rcs_mode is None else rcs_mode
        if self.rcs_mode not in ("predictive", "design"):
            raise ValueError(self.rcs_mode)
        self.rsu, self.bore = np.asarray(rsu, float), float(bore)
        self.target_model, self.tau, self.lazy = target_model, tau, lazy
        self.ue_rcs = ue_vehicle_rcs_dbsm
        lvl_tags, self.cand_level = V.levels()
        self.stages = [tuple(lvl_tags[i]) for i in levels]
        m = site.overrides().get("M", V.P["M"]) if site is not None else V.P["M"]
        sector = any(t >= 100 for st in self.stages for t in st)
        if sector and len(next(iter(V.sector_codebook().values()))) != m:
            raise ValueError(f"the sector codebook is for {V.P['M']} elements, not M={m}; "
                             "use levels without sector beams (e.g. (2, 13))")
        self.ue_loss = UELossCache()
        x, w = np.polynomial.legendre.leggauss(V.P["n_ue_clut"])
        self._clut_x, self._clut_w = x, w / 2
        n = V.P["n_ue_q"]
        self._cell_q = 2 * (np.arange(n) + 0.5) / n - 1

    # -- tracker side -------------------------------------------------------------------------
    @_in_site
    def prediction(self, track, t):
        """Predicted pose and the uncertainty the detector integrates over, at time ``t``.

        Scatterer RCS in the pose nodes (and ``rcs`` for the point model) refer to a 0 dBsm
        dominant scatterer; ``pd_shifts`` moves them to each RCS value.
        """
        x, P = predict_state(track.x, track.P, t - track.t)
        if track.heading is not None:
            h, sh = track.heading, track.heading_std or 0.0
        else:
            h, sh = heading_of(x, P)
        extent, degenerate = self.track_extent(track), False
        if extent is not None:
            keys, xy, _, dom, _ = box_scatterers(x[:2], h, *extent, self.rsu, 0.0)
            if dom is None:
                extent, degenerate = None, True
                refl = x[:2]
            else:
                refl = xy[keys.index(dom)]
        else:
            refl = x[:2]
        J = polar_jacobian(refl, self.rsu, self.bore)
        Sig = J @ P[:2, :2] @ J.T
        u, r, th, _ = V.to_polar(refl[None], self.rsu, self.bore, V.P["rsu_h"], V.P["rc_h"])
        pred = dict(
            x=x,
            P=P,
            heading=h,
            heading_std=sh,
            u_hat=float(u[0]),
            r_hat=float(r[0]),
            theta=float(th[0]),
            s_u=math.sqrt(Sig[0, 0]),
            s_r=math.sqrt(Sig[1, 1]),
            c_ur=float(Sig[0, 1]),
            extent=extent,
            degenerate_box=degenerate,
        )
        if self.target_model == "extended" and extent is not None:
            pred["nodes"] = pose_nodes(x[:2], P[:2, :2], h, sh, extent, self.rsu, self.bore, 0.0)
        return pred

    def default_rcs(self, track=None):
        """Design RCS when the tracker sends none: the site's per-class value for ``track.cls``
        when listed, else the site's ``rcs_prior_dbsm``, else the paper's pooled prior."""
        site = self.site
        cls = getattr(track, "cls", None)
        if site is not None and site.rcs_prior_by_class and cls in site.rcs_prior_by_class:
            return RcsReport(float(site.rcs_prior_by_class[cls]), source=f"site-class:{cls}")
        prior = site.rcs_prior_dbsm if site is not None else None
        return RcsReport(PAPER_RCS_PRIOR_DBSM if prior is None else prior,
                         source="calibration" if prior is None else "site")

    def rcs_nodes(self, rcs, calibrated=True):
        """RCS values (dBsm) and weights the predicted P_D is averaged over.

        With ``calibrated`` the site's SNR calibration nodes are folded in: offsets add to the
        RCS (dB) and weights multiply, so P_D is averaged over both uncertainties.
        """
        if self.rcs_mode == "predictive" and rcs.nodes_dbsm is not None and len(rcs.nodes_dbsm):
            sig = np.asarray(rcs.nodes_dbsm, float)
            w = np.ones(len(sig)) if rcs.weights is None else np.asarray(rcs.weights, float)
            if not (w.shape == sig.shape and np.all(w >= 0) and w.sum() > 0):
                raise ValueError("RcsReport weights must be non-negative, match nodes and sum > 0")
            sig, w = sig, w / w.sum()
        else:
            sig, w = np.array([rcs.design_dbsm], float), np.ones(1)
        if calibrated and self.site is not None:
            off, wo = self.site.snr_nodes()
            sig = (sig[:, None] + off[None, :]).ravel()
            w = (w[:, None] * wo[None, :]).ravel()
        return sig, w

    def calibration_nodes(self):
        """Site SNR calibration as (offsets added to the RCS in dB, weights); paper: ([0], [1])."""
        return self.site.snr_nodes() if self.site is not None else (np.zeros(1), np.ones(1))

    def snr_offset(self):
        return self.site.snr_offset_db if self.site is not None else 0.0

    def resolution_bw(self, grant_bw):
        """Bandwidth of the sensing range processing (cell coupling, delay CRB)."""
        return self.site.resolution_bandwidth(grant_bw) if self.site is not None else grant_bw

    @staticmethod
    def point_nodes(u, r, theta, s_u):
        """Single-reflector pose nodes over a Gaussian direction error (point target)."""
        z, wz = V.pointing_nodes(s_u)
        n = len(z)
        return PoseNodes((u + s_u * z)[:, None], np.full((n, 1), r), np.full((n, 1), theta), wz,
                         np.zeros(1), 0, ["point"])

    def _evidence_nodes(self, pred):
        if pred["extent"] is None:
            return self.point_nodes(pred["u_hat"], pred["r_hat"], pred["theta"], pred["s_u"])
        nodes = pred.get("nodes")
        if nodes is None:
            nodes = pose_nodes(pred["x"][:2], pred["P"][:2, :2], pred["heading"],
                               pred["heading_std"], pred["extent"], self.rsu, self.bore, 0.0)
        return nodes

    def pd_shifts(self, W, pred, n_re, bandwidth, clutter, sigma, screen=None):
        """P_D at each dominant RCS in ``sigma``: array (len(sigma), beams).

        ``screen`` = (threshold, weights) skips beams whose RCS-averaged clutter-free P_D is below
        the threshold (extended model; they return -1).
        """
        W = np.atleast_2d(W)
        sigma = np.atleast_1d(sigma)
        if "nodes" in pred:
            return extended_pd_shifts(W, pred["nodes"], n_re, bandwidth, clutter,
                                      shifts_db=sigma, screen=screen)[0]
        kw = dict(s_r=pred["s_r"], c_ur=pred["c_ur"])
        out = []
        for s in sigma:
            args = (W, pred["u_hat"], pred["r_hat"], pred["s_u"], s, pred["theta"], n_re)
            if clutter is None:
                out.append(predictive_pd(*args, **kw)[0])
            else:
                out.append(predictive_pd(*args, bandwidth, clutter, **kw)[0])
        return np.array(out)

    def evidence_curve(self, W, pred, n_re, bandwidth, clutter):
        """Designated-cell crossing probability of beam ``W`` on the ``curve_sigma`` grid.

        ``evidence_cell="scatterer"``: the cell follows the dominant scatterer, as in the paper's
        conditional detector and the closed-loop back end. ``"predicted"``: the cell is fixed at the
        decision-time predicted dominant position (``BeamDecision.u_hat``, ``r_hat``), and the pose
        error moves the scatterers relative to it. ``"event"``: no designated cell; the curve is the
        look's own detection event (any scatterer cell, pose-averaged, UE leakage, site
        calibration nodes folded in), the event ``pd_pred`` and ``pd_offset_curve`` describe.
        ``bandwidth`` is the resolution bandwidth.
        """
        grid = curve_sigma()
        if self.evidence_cell == "event":
            return self.offset_curve(W, pred, n_re, bandwidth, clutter, *self.calibration_nodes(),
                                     offsets=grid)
        model_grid = grid - self.snr_offset()
        W = np.atleast_2d(W)
        if self.evidence_cell == "predicted":
            nodes = self._evidence_nodes(pred)
            pd = fixed_cell_pd_shifts(W, nodes, n_re, bandwidth, pred["u_hat"], pred["r_hat"],
                                      clutter, shifts_db=model_grid)
            return grid, pd[:, 0]
        if pred["extent"] is None:
            return grid, self.pd_shifts(W, pred, n_re, bandwidth, clutter, model_grid)[:, 0]
        pd = extended_pd_shifts(W, self._evidence_nodes(pred), n_re, bandwidth, clutter,
                                focus="dominant", shifts_db=model_grid)[0]
        return grid, pd[:, 0]

    def quality(self, W, pred, n_re, bandwidth, clutter, sigma, weights, quality_snr_db=None,
                rho=None):
        """Predicted measurement quality per beam.

        Returns (E[strongest-cell SNR] in dB, E[number of cells with SNR >= quality_snr_db]).
        SNR is over noise plus the mean UE leakage of the clutter quadrature; expectations are
        over pose nodes and the RCS (and calibration) nodes. The count is NaN without a quality
        SNR. A point target has one cell. ``rho`` passes precomputed box cell SNRs.
        """
        W = np.atleast_2d(W)
        if "nodes" in pred:
            nodes = pred["nodes"]
            rho = cell_snr(W, nodes, n_re, bandwidth) if rho is None else rho
            cu_n, cr_n, wn = nodes.u, nodes.r, nodes.w
        else:
            z, wn = V.pointing_nodes(pred["s_u"])
            uz = pred["u_hat"] + pred["s_u"] * z
            rho = 10 ** (V.rc_snr_db(array_gain(W, uz), uz, pred["r_hat"], 0.0, pred["theta"],
                                     n_re) / 10)[:, :, None]
            cu_n, cr_n = uz[:, None], np.full((len(uz), 1), pred["r_hat"])
        leak = 0.0
        if clutter is not None:
            cu, cr, cs, ct = clutter[:4]
            cu = np.atleast_1d(cu)
            cw = np.asarray(clutter[4]) if len(clutter) > 4 else np.ones(len(cu)) / len(cu)
            cp = 10 ** (V.rc_snr_db(array_gain(W, cu), cu, cr, cs, ct, n_re) / 10)
            resp = V.rx_rel(cu_n[None] - cu[:, None, None]) * delay_power(cr_n - cr, bandwidth)
            leak = np.einsum("bq,q,qnk->bnk", cp, cw, resp)
        sinr = 10 * np.log10(np.maximum(rho / (1 + leak), 1e-30))
        snr = sinr.max(axis=2) @ wn + float(weights @ sigma)
        if quality_snr_db is None:
            return snr, np.full(len(W), np.nan)
        cells = sum(wk * ((sinr + s >= quality_snr_db).sum(axis=2) @ wn)
                    for s, wk in zip(sigma, weights))
        return snr, cells

    def track_extent(self, track):
        """Box extent used for a track: ``extent_prior(track)`` when set, else ``track.extent``."""
        return self.extent_prior(track) if self.extent_prior is not None else track.extent

    def coverage_points(self, pred, k=None):
        """Directions (u, elevation) of the predicted box inflated by k sigma.

        The box (or a point when there is no extent) is grown by k times the position standard
        deviation along each body axis; its outline (9 points per side) and centre are sampled.
        """
        k = self.coverage_k if k is None else float(k)
        x, P, h = pred["x"], pred["P"], pred["heading"]
        e1 = np.array([math.cos(h), math.sin(h)])
        e2 = np.array([-math.sin(h), math.cos(h)])
        s1 = math.sqrt(max(float(e1 @ P[:2, :2] @ e1), 0.0))
        s2 = math.sqrt(max(float(e2 @ P[:2, :2] @ e2), 0.0))
        L, Wd = pred["extent"] if pred["extent"] is not None else (0.0, 0.0)
        a, b = L / 2 + k * s1, Wd / 2 + k * s2
        t = np.linspace(-1.0, 1.0, 9)
        local = np.concatenate([np.c_[np.full(9, a), t * b], np.c_[np.full(9, -a), t * b],
                                np.c_[t * a, np.full(9, b)], np.c_[t * a, np.full(9, -b)],
                                np.zeros((1, 2))])
        xy = x[:2] + local[:, :1] * e1 + local[:, 1:] * e2
        u, _, th, _ = V.to_polar(xy, self.rsu, self.bore, V.P["rsu_h"], V.P["rc_h"])
        return u, th

    def coverage_beams(self, W, pts):
        """Per beam: (worst-case gain over the points relative to the beam's peak in dB,
        absolute worst-case gain in dBi = array gain M*G plus element gain)."""
        W = np.atleast_2d(W)
        u, th = pts
        G = array_gain(W, u)
        peak = array_gain(W, PEAK_U).max(axis=1)
        cov = 10 * np.log10(np.maximum(G.min(axis=1) / peak, 1e-30))
        dbi = 10 * np.log10(np.maximum(V.M * G, 1e-30)) + V.elem_dbi(np.arcsin(np.clip(u, -1, 1)), th)
        return cov, dbi.min(axis=1)

    def _cell_terms(self, W, pred, n_re, bw, clutter, rho=None):
        """Cell SNRs, mean UE leakage and nodes of beams W for the detected-cell count."""
        nodes = pred["nodes"] if "nodes" in pred else self._evidence_nodes(pred)
        W = np.atleast_2d(W)
        rho = cell_snr(W, nodes, n_re, bw) if rho is None else rho
        return rho, self._mean_leak(W, nodes, n_re, bw, clutter), nodes

    @staticmethod
    def detected_cells(rho, leak, nodes, sigma, weights):
        """Expected number of scatterer cells crossing the threshold, per beam: sum over cells of
        the per-cell crossing probability, averaged over pose and RCS (and calibration) nodes,
        with the mean UE leakage."""
        lp = math.log(V.P["pfa"])
        sinr = rho / (1 + leak)
        return sum(w * (cell_pd(x * sinr, None, lp).sum(axis=2) @ nodes.w)
                   for x, w in zip(10 ** (np.asarray(sigma) / 10), weights))

    def _annotate(self, d, pred, parts, bw, sigma, weights):
        """Fill coverage_db, coverage_gain_dbi and exp_detected_cells of a decision.

        ``parts`` = [(W, n_re, clutter)]. Several parts pool by energy: per point,
        sum_i n_re_i G_i / sum_i n_re_i peak_i (coverage_db) and the n_re-weighted mean gain
        (coverage_gain_dbi); ``info["coverage_maxpool_db"]`` gives the per-point maximum of
        G_i / peak_i instead.
        """
        pts = self.coverage_points(pred)
        u, th = pts
        n = np.array([p[1] for p in parts], float)
        G = np.array([array_gain(np.atleast_2d(p[0]), u)[0] for p in parts])
        peak = np.array([array_gain(np.atleast_2d(p[0]), PEAK_U)[0].max() for p in parts])
        pooled = (n @ G) / (n @ peak)
        d.coverage_db = float(10 * np.log10(max(pooled.min(), 1e-30)))
        elem = V.elem_dbi(np.arcsin(np.clip(u, -1, 1)), th)
        d.coverage_gain_dbi = float((10 * np.log10(np.maximum(V.M * (n @ G) / n.sum(), 1e-30))
                                     + elem).min())
        rho = leak = 0.0
        for W, n_re, cl in parts:
            r_, l_, nodes = self._cell_terms(W, pred, n_re, bw, cl)
            rho, leak = rho + r_, leak + l_
        d.exp_detected_cells = float(self.detected_cells(rho, leak, nodes, sigma, weights)[0])
        d.info["coverage_k"] = self.coverage_k
        if len(parts) > 1:
            d.info["coverage_pooling"] = "energy: sum n_re*G / sum n_re*peak per point"
            d.info["coverage_maxpool_db"] = float(10 * np.log10(max(
                (G / peak[:, None]).max(axis=0).min(), 1e-30)))
        return d

    def offset_curve(self, W, pred, n_re, bandwidth, clutter, sigma, weights, offsets=None):
        """Decision-event P_D of one beam versus an SNR offset added to every target return.

        Same event as ``pd_pred`` (any scatterer cell; pose, RCS and calibration nodes averaged;
        UE leakage), recomputed at each offset in dB. Returns (offsets_db, pd). At offset 0 it
        equals the decision's P_D.
        """
        offsets = PD_OFFSETS_DB if offsets is None else np.asarray(offsets, float)
        shifts = (np.asarray(sigma)[:, None] + offsets[None, :]).ravel()
        pd = self.pd_shifts(np.atleast_2d(W), pred, n_re, bandwidth, clutter, shifts)[:, 0]
        return offsets.copy(), np.asarray(weights) @ pd.reshape(len(sigma), len(offsets))

    @staticmethod
    def meets(req, snr, cells, cov=None):
        if req is None or not req.active():
            return True
        ok = True
        if req.min_coverage_db is not None:
            ok &= cov is not None and bool(cov >= req.min_coverage_db)
        if req.min_cell_snr_db is not None:
            ok &= snr >= req.min_cell_snr_db
        if req.min_cells is not None and req.quality_snr_db is not None:
            ok &= cells >= req.min_cells
        if req.quality_logistic is not None and req.min_quality_prob is not None:
            ok &= req.quality_prob(snr) >= req.min_quality_prob
        return bool(ok)

    def detect(self, W, pred, n_re, bandwidth, clutter, sigma, weights, threshold=None):
        """RCS-averaged predicted P_D per beam (-1 for beams screened out by ``threshold``)."""
        screen = None if threshold is None else (threshold, weights)
        pd = self.pd_shifts(W, pred, n_re, bandwidth, clutter, sigma, screen)
        return np.where((pd < 0).any(axis=0), -1.0, weights @ pd)

    # -- scheduler side -----------------------------------------------------------------------
    @_in_site
    def ue_knowledge(self, g):
        """UE design window, gain-loss directions and clutter quadrature for a grant."""
        g0 = V.GRID[g.k0]
        if g.ue_u is not None and g.ue_u_std:
            s = g.ue_u_std
            uq = g.ue_u + s * ndtri((np.arange(FP["ue_dir_nodes"]) + 0.5) / FP["ue_dir_nodes"])
            x, w = np.polynomial.hermite.hermgauss(9)
            return (g.ue_u, g.ue_u - 2 * s, g.ue_u + 2 * s), uq, g.ue_u + s * math.sqrt(2) * x, \
                w / math.sqrt(math.pi)
        if g.ue_u is not None:
            return (g.ue_u, g.ue_u, g.ue_u), np.array([g.ue_u]), np.array([g.ue_u]), np.ones(1)
        half = 1 / (2 * V.M)  # half the grid-beam cell (1/64 for the paper's M = 32)
        uq = g0 + self._cell_q * half
        return (g0, g0 - half, g0 + half), uq, g0 + self._clut_x * half, self._clut_w

    def clutter(self, g):
        if not V.P["ue_clutter"]:
            return None
        _, _, cu, cw = self.ue_knowledge(g)
        sig_ue = V.P["sigma_ped_dbsm"] if g.ue_kind != "vehicle" else self.ue_rcs
        return (cu, g.ue_range_m, sig_ue, g.ue_theta_deg, cw)

    @_in_site
    def evaluate_grant(self, g, pred):
        """Admissible candidates of one grant with loss and MCS cost; P_D is filled lazily.

        ``n_re`` (integration) comes from the grant; ``bw`` is the resolution bandwidth used for
        cell coupling (the grant bandwidth unless the site sets ``sensing_bandwidth_hz``).
        """
        ue_ctrl, uq, _, _ = self.ue_knowledge(g)
        W, tag = V.candidates(
            ue_ctrl[0], pred["u_hat"], g.k0, self.cand_level, sigma_u=pred["s_u"], ue_ctrl=ue_ctrl
        )
        keep = np.isin(tag, sorted(set().union(*self.stages)))
        W, tag = W[keep], tag[keep]
        if g.ue_u is None:
            loss = self.ue_loss(W, V.GRID[g.k0], uq)
        else:
            ref = V.db(V.gain(V.beam(V.GRID[g.k0]), uq))[0]
            q = ref[None, :] - V.db(V.gain(W, uq))
            loss = np.quantile(q, 1 - V.P["eps_ue"], axis=1, method="inverted_cdf")
        m0 = next(m for m in V.MCS if m[0] == g.mcs)
        mmin = max(g.mcs - g.max_mcs_drop, V.P["mcs_floor"])
        thr = next(m for m in V.MCS if m[0] == mmin)[1]
        ok = loss <= g.sinr_db - thr + 1e-9
        mcs = [min(V.mcs_of(g.sinr_db - l, mmin), m0, key=lambda m: m[2]) if o else None
               for l, o in zip(loss, ok)]
        se = np.array([m[2] if m is not None else np.nan for m in mcs])
        grant_bw = g.n_prb * 12 * V.P["df"]
        return dict(W=W, tag=tag, loss=loss, ok=ok, pd=np.full(len(W), np.nan), dse=m0[2] - se,
                    n_re=g.n_prb * 12 * g.n_sym, bw=self.resolution_bw(grant_bw),
                    grant_bw=grant_bw, mcs=mcs, clutter=self.clutter(g),
                    q_snr=np.full(len(W), np.nan), q_cells=np.full(len(W), np.nan), rho={},
                    q_cov=np.full(len(W), np.nan), q_det=np.full(len(W), np.nan))

    def _rho(self, e, pred, cs):
        """Box cell SNRs of candidates ``cs``, computed once per (grant, candidate) and cached."""
        miss = [c for c in cs if c not in e["rho"]]
        if miss:
            R = cell_snr(e["W"][miss], pred["nodes"], e["n_re"], e["bw"])
            for k, c in enumerate(miss):
                e["rho"][c] = R[k]
        return np.stack([e["rho"][c] for c in cs])

    def _fill_quality(self, rows, pairs, sigma, weights, s_q):
        todo = {}
        for i, c in pairs:
            if np.isnan(rows[i][3]["q_snr"][c]):
                todo.setdefault(i, []).append(c)
        for i, cs in todo.items():
            _, _, pred, e = rows[i]
            cs = np.array(sorted(cs))
            rho = self._rho(e, pred, cs) if "nodes" in pred else None
            e["q_snr"][cs], e["q_cells"][cs] = self.quality(
                e["W"][cs], pred, e["n_re"], e["bw"], e["clutter"], sigma, weights, s_q, rho=rho)
            e["q_cov"][cs] = self.coverage_beams(e["W"][cs], self.coverage_points(pred))[0]

    def _fill_detected(self, rows, pairs, sigma, weights):
        todo = {}
        for i, c in pairs:
            if np.isnan(rows[i][3]["q_det"][c]):
                todo.setdefault(i, []).append(c)
        for i, cs in todo.items():
            _, _, pred, e = rows[i]
            cs = np.array(sorted(cs))
            rho = self._rho(e, pred, cs) if "nodes" in pred else None
            r_, l_, nodes = self._cell_terms(e["W"][cs], pred, e["n_re"], e["bw"], e["clutter"], rho)
            e["q_det"][cs] = self.detected_cells(r_, l_, nodes, sigma, weights)

    def _fill_pd(self, rows, pairs, sigma, weights):
        """Evaluate predicted P_D for the (row, candidate) pairs that have none yet.

        Box model: a cheap rigorous bound (``target.pd_upper_bound``) first marks beams that
        cannot reach ``tau`` with -1, then the cached cell SNRs feed the clutter-free screen and
        the full P_D. A -1 entry means "screened out, below tau".
        """
        todo = {}
        for i, c in pairs:
            if np.isnan(rows[i][3]["pd"][c]):
                todo.setdefault(i, []).append(c)
        for i, cs in todo.items():
            _, _, pred, e = rows[i]
            cs = np.array(sorted(cs))
            if "nodes" not in pred:
                e["pd"][cs] = self.detect(e["W"][cs], pred, e["n_re"], e["bw"], e["clutter"],
                                          sigma, weights, threshold=self.tau)
                continue
            ub = pd_upper_bound(e["W"][cs], pred["nodes"], e["n_re"], sigma, weights)
            keep = cs[ub >= self.tau - 1e-11]
            e["pd"][cs[ub < self.tau - 1e-11]] = -1.0
            if keep.size:
                pd = extended_pd_shifts(e["W"][keep], pred["nodes"], e["n_re"], e["bw"],
                                        e["clutter"], shifts_db=sigma, screen=(self.tau, weights),
                                        rho=self._rho(e, pred, keep))[0]
                e["pd"][keep] = np.where((pd < 0).any(axis=0), -1.0, weights @ pd)
        return sum(len(cs) for cs in todo.values())

    # -- decision -----------------------------------------------------------------------------
    @_in_site
    def plan(self, track, grants, rcs=None, t=None, requirement=None, return_table=False,
             on_unmet=None, objective=None):
        """BeamDecision for one sensing opportunity from a TrackReport, GrantReports and RcsReport.

        ``rcs`` defaults to the site's (or the paper's) design RCS. ``requirement`` overrides
        ``track.requirement``. ``return_table`` evaluates every admissible candidate and returns
        them in ``BeamDecision.candidates``.

        ``on_unmet`` (default: the front end's, "dedicated") applies when no reuse meets both
        P_D >= tau and the requirement:
        - "dedicated": dedicated sensing beam;
        - "reuse_pd": the cheapest reuse with P_D >= tau (what plan gives without a requirement);
        - "best_quality": among reuses with P_D >= tau, the largest ``pred_cell_snr_db``.
        All within the same search; ``info["fallback"]`` names the path taken ("none" when the
        requirement was met or absent).

        ``objective`` (default: the front end's, "pd") selects among feasible reuses (P_D >= tau
        and the requirement, over all stages at once):
        - "pd": cheapest, then highest P_D (the paper's sequential policy, unchanged);
        - "coverage": largest ``coverage_db`` (worst-case gain over the box inflated by
          ``coverage_k`` sigma, relative to the beam peak), then cheapest;
        - "quality": largest ``exp_detected_cells``, then cheapest; combine with
          ``SensingRequirement.min_coverage_db`` for "cells subject to coverage".
        These two do not minimise communication cost (it is reported); score differences below
        ``OBJECTIVE_RESOLUTION`` (0.01 dB, 0.01 cells) are ties settled by cost. ``info["binding"]`` lists
        the constraints that removed candidates ("tau", "min_coverage_db", "requirement").
        """
        objective = self.objective if objective is None else objective
        if objective not in OBJECTIVES:
            raise ValueError(objective)
        on_unmet = self.on_unmet if on_unmet is None else on_unmet
        if on_unmet not in ("dedicated", "reuse_pd", "best_quality"):
            raise ValueError(on_unmet)
        rcs = self.default_rcs(track) if rcs is None else rcs
        req = requirement if requirement is not None else getattr(track, "requirement", None)
        req = req if req is not None and req.active() else None
        s_q = req.quality_snr_db if req is not None else None
        sigma, weights = self.rcs_nodes(rcs)
        rows = []
        for i, g in enumerate(grants):
            pred = self.prediction(track, g.t)
            rows.append((i, g, pred, self.evaluate_grant(g, pred)))
        n_cand = sum(int(e["ok"].sum()) for *_, e in rows)
        n_eval = 0
        admissible = [(i, c) for i, _, _, e in rows for c in np.flatnonzero(e["ok"])]
        if not self.lazy or return_table:
            n_eval += self._fill_pd(rows, admissible, sigma, weights)
        if return_table:
            self._fill_quality(rows, admissible, sigma, weights, s_q)

        def pdq(ic):
            return round(float(rows[ic[0]][3]["pd"][ic[1]]), PD_DECIMALS)

        def order(ic):
            return (-pdq(ic), rows[ic[0]][3]["loss"][ic[1]], ic[0], ic[1])

        if objective != "pd":
            tags = [t_ for st in self.stages for t_ in st]
            staged = [(i, c) for i, c in admissible if np.isin(rows[i][3]["tag"][c], tags)]
            n_eval += self._fill_pd(rows, staged, sigma, weights)
            tau_ok = [ic for ic in staged if pdq(ic) >= self.tau]
            self._fill_quality(rows, tau_ok, sigma, weights, s_q)
            self._fill_detected(rows, tau_ok, sigma, weights)

            def ev(ic):
                return rows[ic[0]][3]

            cov_fail = [ic for ic in tau_ok if req is not None and req.min_coverage_db is not None
                        and not ev(ic)["q_cov"][ic[1]] >= req.min_coverage_db]
            feas = [ic for ic in tau_ok if self.meets(req, ev(ic)["q_snr"][ic[1]],
                                                      ev(ic)["q_cells"][ic[1]],
                                                      ev(ic)["q_cov"][ic[1]])]
            other_fail = [ic for ic in tau_ok if ic not in feas and ic not in cov_fail]
            binding = (["tau"] if len(tau_ok) < len(staged) else []) + (
                ["min_coverage_db"] if cov_fail else []) + (["requirement"] if other_fail else [])
            stats = dict(objective=objective, binding=binding, n_admissible=len(staged),
                         n_tau=len(tau_ok), n_feasible=len(feas))
            if feas:
                col = "q_cov" if objective == "coverage" else "q_det"
                res_ = OBJECTIVE_RESOLUTION[objective]
                i, c = min(feas, key=lambda ic: (-round(ev(ic)[col][ic[1]] / res_),
                                                 ev(ic)["n_re"] * ev(ic)["dse"][ic[1]],
                                                 -pdq(ic), ev(ic)["loss"][ic[1]], ic[0], ic[1]))
                e = ev((i, c))
                stage = next(st for st in self.stages if e["tag"][c] in st)
                d = self._reuse(rows, i, c, e["n_re"] * e["dse"][c], stage, rcs, sigma, weights,
                                req, s_q, n_cand, n_eval,
                                rows_table=admissible if return_table else None,
                                met=None if req is None else True, fallback="none")
                d.info.update(stats)
                return d
            d = self.plan(track, grants, rcs, t, requirement=req, return_table=return_table,
                          on_unmet=on_unmet, objective="pd")
            d.info.update(stats)
            d.info["objective_fallback"] = "no candidate met tau and the requirement"
            return d

        first_pd = None  # cheapest P_D-feasible reuse, for on_unmet="reuse_pd"
        for stage in self.stages:
            pool = {}
            for i, g, pred, e in rows:
                for c in np.flatnonzero(np.isin(e["tag"], stage) & e["ok"]):
                    pool.setdefault(e["n_re"] * e["dse"][c], []).append((i, c))
            for cost in sorted(pool):
                group = pool[cost]
                n_eval += self._fill_pd(rows, group, sigma, weights)
                feasible = [ic for ic in group if pdq(ic) >= self.tau]
                if feasible and first_pd is None:
                    first_pd = (min(feasible, key=order), cost, stage)
                if feasible and req is not None:
                    self._fill_quality(rows, feasible, sigma, weights, s_q)
                    feasible = [ic for ic in feasible
                                if self.meets(req, rows[ic[0]][3]["q_snr"][ic[1]],
                                              rows[ic[0]][3]["q_cells"][ic[1]],
                                              rows[ic[0]][3]["q_cov"][ic[1]])]
                if feasible:
                    i, c = min(feasible, key=order)
                    return self._reuse(rows, i, c, cost, stage, rcs, sigma, weights, req, s_q,
                                       n_cand, n_eval, rows_table=admissible if return_table else None,
                                       met=None if req is None else True, fallback="none")
        if req is not None and on_unmet == "reuse_pd" and first_pd is not None:
            (i, c), cost, stage = first_pd
            return self._reuse(rows, i, c, cost, stage, rcs, sigma, weights, req, s_q, n_cand,
                               n_eval, rows_table=admissible if return_table else None,
                               met=False, fallback="reuse_pd")
        if req is not None and on_unmet == "best_quality" and first_pd is not None:
            staged = [(i, c) for i, c in admissible
                      if np.isin(rows[i][3]["tag"][c], [t_ for st in self.stages for t_ in st])]
            n_eval += self._fill_pd(rows, staged, sigma, weights)
            feasible = [ic for ic in staged if pdq(ic) >= self.tau]
            self._fill_quality(rows, feasible, sigma, weights, s_q)
            i, c = max(feasible, key=lambda ic: (rows[ic[0]][3]["q_snr"][ic[1]],
                                                 -rows[ic[0]][3]["n_re"] * rows[ic[0]][3]["dse"][ic[1]],
                                                 -rows[ic[0]][3]["loss"][ic[1]], -ic[0], -ic[1]))
            e = rows[i][3]
            stage = next(st for st in self.stages if e["tag"][c] in st)
            return self._reuse(rows, i, c, e["n_re"] * e["dse"][c], stage, rcs, sigma, weights,
                               req, s_q, n_cand, n_eval,
                               rows_table=admissible if return_table else None,
                               met=self.meets(req, e["q_snr"][c], e["q_cells"][c], e["q_cov"][c]),
                               fallback="best_quality")
        d = self.dedicated(track, grants, rcs, t, n_cand, n_eval, requirement=req)
        d.info["fallback"] = "dedicated" if req is not None else "none"
        d.candidates = self._table(rows, admissible) if return_table else None
        return d

    def _reuse(self, rows, i, c, cost, stage, rcs, sigma, weights, req, s_q, n_cand, n_eval,
               rows_table, met, fallback):
        _, g, pred, e = rows[i]
        W = e["W"][c]
        self._fill_quality(rows, [(i, c)], sigma, weights, s_q)
        grid, curve = self.evidence_curve(W, pred, e["n_re"], e["bw"], e["clutter"])
        snr = float(e["q_snr"][c])
        offset_curve = self.offset_curve(W, pred, e["n_re"], e["bw"], e["clutter"], sigma, weights)
        d = BeamDecision(
            "reuse",
            g.t,
            grant=i,
            ue_id=g.ue_id,
            weights=W,
            tag=int(e["tag"][c]),
            pd_pred=float(e["pd"][c]),
            pd_curve_sigma=grid,
            pd_curve=curve,
            ue_loss_db=float(e["loss"][c]),
            mcs=e["mcs"][c][0],
            dse=float(e["dse"][c]),
            cost_bits=float(cost),
            n_re=int(e["n_re"]),
            bandwidth=float(e["grant_bw"]),
            rcs_design_dbsm=rcs.design_dbsm,
            rcs_mean_dbsm=float(weights @ sigma),
            n_candidates=n_cand,
            info=dict(stage=stage, rcs_source=rcs.source, n_evaluated=n_eval,
                      rcs_nodes=sigma.tolist(), rcs_weights=weights.tolist(),
                      degenerate_box=pred["degenerate_box"], fallback=fallback),
            pred_cell_snr_db=snr,
            exp_quality_cells=float(e["q_cells"][c]),
            requirement_met=met,
            candidates=self._table(rows, rows_table) if rows_table is not None else None,
            resolution_bw=float(e["bw"]),
            quality_prob=float(req.quality_prob(snr)) if req is not None else float("nan"),
            pd_offset_curve=offset_curve,
            **self._pred_fields(pred),
        )
        return self._annotate(d, pred, [(W, e["n_re"], e["clutter"])], e["bw"], sigma, weights)

    # -- frame-level planning for several tracks ------------------------------------------------
    @_in_site
    def plan_frame(self, tracks, grants, rcs=None, t=None, requirements=None, on_unmet=None,
                   objective=None):
        """Grant assignment for several tracks in one frame (greedy, in the given order).

        A grant carries one executed beam. Track k first checks the grants already taken by
        earlier tracks: if the beam executed there also gives this track P_D >= tau and meets
        its requirement, it shares that grant and beam at no extra communication cost
        (``info["shared_with"]`` = index of the owning track, ``cost_bits`` = 0). Otherwise it
        runs ``plan`` over the grants not yet taken. Put the most important tracks first
        (e.g. confirmed, most hits). ``rcs`` and ``requirements`` are per-track lists (or None).
        Returns one BeamDecision per track; ``grant`` indexes the full ``grants`` list.
        No grant is ever used with two different beams.
        """
        n = len(tracks)
        rcs = [None] * n if rcs is None else list(rcs)
        reqs = [None] * n if requirements is None else list(requirements)
        taken = {}  # grant index -> (owner track, decision)
        out = []
        for k, tr in enumerate(tracks):
            r_k = self.default_rcs(tr) if rcs[k] is None else rcs[k]
            req = reqs[k] if reqs[k] is not None else getattr(tr, "requirement", None)
            req = req if req is not None and req.active() else None
            sigma, weights = self.rcs_nodes(r_k)
            share = None
            for gi, (owner, dprev) in taken.items():
                g = grants[gi]
                pred = self.prediction(tr, g.t)
                e = self.evaluate_grant(g, pred)
                W = dprev.weights[None]
                pd = float(self.detect(W, pred, e["n_re"], e["bw"], e["clutter"], sigma, weights)[0])
                snr, cells = self.quality(W, pred, e["n_re"], e["bw"], e["clutter"], sigma,
                                          weights, req.quality_snr_db if req is not None else None)
                cov = self.coverage_beams(W, self.coverage_points(pred))[0][0]
                if round(pd, PD_DECIMALS) >= self.tau and self.meets(req, snr[0], cells[0], cov):
                    if share is None or pd > share[0]:
                        share = (pd, gi, owner, dprev, pred, e, float(snr[0]), float(cells[0]))
            if share is not None:
                pd, gi, owner, dprev, pred, e, snr, cells = share
                grid, curve = self.evidence_curve(dprev.weights, pred, e["n_re"], e["bw"],
                                                  e["clutter"])
                d = BeamDecision(
                    "reuse", grants[gi].t, grant=gi, ue_id=grants[gi].ue_id,
                    weights=dprev.weights, tag=dprev.tag, pd_pred=pd, pd_curve_sigma=grid,
                    pd_curve=curve, ue_loss_db=dprev.ue_loss_db, mcs=dprev.mcs, dse=dprev.dse,
                    cost_bits=0.0, n_re=dprev.n_re, bandwidth=dprev.bandwidth,
                    rcs_design_dbsm=r_k.design_dbsm, rcs_mean_dbsm=float(weights @ sigma),
                    info=dict(shared_with=owner, fallback="none", rcs_source=r_k.source,
                              degenerate_box=pred["degenerate_box"]),
                    pred_cell_snr_db=snr, exp_quality_cells=cells,
                    requirement_met=None if req is None else True,
                    resolution_bw=float(e["bw"]),
                    quality_prob=float(req.quality_prob(snr)) if req is not None else float("nan"),
                    pd_offset_curve=self.offset_curve(dprev.weights, pred, e["n_re"], e["bw"],
                                                      e["clutter"], sigma, weights),
                    **self._pred_fields(pred))
                d.parts = [self._single_part(d)]
                self._annotate(d, pred, [(dprev.weights, e["n_re"], e["clutter"])], e["bw"],
                               sigma, weights)
                out.append(d)
                continue
            free = [i for i in range(len(grants)) if i not in taken]
            d = self.plan(tr, [grants[i] for i in free], r_k, t, requirement=req, on_unmet=on_unmet,
                          objective=objective)
            if d.grant is not None:
                d.grant = free[d.grant]
            if d.mode == "reuse":
                d.parts = [self._single_part(d)]
                taken[d.grant] = (k, d)
            d.info["shared_with"] = None
            out.append(d)
        return out

    # -- multi-grant looks ----------------------------------------------------------------------
    def _part(self, g, i, e, c):
        return dict(grant=i, ue_id=g.ue_id, tag=int(e["tag"][c]), weights=e["W"][c],
                    n_re=int(e["n_re"]), bandwidth=float(e["grant_bw"]),
                    loss_db=float(e["loss"][c]), mcs=e["mcs"][c][0], dse=float(e["dse"][c]),
                    cost_bits=float(e["n_re"] * e["dse"][c]))

    def _mean_leak(self, W, nodes, n_re, bw, clutter):
        """UE leakage averaged over the clutter quadrature, per beam, node and cell."""
        if clutter is None:
            return np.zeros((len(W),) + nodes.u.shape)
        cu, cr, cs, ct = clutter[:4]
        cu = np.atleast_1d(cu)
        cw = np.asarray(clutter[4]) if len(clutter) > 4 else np.ones(len(cu)) / len(cu)
        cp = 10 ** (V.rc_snr_db(array_gain(W, cu), cu, cr, cs, ct, n_re) / 10)
        resp = V.rx_rel(nodes.u[None] - cu[:, None, None]) * delay_power(nodes.r - cr, bw)
        return np.einsum("bq,q,qnk->bnk", cp, cw, resp)

    def _combined(self, rho, leak, nodes, sigma, weights, s_q):
        """P_D, strongest-cell SNR (dB) and cell count for summed cell SNRs: rho, leak (B, N, K)."""
        lp = math.log(V.P["pfa"])
        sinr = rho / (1 + leak)
        snr = 10 * np.log10(np.maximum(sinr, 1e-30)).max(axis=2) @ nodes.w + float(weights @ sigma)
        pd = np.zeros(len(rho))
        cells = np.zeros(len(rho)) if s_q is not None else np.full(len(rho), np.nan)
        for x, w in zip(10 ** (sigma / 10), weights):
            pd += w * (cell_pd(x * rho, leak, lp).max(axis=2) @ nodes.w)
            if s_q is not None:
                cells += w * ((10 * np.log10(np.maximum(x * sinr, 1e-30)) >= s_q).sum(axis=2)
                              @ nodes.w)
        return pd, snr, cells

    @_in_site
    def plan_multi(self, track, grants, rcs=None, t=None, requirement=None, max_grants=3,
                   on_unmet=None, cost_eps_bits=1.0, objective=None):
        """One sensing look over up to ``max_grants`` grants of the frame (multi-grant).

        Each selected grant illuminates the target with its own candidate beam on its own REs,
        and the look integrates all of them in one CPI. Cell SNRs add over grants (energy
        addition, i.e. maximal-ratio combining over REs at the hypothesised cell):
        SNR_cell = sum_i SNR_i(W_i, n_re_i). UE leakage is added per grant, averaged over each
        grant's UE-direction quadrature. P_D (any cell, pose- and RCS-averaged) and the quality
        terms use the combined values; the cost is sum_i n_re_i * dSE_i, each grant within its
        own MCS budget.

        Selection: if the single-grant ``plan`` already meets P_D >= tau and the requirement, its
        decision is returned (so ``max_grants=1`` is ``plan``). Otherwise a greedy growth runs
        from several seeds: each grant's best-SNR candidate, its best zero-cost candidate and
        its cheapest feasible one (the best single grant is always a seed). A grant is added
        only if the combined strongest-cell SNR strictly rises and P_D does not fall (UE
        leakage can make an addition harmful); a feasible addition wins at least total cost,
        else zero-cost candidates rank by SNR gain and costed ones by gain per bit. Feasible
        sets are pruned (most expensive part first). The cheapest feasible set over all seeds
        is returned; with none, the highest-SNR set, which is never worse than the best single
        grant. ``info["steps_snr_db"]`` gives the combined SNR after each addition.
        ``cost_eps_bits`` is kept for compatibility and no longer used.
        ``objective`` "coverage" / "quality" grow sets that maximise pooled coverage
        (sum n_re*G / sum n_re*peak per point, worst point) or expected detected cells, so grants
        whose beams cover different parts of the inflated box can be combined; the
        best-scoring feasible set is returned (cost is a tie-break, not minimised).
        ``info["steps_score"]`` traces the objective score.
        If no set within ``max_grants`` is feasible, ``on_unmet`` decides: "dedicated" (dedicated
        beam), "reuse_pd" (single-grant cheapest P_D-feasible reuse), "best_quality" (the
        greedy set, flagged unmet).

        All grants share one pose prediction at ``t`` (default: the first grant's time). The
        range-cell coupling uses ``SiteConfig.sensing_bandwidth_hz`` when set, else the summed
        grant bandwidths. Returns a BeamDecision with ``mode="reuse_multi"`` and ``parts``
        (grant, ue_id, tag, weights, n_re, bandwidth, loss_db, mcs, dse, cost_bits); ``weights``
        and ``grant`` are None for a multi-grant look. A one-grant result has ``mode="reuse"``.
        """
        objective = self.objective if objective is None else objective
        if objective not in OBJECTIVES:
            raise ValueError(objective)
        on_unmet = self.on_unmet if on_unmet is None else on_unmet
        rcs = self.default_rcs(track) if rcs is None else rcs
        req = requirement if requirement is not None else getattr(track, "requirement", None)
        req = req if req is not None and req.active() else None
        if objective == "pd" or max_grants <= 1 or len(grants) <= 1:
            single = self.plan(track, grants, rcs, t, requirement=req, on_unmet="dedicated",
                               objective=objective)
            met = single.mode == "reuse" and (req is None or single.requirement_met)
            if met or max_grants <= 1 or len(grants) <= 1:
                d = single if met else self.plan(track, grants, rcs, t, requirement=req,
                                                 on_unmet=on_unmet, objective=objective)
                d.info["multi"] = False
                if d.mode == "reuse":
                    d.parts = [self._single_part(d)]
                return d
        s_q = req.quality_snr_db if req is not None else None
        sigma, weights = self.rcs_nodes(rcs)
        t0 = grants[0].t if t is None else t
        pred = self.prediction(track, t0)
        nodes = pred["nodes"] if "nodes" in pred else self._evidence_nodes(pred)
        res = self.site.sensing_bandwidth_hz if self.site is not None else None
        bw = res if res is not None else float(sum(g.n_prb * 12 * V.P["df"] for g in grants))
        pts = self.coverage_points(pred)
        pool = []
        stage_tags = sorted(set().union(*self.stages))
        for i, g in enumerate(grants):
            e = self.evaluate_grant(g, pred)
            cs = np.flatnonzero(e["ok"] & np.isin(e["tag"], stage_tags))
            if not cs.size:
                continue
            W = e["W"][cs]
            pool.append(dict(i=i, g=g, e=e, cs=cs, cost=e["n_re"] * e["dse"][cs], n=e["n_re"],
                             rho=cell_snr(W, nodes, e["n_re"], bw),
                             leak=self._mean_leak(W, nodes, e["n_re"], bw, e["clutter"]),
                             G=array_gain(W, pts[0]), peak=array_gain(W, PEAK_U).max(axis=1)))

        def metrics(rho, leak, gsum, psum):
            """P_D, strongest-cell SNR, quality cells, pooled coverage (dB), detected cells."""
            pd, snr, cells = self._combined(rho, leak, nodes, sigma, weights, s_q)
            cov = 10 * np.log10(np.maximum((gsum / psum[:, None]).min(axis=1), 1e-30))
            det = self.detected_cells(rho, leak, nodes, sigma, weights)
            return pd, snr, cells, cov, det

        def ok(pd, snr, cells, cov):
            return np.array([round(float(p), PD_DECIMALS) >= self.tau and self.meets(req, s, c, v)
                             for p, s, c, v in zip(pd, snr, cells, cov)])

        def score(snr, cov, det):
            return {"pd": snr, "coverage": cov, "quality": det}[objective]

        def cost_of(chosen):
            return float(sum(pool[a]["cost"][b] for a, b in chosen))

        def sums(chosen):
            rho = sum(pool[a]["rho"][b] for a, b in chosen)
            leak = sum(pool[a]["leak"][b] for a, b in chosen)
            gsum = sum(pool[a]["n"] * pool[a]["G"][b] for a, b in chosen)
            psum = sum(pool[a]["n"] * pool[a]["peak"][b] for a, b in chosen)
            return rho, leak, gsum, psum

        def evaluate(chosen):
            rho, leak, gsum, psum = sums(chosen)
            pd, snr, cells, cov, det = metrics(rho[None], leak[None], np.atleast_2d(gsum),
                                               np.atleast_1d(psum))
            f = bool(ok(pd, snr, cells, cov)[0])
            return float(pd[0]), float(snr[0]), float(score(snr, cov, det)[0]), f

        def grow(seed):
            """Greedy growth from one part. A grant is added only if the objective score
            (strongest-cell SNR for "pd", pooled coverage for "coverage", detected cells for
            "quality") strictly rises and P_D does not fall. For "pd" a feasible addition wins
            at least total cost and growth stops when feasible; otherwise zero-cost candidates
            rank by score gain, costed ones by gain per bit. The other objectives keep adding
            the largest score gain (cheaper first on ties) up to ``max_grants``."""
            chosen = [seed]
            pd0, snr0, sc0, feas = evaluate(chosen)
            steps, steps_score = [snr0], [sc0]
            while len(chosen) < max_grants and not (feas and objective == "pd"):
                used = {k for k, _ in chosen}
                rho, leak, gsum, psum = sums(chosen)
                best_feas, best_free, best_cost, best_any = None, None, None, None
                for k, P in enumerate(pool):
                    if k in used:
                        continue
                    pd, snr, cells, cov, det = metrics(
                        rho[None] + P["rho"], leak[None] + P["leak"], gsum[None] + P["n"] * P["G"],
                        psum + P["n"] * P["peak"])
                    sc = score(snr, cov, det)
                    gain = sc - sc0
                    min_gain = 1e-9 if objective == "pd" else OBJECTIVE_RESOLUTION[objective]
                    better = (gain > min_gain) & (np.round(pd, PD_DECIMALS) >= round(pd0, PD_DECIMALS))
                    total = cost_of(chosen) + P["cost"]
                    if objective != "pd":
                        res_ = OBJECTIVE_RESOLUTION[objective]
                        for jj in np.flatnonzero(better):
                            key = (-round(gain[jj] / res_), total[jj], k, jj)
                            if best_any is None or key < best_any[0]:
                                best_any = (key, k, jj)
                        continue
                    for jj in np.flatnonzero(better & ok(pd, snr, cells, cov)):
                        key = (total[jj], -snr[jj], k, jj)
                        if best_feas is None or key < best_feas[0]:
                            best_feas = (key, k, jj)
                    free = np.flatnonzero(better & (P["cost"] == 0))
                    if free.size:
                        jj = free[np.argmax(gain[free])]
                        if best_free is None or gain[jj] > best_free[0]:
                            best_free = (gain[jj], k, jj)
                    paid = np.flatnonzero(better & (P["cost"] > 0))
                    if paid.size:
                        ratio = gain[paid] / P["cost"][paid]
                        jj = paid[np.argmax(ratio)]
                        if best_cost is None or ratio.max() > best_cost[0]:
                            best_cost = (ratio.max(), k, jj)
                pick = best_any if objective != "pd" else (best_feas or best_free or best_cost)
                if pick is None:
                    break
                _, k, jj = pick
                chosen.append((k, jj))
                pd0, snr0, sc0, feas = evaluate(chosen)
                steps.append(snr0)
                steps_score.append(sc0)
            if objective == "pd" and feas and len(chosen) > 1:  # drop unneeded, expensive first
                for kj in sorted(chosen, key=lambda kj: -pool[kj[0]]["cost"][kj[1]]):
                    rest = [x for x in chosen if x != kj]
                    if evaluate(rest)[3]:
                        chosen = rest
            return chosen, steps, steps_score

        seeds = set()
        for k, P in enumerate(pool):
            pd, snr, cells, cov, det = metrics(P["rho"], P["leak"], P["n"] * P["G"],
                                               P["n"] * P["peak"])
            sc = score(snr, cov, det)
            seeds.add((k, int(np.argmax(sc))))
            free = np.flatnonzero(P["cost"] == 0)
            if free.size:
                seeds.add((k, int(free[np.argmax(sc[free])])))
            f = np.flatnonzero(ok(pd, snr, cells, cov))
            if f.size:
                seeds.add((k, int(f[np.lexsort((-sc[f], P["cost"][f]))[0]])))
        results = []
        for seed in sorted(seeds):
            chosen, steps, steps_score = grow(seed)
            _, snr_c, sc_c, feas = evaluate(chosen)
            results.append((feas, cost_of(chosen), sc_c, chosen, steps, steps_score))
        feasible_sets = [r for r in results if r[0]]
        if feasible_sets and objective == "pd":
            feasible, _, _, chosen, steps, steps_score = min(
                feasible_sets, key=lambda r: (r[1], -r[2], len(r[3])))
        elif feasible_sets:
            res_ = OBJECTIVE_RESOLUTION[objective]
            feasible, _, _, chosen, steps, steps_score = max(
                feasible_sets, key=lambda r: (round(r[2] / res_), -r[1], -len(r[3])))
        elif results:
            res_ = OBJECTIVE_RESOLUTION.get(objective, 1e-9)
            feasible, _, _, chosen, steps, steps_score = max(
                results, key=lambda r: (round(r[2] / res_), -r[1], -len(r[3])))
        else:
            feasible, chosen, steps, steps_score = False, [], [], []
        if not chosen or (not feasible and on_unmet != "best_quality"):
            d = self.plan(track, grants, rcs, t, requirement=req, on_unmet=on_unmet,
                          objective=objective)
            d.info["multi"] = False
            if d.mode == "reuse":
                d.parts = [self._single_part(d)]
            return d
        rho = sum(pool[a]["rho"][b] for a, b in chosen)
        leak = sum(pool[a]["leak"][b] for a, b in chosen)
        pd, snr, cells = (v[0] for v in self._combined(rho[None], leak[None], nodes, sigma,
                                                         weights, s_q))
        parts = [self._part(pool[a]["g"], pool[a]["i"], pool[a]["e"], pool[a]["cs"][b])
                 for a, b in chosen]
        grid = curve_sigma()
        curve = self._multi_curve(parts, pool, chosen, nodes, bw, pred, grid)
        offset_pd = np.array([self._combined(xo * rho[None], leak[None], nodes, sigma, weights,
                                             None)[0][0] for xo in 10 ** (PD_OFFSETS_DB / 10)])
        multi = len(parts) > 1
        d = BeamDecision(
            "reuse_multi" if multi else "reuse",
            t0,
            grant=None if multi else parts[0]["grant"],
            ue_id=None if multi else parts[0]["ue_id"],
            weights=None if multi else parts[0]["weights"],
            tag=None if multi else parts[0]["tag"],
            pd_pred=float(pd),
            pd_curve_sigma=grid,
            pd_curve=curve,
            ue_loss_db=max(p["loss_db"] for p in parts),
            mcs=None if multi else parts[0]["mcs"],
            dse=float(sum(p["dse"] for p in parts)),
            cost_bits=float(sum(p["cost_bits"] for p in parts)),
            n_re=int(sum(p["n_re"] for p in parts)),
            bandwidth=float(sum(p["bandwidth"] for p in parts)),
            rcs_design_dbsm=rcs.design_dbsm,
            rcs_mean_dbsm=float(weights @ sigma),
            n_candidates=int(sum(len(P["cs"]) for P in pool)),
            info=dict(multi=True, fallback="none" if feasible else "best_quality",
                      rcs_source=rcs.source, rcs_nodes=sigma.tolist(),
                      rcs_weights=weights.tolist(), degenerate_box=pred["degenerate_box"],
                      leakage="mean over UE quadrature per grant", objective=objective,
                      steps_snr_db=[float(x) for x in steps],
                      steps_score=[float(x) for x in steps_score], seeds=len(seeds)),
            pred_cell_snr_db=float(snr),
            exp_quality_cells=float(cells),
            requirement_met=None if req is None else bool(feasible),
            resolution_bw=float(bw),
            quality_prob=float(req.quality_prob(snr)) if req is not None else float("nan"),
            parts=parts,
            pd_offset_curve=(PD_OFFSETS_DB.copy(), offset_pd),
            **self._pred_fields(pred),
        )
        ann = [(pool[a]["e"]["W"][pool[a]["cs"][b]], pool[a]["n"], pool[a]["e"]["clutter"])
               for a, b in chosen]
        return self._annotate(d, pred, ann, bw, sigma, weights)

    @staticmethod
    def _single_part(d):
        return dict(grant=d.grant, ue_id=d.ue_id, tag=d.tag, weights=d.weights, n_re=d.n_re,
                    bandwidth=d.bandwidth, loss_db=d.ue_loss_db, mcs=d.mcs, dse=d.dse,
                    cost_bits=d.cost_bits)

    def _multi_curve(self, parts, pool, chosen, nodes, bw, pred, grid):
        """Designated-cell crossing curve of a multi-grant look (summed cell SNR and leakage)."""
        lp = math.log(V.P["pfa"])
        model = grid - self.snr_offset()
        if self.evidence_cell == "event":
            rho = sum(pool[a]["rho"][b] for a, b in chosen)
            leak = sum(pool[a]["leak"][b] for a, b in chosen)
            off, wo = self.calibration_nodes()
            return np.array([self._combined(10 ** (g / 10) * rho[None], leak[None], nodes, off,
                                            wo, None)[0][0] for g in grid])
        if self.evidence_cell == "predicted":
            rho = sum(focus_snr(pool[a]["e"]["W"][[pool[a]["cs"][b]]], nodes, pool[a]["e"]["n_re"],
                                bw, pred["u_hat"], pred["r_hat"])[0] for a, b in chosen)
            leak = 0.0
            for a, b in chosen:
                cl = pool[a]["e"]["clutter"]
                if cl is not None:
                    W = pool[a]["e"]["W"][[pool[a]["cs"][b]]]
                    cu = np.atleast_1d(cl[0])
                    cw = np.asarray(cl[4]) if len(cl) > 4 else np.ones(len(cu)) / len(cu)
                    cp = 10 ** (V.rc_snr_db(array_gain(W, cu), cu, cl[1], cl[2], cl[3],
                                            pool[a]["e"]["n_re"]) / 10)[0]
                    leak += float(cw @ (cp * V.rx_rel(pred["u_hat"] - cu)
                                        * delay_power(pred["r_hat"] - cl[1], bw)))
        else:
            rho = sum(pool[a]["rho"][b][:, nodes.dom] for a, b in chosen)
            leak = sum(pool[a]["leak"][b][:, nodes.dom] for a, b in chosen)
        x = 10 ** (model / 10)
        return cell_pd(x[:, None] * rho[None, :], leak, lp) @ nodes.w

    @staticmethod
    def _table(rows, pairs):
        """Admissible candidates as arrays (grant, ue_id, tag, loss_db, mcs, dse, cost_bits, pd,
        pred_cell_snr_db, exp_quality_cells, weights). pd = -1 means screened out (below tau)."""
        cols = {k: [] for k in ("grant", "ue_id", "tag", "loss_db", "mcs", "dse", "cost_bits", "pd",
                                "pred_cell_snr_db", "exp_quality_cells", "weights")}
        for i, c in pairs:
            _, g, _, e = rows[i]
            for k, v in (("grant", i), ("ue_id", g.ue_id), ("tag", int(e["tag"][c])),
                         ("loss_db", float(e["loss"][c])), ("mcs", e["mcs"][c][0]),
                         ("dse", float(e["dse"][c])), ("cost_bits", float(e["n_re"] * e["dse"][c])),
                         ("pd", float(e["pd"][c])), ("pred_cell_snr_db", float(e["q_snr"][c])),
                         ("exp_quality_cells", float(e["q_cells"][c])), ("weights", e["W"][c])):
                cols[k].append(v)
        return {k: np.array(v) for k, v in cols.items()}

    @_in_site
    def dedicated(self, track, grants, rcs=None, t=None, n_cand=0, n_eval=0, requirement=None):
        rcs = self.default_rcs(track) if rcs is None else rcs
        t = (grants[0].t if grants else track.t) if t is None else t
        req = requirement if requirement is not None else getattr(track, "requirement", None)
        req = req if req is not None and req.active() else None
        sigma, weights = self.rcs_nodes(rcs)
        pred = self.prediction(track, t)
        n_re, grant_bw = V.P["ded_prb"] * 12 * V.P["ded_sym"], V.P["ded_prb"] * 12 * V.P["df"]
        bw = self.resolution_bw(grant_bw)
        width = None
        if self.dedicated_covering:
            # sub-array widths M, M/2, ... down to 4 elements, steered at the predicted centre;
            # the width with the largest absolute worst-case gain over the inflated box wins
            u_c = V.to_polar(pred["x"][None, :2], self.rsu, self.bore, V.P["rsu_h"],
                             V.P["rc_h"])[0][0]
            widths = [w_ for w_ in (V.M, V.M // 2, V.M // 4, V.M // 8) if w_ >= 4]
            beams = np.array([np.exp(-1j * np.pi * V.N_IDX * u_c)
                              * (np.abs(V.N_IDX) < w_ / 2) / math.sqrt(w_) for w_ in widths])
            gain = self.coverage_beams(beams, self.coverage_points(pred))[1]
            width = widths[int(np.argmax(gain))]
            Wd = beams[[int(np.argmax(gain))]]
        else:
            Wd = np.array([V.beam(pred["u_hat"]), V.beam(pred["u_hat"], V.P["q_wide"])])
        pd = self.detect(Wd, pred, n_re, bw, None, sigma, weights)
        sizing = None
        if self.site is not None and self.site.dedicated_sizing:
            # RE ladder: slow-time doubling at the fixed dedicated PRB width, then the CPI cap.
            # Energy scales with n_re (no clutter on a dedicated look), so each step is an RCS
            # shift of 10 log10(n_k / n_base) dB on the same detector.
            cap = int(self.site.dedicated_cap_re or 32768)
            ladder = [n_re * 2 ** k for k in range(32) if n_re * 2 ** k < cap] + [cap]
            ladder = [n for n in ladder if n >= n_re] or [n_re]
            gain = 10 * np.log10(np.array(ladder, float) / n_re)
            shifted = self.pd_shifts(Wd, pred, n_re, bw, None,
                                     (sigma[:, None] + gain[None, :]).ravel())
            pd_k = np.einsum("s,skb->kb", weights, shifted.reshape(len(sigma), len(ladder), -1))
            ok = np.flatnonzero(pd_k.max(axis=1) >= self.tau)
            step = int(ok[0]) if ok.size else len(ladder) - 1
            n_re, pd = ladder[step], pd_k[step]
            sizing = dict(dedicated_n_re=int(n_re), dedicated_step=step,
                          dedicated_ladder=[int(n) for n in ladder],
                          dedicated_cap_hit=not ok.size)
        c = int(np.argmax(pd))
        grid, curve = self.evidence_curve(Wd[c], pred, n_re, bw, None)
        snr, cells = self.quality(Wd[c], pred, n_re, bw, None, sigma, weights,
                                  req.quality_snr_db if req is not None else None)
        cov = self.coverage_beams(Wd[c], self.coverage_points(pred))[0][0]
        se_ref = (np.mean([next(m for m in V.MCS if m[0] == g.mcs)[2] for g in grants])
                  if grants else 0.0)
        d = BeamDecision(
            "dedicated",
            t,
            weights=Wd[c],
            tag=-1 - c if width is None else -100 - width,
            pd_pred=float(pd[c]),
            pd_curve_sigma=grid,
            pd_curve=curve,
            cost_bits=float(n_re * se_ref),
            n_re=n_re,
            bandwidth=grant_bw,
            rcs_design_dbsm=rcs.design_dbsm,
            rcs_mean_dbsm=float(weights @ sigma),
            n_candidates=n_cand,
            info=dict(meets_tau=bool(pd[c] >= self.tau), rcs_source=rcs.source,
                      n_evaluated=n_eval, rcs_nodes=sigma.tolist(), rcs_weights=weights.tolist(),
                      degenerate_box=pred["degenerate_box"]),
            pred_cell_snr_db=float(snr[0]),
            exp_quality_cells=float(cells[0]),
            requirement_met=None if req is None else self.meets(req, snr[0], cells[0], cov),
            resolution_bw=float(bw),
            quality_prob=float(req.quality_prob(snr[0])) if req is not None else float("nan"),
            pd_offset_curve=self.offset_curve(Wd[c], pred, n_re, bw, None, sigma, weights),
            **self._pred_fields(pred),
        )
        if width is not None:
            d.info["dedicated_width"] = width
        if sizing is not None:
            d.info.update(sizing)
        return self._annotate(d, pred, [(Wd[c], n_re, None)], bw, sigma, weights)

    # -- evidence for the RCS estimator -------------------------------------------------------
    @_in_site
    def rcs_look(self, d, rep, track):
        """RcsLook from an executed decision, its SensingReport and the updated TrackReport.

        The evidence is the designated cell's crossing and power (``rep.rcs_*``), on every look:
        ``rcs_stat`` is the cell power over the reported floor (linear, the scale of
        eta = -ln(pfa)); ``rcs_floor_db`` = 10 log10(floor / thermal noise). The cell's unit-RCS
        SNR over that floor is integrated over the posterior pose: cell quadrature of ``track.P``
        and heading for a box, Gaussian direction nodes for a point target (also used when no
        box face is visible). With ``evidence_cell="predicted"`` the cell is fixed at the
        decision's (u_hat, r_hat). The site's ``snr_offset_db`` is removed, so the posterior is
        in calibrated RCS. Returns None when the back end reported no designated-cell evidence.

        With ``evidence_cell="event"`` the evidence is the look's detection event alone
        (``rep.rcs_detected`` if given, else ``rep.detected``: a detection inside the track's
        gate) against the decision-time curve ``d.pd_curve``; the look has no unit nodes, so
        the estimator uses its event term only (``rcs_likelihood="event+truncated"``).
        """
        if self.evidence_cell == "event":
            hit = rep.rcs_detected if rep.rcs_detected is not None else rep.detected
            return RcsLook(d.t, bool(hit), d.pd_curve_sigma, d.pd_curve)
        if rep.rcs_detected is None:
            return None
        look = RcsLook(d.t, bool(rep.rcs_detected), d.pd_curve_sigma, d.pd_curve)
        floor = float(rep.rcs_floor_db or 0.0) + self.snr_offset()
        bw = d.resolution_bw if np.isfinite(d.resolution_bw) else d.bandwidth
        extent = self.track_extent(track)
        box = extent is not None
        if box:
            h0 = track.heading if track.heading is not None else heading_of(track.x, track.P)[0]
            box = box_scatterers(track.x[:2], h0, *extent, self.rsu, 0.0)[3] is not None
        if box:
            if track.heading is not None:
                h, sh = track.heading, track.heading_std or 0.0
            else:
                h, sh = heading_of(track.x, track.P)
            nodes = pose_nodes(track.x[:2], track.P[:2, :2], h, sh, extent, self.rsu,
                               self.bore, 0.0)
        else:
            u, r, th, _ = V.to_polar(track.x[None, :2], self.rsu, self.bore, V.P["rsu_h"],
                                     V.P["rc_h"])
            J = polar_jacobian(track.x[:2], self.rsu, self.bore)[0]
            nodes = self.point_nodes(u[0], r[0], th[0], math.sqrt(float(J @ track.P[:2, :2] @ J)))
        looks = ([(p["weights"], p["n_re"]) for p in d.parts] if d.parts and d.weights is None
                 else [(d.weights, d.n_re)])
        if self.evidence_cell == "predicted":
            unit = sum(focus_snr(W[None], nodes, n, bw, d.u_hat, d.r_hat)[0] for W, n in looks)
        else:
            unit = sum(cell_snr(W[None], nodes, n, bw)[0, :, nodes.dom] for W, n in looks)
        w = nodes.w
        keep = w > 1e-9 * w.max()
        look.eta = float(-math.log(V.P["pfa"]))
        look.stat = float(rep.rcs_stat) if rep.rcs_detected else None
        look.unit_db = 10 * np.log10(np.maximum(unit[keep], 1e-300)) - floor
        look.unit_w = w[keep] / w[keep].sum()
        return look

    @staticmethod
    def _pred_fields(pred):
        return dict(u_hat=pred["u_hat"], r_hat=pred["r_hat"], sigma_u=pred["s_u"], s_r=pred["s_r"])
