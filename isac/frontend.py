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
"""

import math
import numpy as np
from scipy.special import ndtri
from . import model as V
from .detection import predictive_pd
from .interfaces import BeamDecision
from .quadrature import UELossCache
from .rcs import RcsLook
from .target import FP, box_scatterers, pose_nodes, extended_pd_shifts, cell_snr, array_gain
from .tracker import heading_of, predict_state, polar_jacobian

SEQUENTIAL = (11, 13)
PD_DECIMALS = 12


def curve_sigma():
    lo, hi, step = FP["rcs_curve_db"]
    return np.arange(lo, hi + step / 2, step)


class BeamFrontEnd:
    def __init__(self, rsu, bore, target_model="extended", tau=0.9, levels=SEQUENTIAL,
                 ue_vehicle_rcs_dbsm=0.0, rcs_mode=None, lazy=True):
        if target_model not in ("extended", "point"):
            raise ValueError(target_model)
        self.rcs_mode = FP["rcs_mode"] if rcs_mode is None else rcs_mode
        if self.rcs_mode not in ("predictive", "design"):
            raise ValueError(self.rcs_mode)
        self.rsu, self.bore = np.asarray(rsu, float), float(bore)
        self.target_model, self.tau, self.lazy = target_model, tau, lazy
        self.ue_rcs = ue_vehicle_rcs_dbsm
        lvl_tags, self.cand_level = V.levels()
        self.stages = [tuple(lvl_tags[i]) for i in levels]
        self.ue_loss = UELossCache()
        x, w = np.polynomial.legendre.leggauss(V.P["n_ue_clut"])
        self._clut_x, self._clut_w = x, w / 2
        n = V.P["n_ue_q"]
        self._cell_q = 2 * (np.arange(n) + 0.5) / n - 1

    # -- tracker side -------------------------------------------------------------------------
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
        extent = track.extent
        if extent is not None:
            keys, xy, _, dom, _ = box_scatterers(x[:2], h, *extent, self.rsu, 0.0)
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
        )
        if self.target_model == "extended" and extent is not None:
            pred["nodes"] = pose_nodes(x[:2], P[:2, :2], h, sh, extent, self.rsu, self.bore, 0.0)
        return pred

    def rcs_nodes(self, rcs):
        """RCS values (dBsm) and weights the predicted P_D is averaged over."""
        if self.rcs_mode == "predictive" and rcs.nodes_dbsm is not None and len(rcs.nodes_dbsm):
            sig = np.asarray(rcs.nodes_dbsm, float)
            w = np.ones(len(sig)) if rcs.weights is None else np.asarray(rcs.weights, float)
            if not (w.shape == sig.shape and np.all(w >= 0) and w.sum() > 0):
                raise ValueError("RcsReport weights must be non-negative, match nodes and sum > 0")
            return sig, w / w.sum()
        return np.array([rcs.design_dbsm], float), np.ones(1)

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

        With an extent, the cell is the dominant scatterer's and the curve uses the box pose
        nodes, whatever the decision model. Without an extent it is the point detector.
        """
        grid = curve_sigma()
        if pred["extent"] is None:
            return grid, self.pd_shifts(W, pred, n_re, bandwidth, clutter, grid)[:, 0]
        nodes = pred.get("nodes")
        if nodes is None:
            nodes = pose_nodes(pred["x"][:2], pred["P"][:2, :2], pred["heading"],
                               pred["heading_std"], pred["extent"], self.rsu, self.bore, 0.0)
        pd = extended_pd_shifts(np.atleast_2d(W), nodes, n_re, bandwidth, clutter,
                                focus="dominant", shifts_db=grid)[0]
        return grid, pd[:, 0]

    def detect(self, W, pred, n_re, bandwidth, clutter, sigma, weights, threshold=None):
        """RCS-averaged predicted P_D per beam (-1 for beams screened out by ``threshold``)."""
        screen = None if threshold is None else (threshold, weights)
        pd = self.pd_shifts(W, pred, n_re, bandwidth, clutter, sigma, screen)
        return np.where((pd < 0).any(axis=0), -1.0, weights @ pd)

    # -- scheduler side -----------------------------------------------------------------------
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
        uq = g0 + self._cell_q / 64
        return (g0, g0 - 1 / 64, g0 + 1 / 64), uq, g0 + self._clut_x / 64, self._clut_w

    def clutter(self, g):
        if not V.P["ue_clutter"]:
            return None
        _, _, cu, cw = self.ue_knowledge(g)
        sig_ue = V.P["sigma_ped_dbsm"] if g.ue_kind != "vehicle" else self.ue_rcs
        return (cu, g.ue_range_m, sig_ue, g.ue_theta_deg, cw)

    def evaluate_grant(self, g, pred):
        """Admissible candidates of one grant with loss and MCS cost; P_D is filled lazily."""
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
        return dict(W=W, tag=tag, loss=loss, ok=ok, pd=np.full(len(W), np.nan), dse=m0[2] - se,
                    n_re=g.n_prb * 12 * g.n_sym, bw=g.n_prb * 12 * V.P["df"], mcs=mcs,
                    clutter=self.clutter(g))

    def _fill_pd(self, rows, pairs, sigma, weights):
        """Evaluate predicted P_D for the (row, candidate) pairs that have none yet."""
        todo = {}
        for i, c in pairs:
            if np.isnan(rows[i][3]["pd"][c]):
                todo.setdefault(i, []).append(c)
        for i, cs in todo.items():
            _, _, pred, e = rows[i]
            cs = np.array(sorted(cs))
            e["pd"][cs] = self.detect(e["W"][cs], pred, e["n_re"], e["bw"], e["clutter"],
                                      sigma, weights, threshold=self.tau)
        return sum(len(cs) for cs in todo.values())

    # -- decision -----------------------------------------------------------------------------
    def plan(self, track, grants, rcs, t=None):
        """BeamDecision for one sensing opportunity from a TrackReport, GrantReports and RcsReport."""
        sigma, weights = self.rcs_nodes(rcs)
        rows = []
        for i, g in enumerate(grants):
            pred = self.prediction(track, g.t)
            rows.append((i, g, pred, self.evaluate_grant(g, pred)))
        n_cand = sum(int(e["ok"].sum()) for *_, e in rows)
        n_eval = 0
        if not self.lazy:
            n_eval += self._fill_pd(rows, [(i, c) for i, _, _, e in rows
                                           for c in np.flatnonzero(e["ok"])], sigma, weights)
        for stage in self.stages:
            pool = {}
            for i, g, pred, e in rows:
                for c in np.flatnonzero(np.isin(e["tag"], stage) & e["ok"]):
                    pool.setdefault(e["n_re"] * e["dse"][c], []).append((i, c))
            for cost in sorted(pool):
                group = pool[cost]
                n_eval += self._fill_pd(rows, group, sigma, weights)
                q = {ic: round(float(rows[ic[0]][3]["pd"][ic[1]]), PD_DECIMALS) for ic in group}
                feasible = [ic for ic in group if q[ic] >= self.tau]
                if not feasible:
                    continue
                i, c = min(feasible, key=lambda ic: (-q[ic], rows[ic[0]][3]["loss"][ic[1]],
                                                     ic[0], ic[1]))
                _, g, pred, e = rows[i]
                W = e["W"][c]
                grid, curve = self.evidence_curve(W, pred, e["n_re"], e["bw"], e["clutter"])
                return BeamDecision(
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
                    bandwidth=float(e["bw"]),
                    rcs_design_dbsm=rcs.design_dbsm,
                    rcs_mean_dbsm=float(weights @ sigma),
                    n_candidates=n_cand,
                    info=dict(stage=stage, rcs_source=rcs.source, n_evaluated=n_eval,
                              rcs_nodes=sigma.tolist(), rcs_weights=weights.tolist()),
                    **self._pred_fields(pred),
                )
        return self.dedicated(track, grants, rcs, t, n_cand, n_eval)

    def dedicated(self, track, grants, rcs, t=None, n_cand=0, n_eval=0):
        t = (grants[0].t if grants else track.t) if t is None else t
        sigma, weights = self.rcs_nodes(rcs)
        pred = self.prediction(track, t)
        Wd = np.array([V.beam(pred["u_hat"]), V.beam(pred["u_hat"], V.P["q_wide"])])
        n_re, bw = V.P["ded_prb"] * 12 * V.P["ded_sym"], V.P["ded_prb"] * 12 * V.P["df"]
        pd = self.detect(Wd, pred, n_re, bw, None, sigma, weights)
        c = int(np.argmax(pd))
        grid, curve = self.evidence_curve(Wd[c], pred, n_re, bw, None)
        se_ref = (np.mean([next(m for m in V.MCS if m[0] == g.mcs)[2] for g in grants])
                  if grants else 0.0)
        return BeamDecision(
            "dedicated",
            t,
            weights=Wd[c],
            tag=-1 - c,
            pd_pred=float(pd[c]),
            pd_curve_sigma=grid,
            pd_curve=curve,
            cost_bits=float(n_re * se_ref),
            n_re=n_re,
            bandwidth=bw,
            rcs_design_dbsm=rcs.design_dbsm,
            rcs_mean_dbsm=float(weights @ sigma),
            n_candidates=n_cand,
            info=dict(meets_tau=bool(pd[c] >= self.tau), rcs_source=rcs.source,
                      n_evaluated=n_eval, rcs_nodes=sigma.tolist(), rcs_weights=weights.tolist()),
            **self._pred_fields(pred),
        )

    # -- evidence for the RCS estimator -------------------------------------------------------
    def rcs_look(self, d, rep, track):
        """RcsLook from an executed decision, its SensingReport and the updated TrackReport.

        The evidence is the designated cell's crossing and power (``rep.rcs_*``), on every look.
        Its unit-RCS SNR over the measured floor is integrated over the posterior pose: cell quadrature of ``track.P`` and heading for a
        box, Gauss-Hermite in direction for a point target. Returns None when the back end
        reported no designated-cell evidence.
        """
        if rep.rcs_detected is None:
            return None
        look = RcsLook(d.t, bool(rep.rcs_detected), d.pd_curve_sigma, d.pd_curve)
        floor = float(rep.rcs_floor_db or 0.0)
        if track.extent is not None:
            if track.heading is not None:
                h, sh = track.heading, track.heading_std or 0.0
            else:
                h, sh = heading_of(track.x, track.P)
            nodes = pose_nodes(track.x[:2], track.P[:2, :2], h, sh, track.extent, self.rsu,
                               self.bore, 0.0)
            unit = cell_snr(d.weights[None], nodes, d.n_re, d.bandwidth)[0, :, nodes.dom]
            w = nodes.w
        else:
            u, r, th, _ = V.to_polar(track.x[None, :2], self.rsu, self.bore, V.P["rsu_h"],
                                     V.P["rc_h"])
            s_u = math.sqrt(float(polar_jacobian(track.x[:2], self.rsu, self.bore)[0]
                                  @ track.P[:2, :2]
                                  @ polar_jacobian(track.x[:2], self.rsu, self.bore)[0]))
            x, w = np.polynomial.hermite_e.hermegauss(FP["ext_heading_nodes"] * 3)
            un = u[0] + s_u * x
            unit = 10 ** (V.rc_snr_db(array_gain(d.weights[None], un)[0], un, r[0], 0.0, th[0],
                                      d.n_re) / 10)
            w = w / w.sum()
        keep = w > 1e-9 * w.max()
        look.eta = float(-math.log(V.P["pfa"]))
        look.stat = float(rep.rcs_stat) if rep.rcs_detected else None
        look.unit_db = 10 * np.log10(np.maximum(unit[keep], 1e-300)) - floor
        look.unit_w = w[keep] / w[keep].sum()
        return look

    @staticmethod
    def _pred_fields(pred):
        return dict(u_hat=pred["u_hat"], r_hat=pred["r_hat"], sigma_u=pred["s_u"], s_r=pred["s_r"])
