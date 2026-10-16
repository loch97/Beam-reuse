"""Beam front end: chooses a sensing beam from tracker and scheduler reports.

For each sensing opportunity the front end

1. predicts the tracker's state to each grant's transmission time (the gap is the scheduler's
   actual timing, not a fixed ``gap_k``);
2. builds the paper's admissible candidates around the grant's serving beam and the predicted
   dominant-scatterer direction;
3. bounds the UE gain loss over the scheduler's UE-direction knowledge (SRS estimate, or the
   grid-beam cell under U1) and keeps beams whose loss fits the grant's measured SINR within the
   permitted MCS drop;
4. predicts P_D with the multi-scatterer model (or the paper's point model), using the RCS design
   value reported by the tracker, the grant's REs and bandwidth, and the UE as clutter;
5. returns the cheapest reuse that reaches ``tau``, trying the sector ladder before the two-lobe
   baseline (the paper's sequential policy). Without one it falls back to a dedicated sensing beam.
"""

import math
import numpy as np
from scipy.special import ndtri
from . import model as V
from .detection import predictive_pd
from .interfaces import BeamDecision
from .quadrature import UELossCache
from .target import FP, box_scatterers, pose_nodes, extended_pd
from .tracker import heading_of, predict_state, polar_jacobian

SEQUENTIAL = (11, 13)


class BeamFrontEnd:
    def __init__(self, rsu, bore, target_model="extended", tau=0.9, levels=SEQUENTIAL,
                 ue_vehicle_rcs_dbsm=0.0):
        if target_model not in ("extended", "point"):
            raise ValueError(target_model)
        self.rsu, self.bore = np.asarray(rsu, float), float(bore)
        self.target_model, self.tau = target_model, tau
        self.ue_rcs = ue_vehicle_rcs_dbsm
        lvl_tags, self.cand_level = V.levels()
        self.stages = [tuple(lvl_tags[i]) for i in levels]
        self.ue_loss = UELossCache()
        x, w = np.polynomial.legendre.leggauss(V.P["n_ue_clut"])
        self._clut_x, self._clut_w = x, w / 2
        n = V.P["n_ue_q"]
        self._cell_q = 2 * (np.arange(n) + 0.5) / n - 1

    # -- tracker side -------------------------------------------------------------------------
    def prediction(self, track, t, rcs_dbsm):
        """Predicted pose and the uncertainty the detector integrates over, at time ``t``."""
        x, P = predict_state(track.x, track.P, t - track.t)
        if track.heading is not None:
            h, sh = track.heading, track.heading_std or 0.0
        else:
            h, sh = heading_of(x, P)
        extent = track.extent
        if extent is not None:
            keys, xy, _, dom, _ = box_scatterers(x[:2], h, *extent, self.rsu, rcs_dbsm)
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
            rcs=rcs_dbsm,
        )
        if self.target_model == "extended" and extent is not None:
            pred["nodes"] = pose_nodes(
                x[:2], P[:2, :2], h, sh, extent, self.rsu, self.bore, rcs_dbsm
            )
        return pred

    def detect(self, W, pred, n_re, bandwidth, clutter=None, threshold=None):
        if "nodes" in pred:
            return extended_pd(W, pred["nodes"], n_re, bandwidth, clutter, threshold=threshold)[0]
        kw = dict(s_r=pred["s_r"], c_ur=pred["c_ur"])
        args = (W, pred["u_hat"], pred["r_hat"], pred["s_u"], pred["rcs"], pred["theta"], n_re)
        if clutter is None:
            return predictive_pd(*args, **kw)[0]
        return predictive_pd(*args, bandwidth, clutter, **kw)[0]

    # -- scheduler side -----------------------------------------------------------------------
    def ue_knowledge(self, g):
        """UE design window, gain-loss directions and clutter quadrature for a grant."""
        g0 = V.GRID[g.k0]
        if g.ue_u is not None and g.ue_u_std:
            s = g.ue_u_std
            uq = g.ue_u + s * ndtri((np.arange(FP["ue_dir_nodes"]) + 0.5) / FP["ue_dir_nodes"])
            cu = g.ue_u + s * math.sqrt(2) * np.polynomial.hermite.hermgauss(9)[0]
            cw = np.polynomial.hermite.hermgauss(9)[1] / math.sqrt(math.pi)
            return (g.ue_u, g.ue_u - 2 * s, g.ue_u + 2 * s), uq, cu, cw
        if g.ue_u is not None:
            return (g.ue_u, g.ue_u, g.ue_u), np.array([g.ue_u]), np.array([g.ue_u]), np.ones(1)
        uq = g0 + self._cell_q / 64
        return (g0, g0 - 1 / 64, g0 + 1 / 64), uq, g0 + self._clut_x / 64, self._clut_w

    def evaluate_grant(self, g, pred):
        """Admissible candidates of one grant with loss, MCS cost and predicted P_D."""
        ue_ctrl, uq, cu, cw = self.ue_knowledge(g)
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
        n_re, bw = g.n_prb * 12 * g.n_sym, g.n_prb * 12 * V.P["df"]
        pd = np.full(len(W), -1.0)
        if ok.any():
            sig_ue = V.P["sigma_ped_dbsm"] if g.ue_kind != "vehicle" else self.ue_rcs
            clut = (cu, g.ue_range_m, sig_ue, g.ue_theta_deg, cw) if V.P["ue_clutter"] else None
            idx = np.flatnonzero(ok)
            pd[idx] = self.detect(W[idx], pred, n_re, bw, clut, threshold=self.tau)
        mcs = [min(V.mcs_of(g.sinr_db - l, mmin), m0, key=lambda m: m[2]) if o else None
               for l, o in zip(loss, ok)]
        se = np.array([m[2] if m is not None else np.nan for m in mcs])
        return dict(W=W, tag=tag, loss=loss, ok=ok, pd=pd, dse=m0[2] - se, n_re=n_re, bw=bw,
                    mcs=mcs)

    # -- decision -----------------------------------------------------------------------------
    def plan(self, track, grants, rcs, t=None):
        """BeamDecision for one sensing opportunity from a TrackReport, GrantReports and RcsReport."""
        rows = []
        for i, g in enumerate(grants):
            pred = self.prediction(track, g.t, rcs.design_dbsm)
            rows.append((i, g, pred, self.evaluate_grant(g, pred)))
        n_cand = sum(int(e["ok"].sum()) for *_, e in rows)
        for stage in self.stages:
            best = None
            for i, g, pred, e in rows:
                hit = np.flatnonzero(np.isin(e["tag"], stage) & e["ok"] & (e["pd"] >= self.tau))
                for c in hit:
                    key = (e["n_re"] * e["dse"][c], -e["pd"][c], e["loss"][c], i)
                    if best is None or key < best[0]:
                        best = (key, i, g, pred, e, c)
            if best is not None:
                _, i, g, pred, e, c = best
                return BeamDecision(
                    "reuse",
                    g.t,
                    grant=i,
                    ue_id=g.ue_id,
                    weights=e["W"][c],
                    tag=int(e["tag"][c]),
                    pd_pred=float(e["pd"][c]),
                    ue_loss_db=float(e["loss"][c]),
                    mcs=e["mcs"][c][0],
                    dse=float(e["dse"][c]),
                    cost_bits=float(e["n_re"] * e["dse"][c]),
                    n_re=int(e["n_re"]),
                    bandwidth=float(e["bw"]),
                    rcs_design_dbsm=rcs.design_dbsm,
                    n_candidates=n_cand,
                    info=dict(stage=stage, rcs_source=rcs.source),
                    **self._pred_fields(pred),
                )
        return self.dedicated(track, grants, rcs, t, n_cand)

    def dedicated(self, track, grants, rcs, t=None, n_cand=0):
        t = (grants[0].t if grants else track.t) if t is None else t
        pred = self.prediction(track, t, rcs.design_dbsm)
        Wd = np.array([V.beam(pred["u_hat"]), V.beam(pred["u_hat"], V.P["q_wide"])])
        n_re, bw = V.P["ded_prb"] * 12 * V.P["ded_sym"], V.P["ded_prb"] * 12 * V.P["df"]
        pd = self.detect(Wd, pred, n_re, bw)
        c = int(np.argmax(pd))
        se_ref = np.mean([next(m for m in V.MCS if m[0] == g.mcs)[2] for g in grants]) if grants else 0.0
        return BeamDecision(
            "dedicated",
            t,
            weights=Wd[c],
            tag=-1 - c,
            pd_pred=float(pd[c]),
            cost_bits=float(n_re * se_ref),
            n_re=n_re,
            bandwidth=bw,
            rcs_design_dbsm=rcs.design_dbsm,
            n_candidates=n_cand,
            info=dict(meets_tau=bool(pd[c] >= self.tau), rcs_source=rcs.source),
            **self._pred_fields(pred),
        )

    @staticmethod
    def _pred_fields(pred):
        return dict(u_hat=pred["u_hat"], r_hat=pred["r_hat"], sigma_u=pred["s_u"], s_r=pred["s_r"])
