"""Compatibility and link-cost evaluation for the paper settings."""

from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from . import model as V

P = V.P
TAU = 0.9


def geometry():
    meta = json.loads((ROOT / "data/scene_frames.json").read_text())
    rsu = np.array(meta["rows"][0]["rsu_xy_m"], float)
    bore = math.radians(meta["rows"][0]["boresight_az_deg"])
    return (meta, rsu, bore)


def lane_points(rsu, bore, spacing, ymin=-80.0):
    """UE points on the lanes/sidewalk; rows at -80 + k*spacing (the step-1 rows), extended down to ymin."""
    xs = V.LANES_X
    ys = np.arange(-80 - spacing * math.floor((-80 - ymin) / spacing + 1e-09), -11, spacing)
    pts = np.array([(x, y) for x in xs for y in ys])
    u, r, th, cosb = V.to_polar(pts, rsu, bore, P["rsu_h"], P["ue_h"])
    keep = cosb > 0.1
    kind = np.array(["sidewalk" if x == 12.25 else "vehicle" for x, _ in pts])
    return (pts[keep], u[keep], r[keep], th[keep], kind[keep])


def compat_arrays(
    inr,
    sigma_x,
    spacing,
    recs,
    seed,
    extra=None,
    n_frames=None,
    eval_levels=None,
    ue_loss_fn=None,
    predictive_screen_fn=None,
    frame_hook=None,
):
    """Evaluate beam compatibility, link costs and frozen-decision stress scores.

    ``frame_hook(row, rsu, bore)``, when given, is called at the start of every frame so that a
    target model needing the frame's pose and class can configure itself.
    """
    P["inr"], P["sigma_x"] = (inr, sigma_x)
    meta, rsu, bore = geometry()
    if n_frames is not None:
        meta = dict(meta, rows=meta["rows"][:n_frames])
    lvl_tags, cand_level = V.levels()
    rep_rng = np.random.default_rng(seed + 7777)
    rng = np.random.default_rng(seed)
    sp, su, sr, sth, skind = lane_points(rsu, bore, spacing, V.cell_ymin(rsu))
    prb = rng.choice(P["prbs"], size=len(sp))
    n_re, bg = (prb * 12 * P["symbols"], prb * 12 * P["df"])
    sig_ue = np.where(skind == "vehicle", V.rcs_class_median(recs, "car"), P["sigma_ped_dbsm"])
    _, sr_rc, th_rc, _ = V.to_polar(sp, rsu, bore, P["rsu_h"], P["rc_h"])
    nf, nd, nl, ndm, ns = (
        len(meta["rows"]),
        P["n_draws"],
        len(lvl_tags),
        len(P["dm_list"]),
        len(sp),
    )
    C = np.zeros((nf, nd, nl, ndm, ns), bool)
    DSE = np.full(C.shape, np.nan)
    PDP = np.zeros(C.shape)
    LOSS = np.full(C.shape, np.nan)
    FDED = np.zeros((nf, nd), bool)
    SE0 = np.zeros((nf, nd, ns))
    X, XDED = ({}, {})
    TSEL = np.full(C.shape, -1, int)
    ue_model = P["ue_model"]
    if ue_model == "U1":
        _x, _w = np.polynomial.legendre.leggauss(P["n_ue_clut"])
        _nq = P["n_ue_q"]
        _tq = 2 * (np.arange(_nq) + 0.5) / _nq - 1

    def comm_state():
        g0, kk = V.ue_link(su, sr, sth, skind, sp, rsu, rng, bore)
        b = [V.mcs_of(g) for g in g0]
        return (g0, kk, b, np.array([m is not None and m[0] >= P["mcs_floor"] for m in b]))

    for f, row in enumerate(meta["rows"]):
        if frame_hook is not None:
            frame_hook(row, rsu, bore)
        keys, xy, sig_true, dom = V.vehicle_rcs(row, rsu, V.rcs_truth(recs, row["key"]))
        prior = V.rcs_prior(recs, V.aspect_bin(row["aspect_deg"]))
        u_rc, r_rc, th_rc_t, _ = V.to_polar(xy, rsu, bore, P["rsu_h"], P["rc_h"])
        truth = dict(u=u_rc, r=r_rc, sigma=sig_true, theta=th_rc_t)
        i = keys.index(dom)
        u_dom, r_dom, th_dom = (u_rc[i], r_rc[i], th_rc_t[i])
        sigma_u = sigma_x * math.sqrt(max(1 - u_dom**2, 0)) / r_dom
        s_r, c_ur = (None, None)
        true_unc, Jpar = (None, None)
        sigma_u, s_r, c_ur, Jpar = V.track_covariance(
            xy[i], rsu, bore, row["heading_rad"], with_J=True
        )
        Lc = np.linalg.cholesky(np.array([[sigma_u**2, c_ur], [c_ur, s_r**2]]))
        true_unc = (sigma_u, s_r, c_ur)
        ci = P.get("ctrl_inflate", 1.0)
        sigma_u, s_r, c_ur = (ci * sigma_u, ci * s_r, ci * ci * c_ur)
        tu = true_unc
        for d in range(nd):
            comm = comm_state()
            gamma0, k0, base, elig = comm
            SE0[f, d] = [m[2] if m is not None else 0.0 for m in base]
            rng.normal()
            zz = Lc @ rep_rng.normal(size=2)
            u_hat, r_hat = (u_dom + zz[0], r_dom + zz[1])
            Wd = [V.beam(u_hat), V.beam(u_hat, P["q_wide"])]
            Wd = np.array(Wd)
            nre_d = P["ded_prb"] * 12 * P["ded_sym"]
            pp, _ = V.predictive_pd(
                Wd, u_hat, r_hat, sigma_u, prior, th_dom, nre_d, s_r=s_r, c_ur=c_ur
            )
            pt = np.full(len(Wd), np.nan)
            FDED[f, d] = pp.max() >= TAU
            if extra is not None:
                if extra is not None:
                    extra.ctx = dict(
                        u_hat=u_hat,
                        sigma_u=tu[0],
                        r_hat=r_hat,
                        s_r=tu[1],
                        c_ur=tu[2],
                        jpar=Jpar,
                        prior=prior,
                        th=th_dom,
                        u_ue=None,
                        truth=truth,
                        frame=f,
                    )
                for name, v in extra(
                    f, Wd[[int(np.argmax(pp))]], nre_d, P["ded_prb"] * 12 * P["df"], None
                ).items():
                    XDED.setdefault(name, np.zeros((nf, nd)))[f, d] = (
                        v[0] if FDED[f, d] else 0.0
                    )
            for j in np.flatnonzero(elig):
                g0j = V.GRID[k0[j]]
                if ue_model == "U0":
                    ue_ctrl = (su[j], su[j], su[j])
                elif ue_model == "U1":
                    ue_ctrl = (g0j, g0j - 1 / 64, g0j + 1 / 64)
                else:
                    raise ValueError(ue_model)
                W, tag = V.candidates(
                    su[j], u_hat, k0[j], cand_level, sigma_u=sigma_u, ue_ctrl=ue_ctrl
                )
                in_lv = [np.isin(tag, t) for t in lvl_tags]
                if eval_levels is not None:
                    needed = np.logical_or.reduce([in_lv[lv] for lv in eval_levels])
                    W, tag = (W[needed], tag[needed])
                    in_lv = [mask[needed] for mask in in_lv]
                loss_true = V.db(V.gain(V.beam(g0j), su[j]))[0] - V.db(V.gain(W, su[j]))
                clut = (su[j], sr_rc[j], sig_ue[j], th_rc[j]) if P["ue_clutter"] else None
                clut_c = clut
                if ue_model == "U1":
                    uq = g0j + _tq / 64
                    if ue_loss_fn is None:
                        Lq = V.db(V.gain(V.beam(g0j), uq))[0][None, :] - V.db(V.gain(W, uq))
                        loss = np.quantile(Lq, 1 - P["eps_ue"], axis=1, method="inverted_cdf")
                    else:
                        loss = ue_loss_fn(W, g0j, uq)
                    if clut is not None:
                        clut_c = (g0j + _x / 64, sr_rc[j], sig_ue[j], th_rc[j], _w / 2)
                else:
                    loss = loss_true
                prune = P.get("prune_admissible", False)
                mlo = max(base[j][0] - max(P["dm_list"]), P["mcs_floor"])
                thr_l = next((m for m in V.MCS if m[0] == mlo))[1]
                ev = np.flatnonzero(loss <= gamma0[j] - thr_l + 1e-09)
                pd_pred = np.full(len(W), -1.0)
                if ev.size:
                    screen = (
                        V.predictive_pd
                        if predictive_screen_fn is None
                        else predictive_screen_fn
                    )
                    pd_pred[ev] = screen(
                        W[ev],
                        u_hat,
                        r_hat,
                        sigma_u,
                        prior,
                        th_dom,
                        n_re[j],
                        bg[j],
                        clut_c,
                        s_r=s_r,
                        c_ur=c_ur,
                    )[0]
                m0 = base[j]
                sel = {}
                for q, dm in enumerate(P["dm_list"]):
                    mmin = max(m0[0] - dm, P["mcs_floor"])
                    thr = next((m for m in V.MCS if m[0] == mmin))[1]
                    ok_b = loss <= gamma0[j] - thr + 1e-09
                    se = np.array(
                        [
                            min(V.mcs_of(gamma0[j] - l, mmin)[2], m0[2]) if o else np.nan
                            for l, o in zip(loss, ok_b)
                        ]
                    )
                    dse = m0[2] - se
                    for lv in range(nl):
                        if eval_levels is not None and lv not in eval_levels:
                            continue
                        ok = in_lv[lv] & ok_b & (pd_pred >= TAU)
                        if ok.any():
                            idx = np.flatnonzero(ok)
                            c = idx[np.lexsort((loss[idx], -pd_pred[idx], dse[idx]))[0]]
                            C[f, d, lv, q, j] = True
                            DSE[f, d, lv, q, j], PDP[f, d, lv, q, j] = (dse[c], pd_pred[c])
                            LOSS[f, d, lv, q, j] = loss[c]
                            TSEL[f, d, lv, q, j] = tag[c]
                            sel[lv, q] = c
                if extra is not None and sel:
                    uc = sorted(set(sel.values()))
                    if extra is not None:
                        extra.ctx = dict(
                            u_hat=u_hat,
                            sigma_u=tu[0],
                            r_hat=r_hat,
                            s_r=tu[1],
                            c_ur=tu[2],
                            jpar=Jpar,
                            prior=prior,
                            th=th_dom,
                            u_ue=su[j],
                            g0=V.GRID[k0[j]],
                            truth=truth,
                            frame=f,
                        )
                    pos = {c: i for i, c in enumerate(uc)}
                    for name, v in extra(f, W[uc], n_re[j], bg[j], clut).items():
                        arr = X.setdefault(name, np.full(C.shape, np.nan))
                        for (lv, q), c in sel.items():
                            arr[f, d, lv, q, j] = v[pos[c]]
        print(f"  frame {f + 1}/{nf} done", flush=True)
    return dict(
        C=C,
        DSE=DSE,
        PDP=PDP,
        LOSS=LOSS,
        FDED=FDED,
        SE0=SE0,
        N_RE=n_re,
        X=X,
        XDED=XDED,
        TSEL=TSEL,
    )
