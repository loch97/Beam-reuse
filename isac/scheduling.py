"""Exact fixed-grant selection probabilities, costs and stress scores."""

from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from math import comb
from scipy.special import gammaln
from . import model as V

P = V.P
TAU = 0.9
N_EVAL = 5
STRESS = ["s1", "s2", "b+2", "b-2"]
LEVELS = [0, 2, 11, 13]
POLICIES = ["Unmodified", "Beam steering", "Broadened beam", "Multibeam", "Sequential"]


def log_comb(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def rank_weights(ns, N, K):
    """P(rank-k compatible UE is the one used), k = 1..K, exactly-N uniform scheduler."""
    k = np.arange(1, K + 1)
    ok = ns - k >= N - 1
    return np.where(ok, np.exp(log_comb(ns - k, N - 1) - log_comb(ns, N)), 0.0)


def expectations(A):
    nf, nd, _, nq, n = A["C"].shape
    result = {
        p: {
            k: np.zeros((nq, 10))
            for k in ["need", "dedicated", "unserved", "adaptation", "displacement", "cost"]
        }
        for p in POLICIES
    }

    def order(f, d, l, q, idx):
        return idx[
            np.lexsort(
                (
                    idx,
                    A["LOSS"][f, d, l, q, idx],
                    -A["PDP"][f, d, l, q, idx],
                    A["DSE"][f, d, l, q, idx],
                )
            )
        ]

    for f in range(nf):
        for d in range(nd):
            for q in range(nq):
                for p, l in zip(POLICIES, [0, 2, 11, 13, None]):
                    if l is None:
                        a = np.flatnonzero(A["C"][f, d, 11, q])
                        b = np.setdiff1d(np.flatnonzero(A["C"][f, d, 13, q]), a)
                        blocks = [(11, order(f, d, 11, q, a)), (13, order(f, d, 13, q, b))]
                    else:
                        idx = np.flatnonzero(A["C"][f, d, l, q])
                        blocks = [(l, order(f, d, l, q, idx))]
                    selected = np.concatenate([idx for _, idx in blocks])
                    K = len(selected)
                    un = np.ones(n, bool)
                    un[selected] = False
                    for N in range(1, 11):
                        denom = comb(n, N)
                        v = comb(n - K, N) / denom if n - K >= N else 0.0
                        ad = 0.0
                        offset = 0
                        wsum = 0.0
                        for lv, idx in blocks:
                            w = np.array(
                                [
                                    (
                                        comb(n - offset - r, N - 1) / denom
                                        if n - offset - r >= N - 1
                                        else 0.0
                                    )
                                    for r in range(1, len(idx) + 1)
                                ]
                            )
                            ad += float(w @ (A["N_RE"][idx] * A["DSE"][f, d, lv, q, idx]))
                            wsum += w.sum()
                            offset += len(idx)
                        assert abs(wsum + v - 1) < 1e-12
                        used = float(A["FDED"][f, d]) * v
                        disp = (
                            used
                            * V.P["ded_prb"]
                            * 12
                            * V.P["ded_sym"]
                            * A["SE0"][f, d, un].mean()
                            if used
                            else 0.0
                        )
                        vals = [v, used, v - used, ad, disp, ad + disp]
                        for key, value in zip(result[p], vals):
                            result[p][key][q, N - 1] += value / (nf * nd)
    assert all((np.max(np.abs(result[p]["adaptation"][0])) < 1e-12 for p in POLICIES))
    assert all((np.all(np.diff(result[p]["dedicated"], axis=1) <= 1e-12) for p in POLICIES))
    return {p: {k: v.tolist() for k, v in r.items()} for p, r in result.items()}


def summarize(A, eval_levels=None):
    C, F, ns = (A["C"], A["FDED"], A["C"].shape[-1])
    nf, nd, nl, ndm, _ = C.shape
    res = {}
    for lv in range(nl):
        if eval_levels is not None and lv not in eval_levels:
            continue
        for q, dm in enumerate(P["dm_list"]):
            per = {
                k: dict(sys=np.zeros(nf), rw=np.zeros(nf), rx=np.zeros(nf), ro=np.zeros(nf))
                for k in STRESS
            }
            ded = np.zeros(nf)
            for f in range(nf):
                for d in range(nd):
                    idx = np.flatnonzero(C[f, d, lv, q])
                    K = len(idx)
                    v = (
                        math.exp(log_comb(ns - K, N_EVAL) - log_comb(ns, N_EVAL))
                        if ns - K >= N_EVAL
                        else 0.0
                    )
                    ded[f] += F[f, d] * v / nd
                    w, o = (None, None)
                    if K:
                        o = idx[
                            np.lexsort(
                                (
                                    idx,
                                    A["LOSS"][f, d, lv, q, idx],
                                    -A["PDP"][f, d, lv, q, idx],
                                    A["DSE"][f, d, lv, q, idx],
                                )
                            )
                        ]
                        w = rank_weights(ns, N_EVAL, K)
                    for k in STRESS:
                        pk = per[k]
                        if K:
                            x = A["X"][k][f, d, lv, q, o]
                            pk["rw"][f] += w.sum() / nd
                            pk["rx"][f] += float(w @ x) / nd
                            pk["ro"][f] += float(w @ (x < TAU)) / nd
                            pk["sys"][f] += float(w @ x) / nd
                        pk["sys"][f] += F[f, d] * v * float(A["XDED"][k][f, d]) / nd
            r = dict(o_ded=float(ded.mean()), o_ded_frames=ded.tolist())
            for k in STRESS:
                pk = per[k]
                r[k] = dict(
                    P_det=float(pk["sys"].mean()),
                    P_out=(
                        float(pk["ro"].sum() / pk["rw"].sum()) if pk["rw"].sum() > 0 else None
                    ),
                    mean_reuse=(
                        float(pk["rx"].sum() / pk["rw"].sum()) if pk["rw"].sum() > 0 else None
                    ),
                    P_det_frames=pk["sys"].tolist(),
                    P_out_frames=[
                        float(a / b) if b > 0 else None for a, b in zip(pk["ro"], pk["rw"])
                    ],
                    mean_reuse_frames=[
                        float(a / b) if b > 0 else None for a, b in zip(pk["rx"], pk["rw"])
                    ],
                    reuse_weight_frames=pk["rw"].tolist(),
                )
            r["nominal_check_max_abs"] = float(
                np.nanmax(
                    np.abs(
                        A["X"]["s1"][:, :, lv, q][C[:, :, lv, q]]
                        - A["PDP"][:, :, lv, q][C[:, :, lv, q]]
                    )
                )
            )
            res[f"L{lv}_dm{dm}"] = r
    return res


def _order(A, f, d, lv, q, idx):
    """Policy order of compatible locations: min dSE, max p̄_D, min loss, spot index."""
    return idx[
        np.lexsort(
            (
                idx,
                A["LOSS"][f, d, lv, q, idx],
                -A["PDP"][f, d, lv, q, idx],
                A["DSE"][f, d, lv, q, idx],
            )
        )
    ]


def _lc(n, k):
    return log_comb(n, k) if 0 <= k <= n else -np.inf


def cascade(A, l1, lt):
    """Round-12 three-tier policy at N = 5 under the uniform exactly-N scheduler (exact):
    tier 1 = best scheduled location compatible at level l1 (contiguous-beam adaptation), else
    tier 2 = best scheduled location compatible at level lt (embedded two-lobe lobe) among the others,
    else tier 3 = standalone look (if feasible). Rank-k of tier 1 is used w.p. C(n-k, N-1)/C(n, N);
    rank-k of the tier-2 list (locations not tier-1 compatible) w.p. C(n-K1-k, N-1)/C(n, N)."""
    C, F, ns, N = (A["C"], A["FDED"], A["C"].shape[-1], N_EVAL)
    nf, nd, nl, ndm, _ = C.shape
    nre = A["N_RE"].astype(float)
    ded_re = P["ded_prb"] * 12 * P["ded_sym"]
    out = {}
    for q, dm in enumerate(P["dm_list"]):
        fr = {
            k: np.zeros(nf)
            for k in (
                "o_need",
                "o_embed",
                "o_std",
                "o_unserved",
                "bits_reuse",
                "bits_embed",
                "bits_std",
                "w1",
                "w2",
            )
        }
        st = {k: dict(sys=np.zeros(nf), o1=np.zeros(nf), o2=np.zeros(nf)) for k in STRESS}
        L0 = _lc(ns, N)
        for f in range(nf):
            for d in range(nd):
                a = np.flatnonzero(C[f, d, l1, q])
                b = np.setdiff1d(np.flatnonzero(C[f, d, lt, q]), a)
                K1, K2 = (len(a), len(b))
                vR = math.exp(_lc(ns - K1, N) - L0)
                vU = math.exp(_lc(ns - K1 - K2, N) - L0)
                o1 = _order(A, f, d, l1, q, a) if K1 else a
                o2 = _order(A, f, d, lt, q, b) if K2 else b
                w1 = np.array([math.exp(_lc(ns - k, N - 1) - L0) for k in range(1, K1 + 1)])
                w2 = np.array(
                    [math.exp(_lc(ns - K1 - k, N - 1) - L0) for k in range(1, K2 + 1)]
                )
                fr["o_need"][f] += vR / nd
                fr["o_embed"][f] += (vR - vU) / nd
                fr["o_std"][f] += F[f, d] * vU / nd
                fr["o_unserved"][f] += (1 - F[f, d]) * vU / nd
                fr["w1"][f] += w1.sum() / nd
                fr["w2"][f] += w2.sum() / nd
                if K1:
                    fr["bits_reuse"][f] += (
                        float(w1 @ (A["DSE"][f, d, l1, q, o1] * nre[o1])) / nd
                    )
                if K2:
                    fr["bits_embed"][f] += (
                        float(w2 @ (A["DSE"][f, d, lt, q, o2] * nre[o2])) / nd
                    )
                if F[f, d] and vU > 0:
                    un = np.ones(ns, bool)
                    un[a] = False
                    un[b] = False
                    fr["bits_std"][f] += vU * ded_re * A["SE0"][f, d, un].mean() / nd
                for k in STRESS:
                    x1 = A["X"][k][f, d, l1, q, o1] if K1 else np.zeros(0)
                    x2 = A["X"][k][f, d, lt, q, o2] if K2 else np.zeros(0)
                    s = st[k]
                    s["sys"][f] += (
                        float(w1 @ x1)
                        + float(w2 @ x2)
                        + F[f, d] * vU * float(A["XDED"][k][f, d])
                    ) / nd
                    s["o1"][f] += float(w1 @ (x1 < TAU)) / nd
                    s["o2"][f] += float(w2 @ (x2 < TAU)) / nd
        r = {k: float(v.mean()) for k, v in fr.items()}
        r["bits_total"] = r["bits_reuse"] + r["bits_embed"] + r["bits_std"]
        r["frames"] = {k: v.tolist() for k, v in fr.items()}
        for k in STRESS:
            s = st[k]
            r[k] = dict(
                P_det=float(s["sys"].mean()),
                P_out_tier1=(
                    float(s["o1"].sum() / fr["w1"].sum()) if fr["w1"].sum() > 0 else None
                ),
                P_out_tier2=(
                    float(s["o2"].sum() / fr["w2"].sum()) if fr["w2"].sum() > 0 else None
                ),
                P_det_frames=s["sys"].tolist(),
            )
        out[f"dm{dm}"] = r
    return out
