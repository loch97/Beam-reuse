"""Multi-scatterer (extended) target model and its uncertainty-averaged detection probability.

A vehicle is a rigid L x W box. Its RSU-visible faces contribute a face-centre scatterer and their
end corners; the face seen most frontally is dominant. Every scatterer is an independent Swerling-1
reflector whose mean RCS is the dominant value plus a fixed offset. Returns that fall in the same
receive cell add in power, so the per-cell statistic stays exponential and the paper's detector
P_D = exp(ln(Pfa) (1 + L) / (1 + L + rho)) still applies, now with rho summed over scatterers
through the receive-array and finite-subcarrier delay responses.

Pose uncertainty (position and heading of the box) is integrated over an exact-mass cell
quadrature of the tracker's predicted Gaussian; all scatterers move rigidly with the pose node.
The receiver tests the cells of every scatterer, and P_D = E[max_c P_D,c] is a lower bound on the
probability that at least one cell detects. ``focus="dominant"`` tests the dominant cell only.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import numpy as np
from scipy.special import ndtr, gammaln, gammainc
from scipy.stats import ncx2
from . import model as V
from .detection import delay_power

ROOT = Path(__file__).resolve().parents[1]
FP = json.loads((ROOT / "data/frontend_parameters.json").read_text())


def rot(h):
    c, s = math.cos(h), math.sin(h)
    return np.array([[c, -s], [s, c]])


def box_scatterers(center, heading, L, W, rsu, sig_dom, offsets=None):
    """Visible box scatterers. Same construction as ``model.vehicle_rcs``.

    Returns (keys, xy, sigma_dbsm, dominant key, body-frame offsets). When no face is visible (the
    RSU lies inside the box, e.g. a ghost track next to the RSU) all are empty and the dominant key
    is None; callers fall back to a point target.
    """
    off = FP["scatterer_offsets_db"] if offsets is None else offsets
    c = np.asarray(center, float)
    ex = np.array([math.cos(heading), math.sin(heading)])
    ey = np.array([-math.sin(heading), math.cos(heading)])
    faces = {
        "front": (c + ex * L / 2, ex, ey * W / 2),
        "rear": (c - ex * L / 2, -ex, ey * W / 2),
        "left": (c + ey * W / 2, ey, ex * L / 2),
        "right": (c - ey * W / 2, -ey, ex * L / 2),
    }
    best, dom = -1, None
    pts = {}
    for name, (cf, nf, half) in faces.items():
        to_rsu = np.asarray(rsu, float) - cf
        vis = float(nf @ to_rsu)
        if vis <= 0:
            continue
        cos_inc = vis / np.linalg.norm(to_rsu)
        if cos_inc > best:
            best, dom = cos_inc, name
        pts["c_" + name] = cf
        for s in (-1, 1):
            corner = cf + s * half
            pts["k_%.2f_%.2f" % tuple(corner)] = corner
    if dom is None:
        return [], np.zeros((0, 2)), np.zeros(0), None, np.zeros((0, 2))
    keys = list(pts)
    xy = np.array([pts[k] for k in keys])
    sig = np.array(
        [
            sig_dom + off["dominant_face"]
            if k == "c_" + dom
            else sig_dom + off["corner" if k.startswith("k_") else "other_face"]
            for k in keys
        ]
    )
    body = (xy - c) @ rot(heading)
    return keys, xy, sig, "c_" + dom, body


@dataclass
class PoseNodes:
    """Scatterer polar coordinates at each pose node: arrays (n_nodes, n_scatterers)."""

    u: np.ndarray
    r: np.ndarray
    theta: np.ndarray
    w: np.ndarray
    sigma_dbsm: np.ndarray
    dom: int
    keys: list
    steer: dict = field(default_factory=dict, repr=False, compare=False)


def _axis_nodes(n, zmax):
    edges = np.linspace(-zmax, zmax, n + 1)
    cdf = ndtr(edges)
    cdf[0], cdf[-1] = 0.0, 1.0
    return 0.5 * (edges[:-1] + edges[1:]), np.diff(cdf)


def _u_gradient(xy, rsu, bore):
    d = np.asarray(xy, float) - rsu
    h = float(np.hypot(*d))
    n = d / h
    a = np.array([-math.sin(bore), math.cos(bore)])
    return (a - float(a @ n) * n) / h


def pose_nodes(mean_xy, cov_xy, heading, heading_std, extent, rsu, bore, sig_dom, h_rc=None,
               per_pose=False):
    """Cell quadrature over position (principal axes of ``cov_xy``) and heading.

    The cell width on each axis keeps the induced direction step below ``ext_u_cell`` and below
    1/``ext_cells_per_sigma`` standard deviations, capped at ``ext_max_cells`` cells per axis.
    Heading nodes are added when the heading spread moves the box ends by more than
    ``ext_heading_min_disp_m``. Visibility and the dominant face are fixed at the mean pose
    (default); ``per_pose=True`` recomputes them at every node (``box_scatterers`` there), so a
    face that turns away stops reflecting and the dominant-face RCS moves with the pose. The
    node arrays then span every scatterer visible at some node, labelled by its body-frame
    position; a scatterer hidden at a node gets -inf dBsm (no power, no cell) there, so
    ``sigma_dbsm`` is (nodes, scatterers). ``dom`` stays the mean pose's dominant face (the RCS
    reference of the evidence curves).
    """
    h_rc = V.P["rc_h"] if h_rc is None else h_rc
    L, W = extent
    keys, _, sig, dom, body = box_scatterers(mean_xy, heading, L, W, rsu, sig_dom)
    if dom is None:
        raise ValueError("no visible box face: the RSU lies inside the target box")
    lam, E = np.linalg.eigh(0.5 * (cov_xy + np.asarray(cov_xy).T))
    lam = np.maximum(lam, 0.0)
    grad = _u_gradient(mean_xy, rsu, bore)
    zmax, cps = FP["ext_z_max"], FP["ext_cells_per_sigma"]
    axes = []
    for i in range(2):
        s = math.sqrt(lam[i])
        if s < 1e-6:
            axes.append((np.zeros(1), np.ones(1)))
            continue
        su = abs(float(grad @ E[:, i])) * s
        hz = min(1.0 / cps, FP["ext_u_cell"] / su) if su > 0 else 1.0 / cps
        n = min(int(math.ceil(2 * zmax / hz)), FP["ext_max_cells"])
        axes.append(_axis_nodes(max(n, 1), zmax))
    (z1, w1), (z2, w2) = axes
    Z1, Z2 = np.meshgrid(z1, z2, indexing="ij")
    wxy = np.outer(w1, w2).ravel()
    keep = wxy >= FP["ext_min_mass"] * wxy.max()
    dxy = (np.column_stack([Z1.ravel(), Z2.ravel()]) * np.sqrt(lam)) @ E.T
    dxy, wxy = dxy[keep], wxy[keep]
    if heading_std * max(L, W) / 2 > FP["ext_heading_min_disp_m"] and FP["ext_heading_nodes"] > 1:
        zh, wh = np.polynomial.hermite_e.hermegauss(FP["ext_heading_nodes"])
        wh = wh / wh.sum()
    else:
        zh, wh = np.zeros(1), np.ones(1)
    centers = np.asarray(mean_xy, float) + dxy
    w = np.concatenate([wxy * x for x in wh])
    if per_pose:
        return _per_pose_nodes(centers, heading, heading_std, zh, w, L, W, rsu, bore, sig_dom,
                               h_rc, body[keys.index(dom)])
    xy = np.concatenate(
        [centers[:, None, :] + (body @ rot(heading + heading_std * z).T)[None] for z in zh]
    )
    n, k = xy.shape[:2]
    u, r, th, _ = V.to_polar(xy.reshape(-1, 2), rsu, bore, V.P["rsu_h"], h_rc)
    return PoseNodes(
        u.reshape(n, k), r.reshape(n, k), th.reshape(n, k), w / w.sum(), sig, keys.index(dom), keys
    )


def _per_pose_nodes(centers, heading, heading_std, zh, w, L, W, rsu, bore, sig_dom, h_rc,
                    dom_body):
    """``pose_nodes`` with visibility and the dominant face recomputed at every node."""
    label = lambda b: (int(round(2 * b[0] / L)), int(round(2 * b[1] / W)))
    rows = []
    for z in zh:
        h = heading + heading_std * z
        for c in centers:
            keys, xy, sig, dom, body = box_scatterers(c, h, L, W, rsu, sig_dom)
            rows.append({label(b): (p, s_) for b, p, s_ in zip(body, xy, sig)} if dom else {})
    labels = sorted(set().union(*rows))
    if not labels:
        raise ValueError("no visible box face at any pose node")
    n, k = len(rows), len(labels)
    xy = np.empty((n, k, 2))
    sig = np.full((n, k), -np.inf)
    hs = np.repeat(heading + heading_std * np.asarray(zh), len(centers))
    cs = np.tile(centers, (len(zh), 1))
    for j, lab in enumerate(labels):
        off = np.array([lab[0] * L / 2, lab[1] * W / 2])
        ch, sh = np.cos(hs), np.sin(hs)
        xy[:, j] = cs + np.column_stack([ch * off[0] - sh * off[1], sh * off[0] + ch * off[1]])
        for i, r_ in enumerate(rows):
            if lab in r_:
                sig[i, j] = r_[lab][1]
    u, r, th, _ = V.to_polar(xy.reshape(-1, 2), rsu, bore, V.P["rsu_h"], h_rc)
    keys = ["pp_%d_%d" % lab for lab in labels]
    dom = labels.index(label(dom_body)) if label(dom_body) in labels else int(np.argmax(sig.max(0)))
    return PoseNodes(u.reshape(n, k), r.reshape(n, k), th.reshape(n, k), w / w.sum(), sig, dom,
                     keys)


def array_gain(W, u):
    """|a(u)^T w|^2 / M without the byte-keyed response cache used by the batch simulation."""
    W = np.atleast_2d(W)
    A = np.exp(1j * np.pi * np.multiply.outer(np.asarray(u, float), V.N_IDX))
    return np.abs(np.tensordot(W, A, axes=([1], [-1]))) ** 2 / V.M


def node_gain(W, nodes):
    """``array_gain(W, nodes.u)`` with the steering matrix cached on ``nodes`` (per array size)."""
    W = np.atleast_2d(W)
    A = nodes.steer.get(V.M)
    if A is None:
        A = nodes.steer[V.M] = np.exp(1j * np.pi * np.multiply.outer(np.asarray(nodes.u, float),
                                                                     V.N_IDX))
    return np.abs(np.tensordot(W, A, axes=([1], [-1]))) ** 2 / V.M


def cell_snr(W, nodes, n_re, bandwidth):
    """Mean SNR in each scatterer's receive cell: (beams, nodes, cells), linear."""
    tx = node_gain(W, nodes)
    rho = 10 ** (V.rc_snr_db(tx, nodes.u, nodes.r, nodes.sigma_dbsm, nodes.theta, n_re) / 10)
    du = nodes.u[:, :, None] - nodes.u[:, None, :]
    dr = nodes.r[:, :, None] - nodes.r[:, None, :]
    coupling = V.rx_rel(du) * delay_power(dr, bandwidth)
    out = np.einsum("bnk,nck->bnc", rho, coupling, optimize=True)
    if np.ndim(nodes.sigma_dbsm) == 2:  # per-pose visibility: a hidden scatterer has no cell
        out *= np.isfinite(nodes.sigma_dbsm)[None]
    return out


def cell_pd(rho, leak=None, lp=None):
    """Probability that a cell crosses the square-law threshold.

    ``rho`` is the cell's mean target SNR over thermal noise and ``leak`` its UE leakage, both
    over noise; the threshold is eta = -ln(pfa) on the noise-plus-leakage scale. The target power
    in the cell is Gamma distributed with order m = ``model.P["speckle_m"]``:
    - m unset or 1: exponential (Rayleigh speckle, Swerling 1), P = exp(ln pfa (1+L)/(1+L+rho)),
      evaluated exactly as before;
    - m != 1: exact negative-binomial series (m > 1 less fluctuation);
    - m = inf: deterministic cell power, i.e. Marcum Q (Poisson series).
    P is increasing in rho for every m, so the screening bounds stay valid. ``lp`` may be an
    array (element-wise thresholds, broadcast against rho).
    """
    lp = math.log(V.P["pfa"]) if lp is None else lp
    m = V.P.get("speckle_m")
    if m is None or m == 1:
        if leak is None:
            return np.exp(lp / (1 + rho))
        return np.exp(lp * (1 + leak) / (1 + leak + rho))
    s = np.asarray(rho, float) if leak is None else np.asarray(rho / (1 + leak), float)
    eta = -lp
    # Exact: the statistic is a Poisson(S) mixture of Gamma(k + 1, 1) variables; averaging the
    # Poisson over S ~ Gamma(m, s/m) gives a negative binomial. P(miss) = sum_k pmf(k) P(k+1, eta)
    # with P the regularised lower incomplete gamma, negligible once k >> eta.
    s_ = np.maximum(s, 1e-300)
    miss = np.zeros(s.shape)
    k = -1
    while True:
        k += 1
        tail = gammainc(k + 1, eta)
        if np.max(tail) < 1e-12:
            break
        if np.isinf(m):
            logpmf = -s_ + k * np.log(s_) - gammaln(k + 1)
        else:
            logpmf = (gammaln(k + m) - gammaln(m) - gammaln(k + 1)
                      + m * np.log(m / (m + s_)) + k * np.log(s_ / (m + s_)))
        miss += np.exp(logpmf) * tail
    return 1.0 - miss


def gain_upper_bound(W, u_lo, u_hi):
    """Rigorous upper bound of G_w(u) = |a(u)^T w|^2 / M over u in [u_lo, u_hi], per beam.

    G is a trigonometric polynomial of degree M - 1 in pi*u, so by Bernstein's inequality
    |dG/du| <= pi (M - 1) max G <= pi (M - 1) ||w||_1^2 / M. The maximum on a grid of step h
    plus h/2 times that slope bounds the interval maximum. h is chosen so the margin is 5% of
    ||w||_1^2 / M (5% of the peak for unit-modulus beams).
    """
    W = np.atleast_2d(W)
    m = W.shape[1]
    glob = np.abs(W).sum(axis=1) ** 2 / m
    h = 0.1 / (math.pi * max(m - 1, 1))
    n = max(2, int(math.ceil((u_hi - u_lo) / h)) + 1)
    grid = np.linspace(u_lo, u_hi, n)
    step = (u_hi - u_lo) / (n - 1) if n > 1 else 0.0
    return array_gain(W, grid).max(axis=1) + 0.5 * step * math.pi * (m - 1) * glob


def pd_upper_bound(W, nodes, n_re, shifts_db, weights):
    """Cheap clutter-free upper bound of the RCS-averaged extended P_D, per beam.

    Every scatterer gets the bounded transmit gain over the direction span of all pose nodes, its
    smallest range over the nodes and the maximum element gain; cell coupling is bounded by 1, so
    the cell SNR is at most the sum over scatterers. UE leakage only lowers P_D.
    """
    lo, hi = float(nodes.u.min()), float(nodes.u.max())
    g = gain_upper_bound(W, lo, hi)
    r_min = nodes.r.min(axis=0)
    sig = nodes.sigma_dbsm if np.ndim(nodes.sigma_dbsm) == 1 else nodes.sigma_dbsm.max(axis=0)
    unit = float(np.sum(10 ** (V.rc_snr_db(1.0, 0.0, r_min, sig, 90.0, n_re) / 10)))
    lp = math.log(V.P["pfa"])
    scale = 10 ** (np.atleast_1d(np.asarray(shifts_db, float)) / 10)
    pd = cell_pd(scale[:, None] * unit * g[None, :], None, lp)
    return np.asarray(weights) @ pd


def focus_snr(W, nodes, n_re, bandwidth, u0, r0):
    """Mean SNR in one fixed receive cell (u0, r0) at every pose node: (beams, nodes), linear."""
    tx = array_gain(W, nodes.u)
    rho = 10 ** (V.rc_snr_db(tx, nodes.u, nodes.r, nodes.sigma_dbsm, nodes.theta, n_re) / 10)
    return np.einsum("bnk,nk->bn", rho,
                     V.rx_rel(u0 - nodes.u) * delay_power(r0 - nodes.r, bandwidth))


def fixed_cell_pd_shifts(W, nodes, n_re, bandwidth, u0, r0, clutter=None, shifts_db=(0.0,)):
    """Crossing probability of the fixed cell (u0, r0), averaged over pose nodes: (shifts, beams).

    This is the designated-cell event when the back end reads the cell at the predicted position
    rather than wherever the scatterer actually is.
    """
    W = np.atleast_2d(W)
    rho = focus_snr(W, nodes, n_re, bandwidth, u0, r0)
    scale = 10 ** (np.atleast_1d(np.asarray(shifts_db, float)) / 10)
    lp = math.log(V.P["pfa"])
    if clutter is None:
        leaks, cw = np.zeros((len(W), 1)), np.ones(1)
    else:
        cu, cr, cs, ct = clutter[:4]
        cu = np.atleast_1d(cu)
        cw = np.asarray(clutter[4]) if len(clutter) > 4 else np.ones(len(cu)) / len(cu)
        cp = 10 ** (V.rc_snr_db(array_gain(W, cu), cu, cr, cs, ct, n_re) / 10)
        leaks = cp * (V.rx_rel(u0 - cu) * delay_power(r0 - cr, bandwidth))[None, :]
    out = np.zeros((len(scale), len(W)))
    for i, x in enumerate(scale):
        for q in range(leaks.shape[1]):
            L = leaks[:, q, None]
            out[i] += cw[q] * (cell_pd(x * rho, L, lp) @ nodes.w)
    return out


def extended_pd_shifts(W, nodes, n_re, bandwidth, clutter=None, focus=None, shifts_db=(0.0,),
                       screen=None, rho=None, need_upper=True):
    """P_D of the extended target for each RCS shift and beam: arrays (shifts, beams).

    Every scatterer RCS in ``nodes`` moves by the same shift (dB), so a shift evaluates the target
    at another dominant RCS. ``clutter`` = (cu, cr, cs, ct[, cw]) as in ``detection.predictive_pd``:
    UE-reflector directions ``cu`` with quadrature weights ``cw``, range ``cr``, RCS ``cs`` and
    elevation ``ct``. ``screen`` = (threshold, shift weights): beams whose shift-averaged
    clutter-free value (an upper bound) is below the threshold skip the clutter evaluation and
    return -1. ``rho`` passes precomputed ``cell_snr(W, nodes, n_re, bandwidth)``.
    Returns (pd, clutter-free pd). ``need_upper=False`` skips the clutter-free value when clutter
    is given and there is no screen; it is then returned as None.
    """
    focus = FP["ext_focus"] if focus is None else focus
    W = np.atleast_2d(W)
    scale = 10 ** (np.atleast_1d(np.asarray(shifts_db, float)) / 10)
    lp = math.log(V.P["pfa"])
    rho = cell_snr(W, nodes, n_re, bandwidth) if rho is None else rho
    if focus == "dominant":
        rho = rho[:, :, [nodes.dom]]
        cu_nodes, cr_nodes = nodes.u[:, [nodes.dom]], nodes.r[:, [nodes.dom]]
    elif focus == "any":
        cu_nodes, cr_nodes = nodes.u, nodes.r
    else:
        raise ValueError(focus)
    best = rho.max(axis=2)
    if clutter is not None and screen is None and not need_upper:
        upper = None
    else:
        upper = _chunked_shifts(scale, best.size,
                                lambda x: cell_pd(x[:, None, None] * best[None], None, lp)
                                @ nodes.w)
    if clutter is None:
        return upper, upper
    active = np.arange(len(W))
    if screen is not None:
        threshold, weights = screen
        active = np.flatnonzero(np.asarray(weights) @ upper >= threshold - 1e-11)
    out = np.full((len(scale), len(W)), -1.0)
    if not len(active):
        return out, upper
    rho = rho[active]
    cu, cr, cs, ct = clutter[:4]
    cu = np.atleast_1d(cu)
    cw = np.asarray(clutter[4]) if len(clutter) > 4 else np.ones(len(cu)) / len(cu)
    cp = 10 ** (V.rc_snr_db(array_gain(W[active], cu), cu, cr, cs, ct, n_re) / 10)
    resp = V.rx_rel(cu_nodes[None] - cu[:, None, None]) * delay_power(cr_nodes - cr, bandwidth)
    out[:, active] = _clutter_pd(scale, rho, cp, resp, cw, nodes.w, lp)
    return out, upper


# Chunk sizes of the vectorised P_D kernels (elements per temporary array).
CHUNK_ELEMS = 2_000_000
# Per-cell tolerance of the bracketed clutter average; the result moves by at most half of it.
CLUTTER_EPS = 1e-13


def _chunked_shifts(scale, per_shift, fn):
    """Stack fn(scale chunk) over shift chunks of at most CHUNK_ELEMS elements."""
    step = max(1, CHUNK_ELEMS // max(int(per_shift), 1))
    return np.concatenate([fn(scale[i:i + step]) for i in range(0, len(scale), step)])


def _clutter_pd(scale, rho, cp, resp, cw, wn, lp):
    """Clutter-averaged any-cell P_D: array (shifts, beams).

    Equals, for every shift x, sum_n wn max_k sum_q cw_q cell_pd(x rho, L_q) with
    L_q = cp_q resp_q (rho (B, N, K), cp (B, Q), resp (Q, N, K)) to within CLUTTER_EPS / 2.
    The quadrature over the UE-leakage nodes q is the cost. For Rayleigh speckle it is replaced by
    f(mean L), with f(L) = exp(lp (1 + L) / (1 + L + y)) and y = x rho, wherever the Taylor
    remainder bound proves it close:
        |sum_q cw_q f(L_q) - sum_q cw_q f(mean L)| <= sum_q cw_q Var_q(L) max|f''| / 2,
        |f''| <= f(0) (lp^2 g1^2 + |lp| g2), g1 = y / (1 + y)^2, g2 = 2 y / (1 + y)^3, all L >= 0.
    The mean-leak value +- that bound brackets the exact value; a cell whose upper bound is below
    another cell's lower bound cannot be the maximum and is not refined. Only cells whose bound
    exceeds CLUTTER_EPS / 2 and that can be the maximum evaluate the quadrature. Other speckle
    orders evaluate the quadrature everywhere.
    """
    B, N, K = rho.shape
    cwq = np.asarray(cw, float)
    wsum = float(cwq.sum())
    cwn = cwq / wsum
    lbar, lvar = np.empty(rho.shape), np.empty(rho.shape)
    for b in range(B):
        L = cp[b][:, None, None] * resp
        lbar[b] = np.tensordot(cwn, L, 1)
        lvar[b] = np.tensordot(cwn, (L - lbar[b]) ** 2, 1)
    respT = resp.reshape(len(cwq), N * K).T
    m = V.P.get("speckle_m")
    if m is not None and m != 1:
        def exact(x):
            pd = np.zeros((len(x),) + rho.shape)
            for q in range(len(cwq)):
                leak = cp[:, q, None, None] * resp[q][None]
                pd += cwq[q] * cell_pd(x[:, None, None, None] * rho[None], leak[None], lp)
            return pd.max(axis=-1) @ wn

        return _chunked_shifts(scale, 2 * rho.size, exact)
    half_var = 0.5 * wsum * lvar
    # Per cell, the SNR window [ylo, yhi] outside which the Taylor bound is below
    # CLUTTER_EPS / 2 (from a rigorous envelope of |f''| on a log grid of y); elements outside
    # their window use f(mean L) directly. Cells as contiguous (beam, node) slices so the max
    # over cells is elementwise.
    edges, env = _curvature_envelope(lp)
    thr = CLUTTER_EPS / 2 / np.maximum(half_var, 1e-300)
    left = np.maximum.accumulate(env)
    right = np.maximum.accumulate(env[::-1])[::-1]
    j0 = np.searchsorted(left, thr, side="right")  # first interval with env > thr
    j1 = len(env) - np.searchsorted(right[::-1], thr, side="right") - 1  # last such interval
    is_open = j0 < len(env)
    ylo = np.where(is_open, edges[np.minimum(j0, len(env) - 1)], np.inf)
    yhi = np.where(is_open, edges[np.clip(j1 + 1, 0, len(edges) - 1)], -np.inf)
    yhi = np.where(is_open & (j1 == len(env) - 1), np.inf, yhi)
    rk = [np.ascontiguousarray(rho[:, :, k]).ravel() for k in range(K)]
    lk = [np.ascontiguousarray(lbar[:, :, k]).ravel() for k in range(K)]
    lo_k = [np.ascontiguousarray(ylo[:, :, k]).ravel() for k in range(K)]
    hi_k = [np.ascontiguousarray(yhi[:, :, k]).ravel() for k in range(K)]
    hv_k = [np.ascontiguousarray(half_var[:, :, k]).ravel() for k in range(K)]
    open_k = [bool(np.any(np.isfinite(lo))) for lo in lo_k]
    step = max(1, CHUNK_ELEMS // len(cwq))

    def chunk(x):
        # per cell slice k: the mean-leak value for every element, then for elements inside
        # their window the Taylor bound (gathered when the window is sparse, dense otherwise)
        vals, bounds = [], []
        for k in range(K):
            y = x[:, None] * rk[k][None]
            v = wsum * cell_pd(y, lk[k][None], lp)
            vals.append(v)
            if not open_k[k]:
                bounds.append(None)
                continue
            win = (y >= lo_k[k][None]) & (y <= hi_k[k][None])
            if win.mean() > 0.5:
                yy, idx = y, None
            else:
                idx = np.nonzero(win)
                yy = y[idx]
            e = np.exp(lp / (1 + yy))
            g1 = yy / (1 + yy) ** 2
            hv = hv_k[k][None] if idx is None else hv_k[k][idx[1]]
            bnd = e * (lp * lp * g1 * g1 + abs(lp) * 2 * g1 / (1 + yy)) * hv
            bounds.append((idx, yy, e, bnd))
        if all(bd is None for bd in bounds):
            return np.maximum.reduce(vals).reshape(len(x), B, N) @ wn
        low = [v - CLUTTER_EPS / 2 for v in vals]
        for k, bd in enumerate(bounds):
            if bd is None:
                continue
            idx, _, _, bnd = bd
            if idx is None:
                np.minimum(low[k], vals[k] - bnd, out=low[k])
            else:
                low[k][idx] = np.minimum(low[k][idx], vals[k][idx] - bnd)
        lower = np.maximum.reduce(low)
        for k, bd in enumerate(bounds):
            if bd is None:
                continue
            idx, yy, e, bnd = bd
            v = vals[k] if idx is None else vals[k][idx]
            lw = lower if idx is None else lower[idx]
            need = (bnd > CLUTTER_EPS / 2) & (np.minimum(wsum * e, v + bnd) >= lw)
            if idx is None:
                s_, c_ = np.nonzero(need)
                ys = yy[s_, c_]
            else:
                sel = np.flatnonzero(need)
                s_, c_, ys = idx[0][sel], idx[1][sel], yy[sel]
            for i in range(0, len(s_), step):
                j = slice(i, i + step)
                b_, n_ = np.divmod(c_[j], N)
                leak = cp[b_] * respT[n_ * K + k]
                vals[k][s_[j], c_[j]] = cell_pd(ys[j][:, None], leak, lp) @ cwq
        return np.maximum.reduce(vals).reshape(len(x), B, N) @ wn

    return _chunked_shifts(scale, 2 * rho.size, chunk)


_ENVELOPES = {}


def _curvature_envelope(lp):
    """Grid edges and, per interval, an upper bound of exp(lp/(1+y)) (lp^2 g1^2 + 2|lp| h) with
    g1 = y/(1+y)^2, h = y/(1+y)^3 over that interval of y (last interval: to infinity).

    exp(lp/(1+y)) increases in y; g1 peaks at y = 1 and h at y = 1/2, so each factor's maximum
    over [a, b] is at b or at the clipped peak.
    """
    if lp not in _ENVELOPES:
        edges = np.concatenate([[0.0], np.logspace(-4, 8, 481)])
        a, b = edges[:-1], edges[1:]
        g1 = lambda y: y / (1 + y) ** 2
        h = lambda y: y / (1 + y) ** 3
        g1m, hm = g1(np.clip(1.0, a, b)), h(np.clip(0.5, a, b))
        env = np.exp(lp / (1 + b)) * (lp * lp * g1m ** 2 + abs(lp) * 2 * hm)
        tail = lp * lp * g1(edges[-1]) ** 2 + abs(lp) * 2 * h(edges[-1])
        _ENVELOPES[lp] = (edges, np.append(env, tail))
    return _ENVELOPES[lp]


# -- expected CLEAN points on the target ----------------------------------------------------------
POINTS_PATCH_M = 0.25      # patch spacing along a visible face
POINTS_POSE_NODES = 5      # Gauss-Hermite nodes per principal position axis
POINTS_HEADING_NODES = 3   # heading nodes when the heading spread is positive


def surface_patches(center, heading, L, W, rsu, offsets=None, spacing=POINTS_PATCH_M):
    """Visible box faces as lines of patches: (xy (P, 2), sigma_dbsm (P,)) for a 0 dBsm
    dominant scatterer.

    Each visible face carries the power of its box-model scatterers (``box_scatterers``): its
    centre and half of each corner it shares with another visible face (all of an unshared
    corner), spread uniformly over patches ``spacing`` apart. The total RCS equals the box
    model's, so the calibration of the dominant RCS carries over. Empty when no face is visible.
    """
    keys, xy, sig, dom, _ = box_scatterers(center, heading, L, W, rsu, 0.0, offsets)
    if dom is None:
        return np.zeros((0, 2)), np.zeros(0)
    pw = dict(zip(keys, 10 ** (np.asarray(sig) / 10)))
    pos = dict(zip(keys, xy))
    c = np.asarray(center, float)
    ex = np.array([math.cos(heading), math.sin(heading)])
    ey = np.array([-math.sin(heading), math.cos(heading)])
    faces = {"front": (ex * L / 2, ey * W / 2), "rear": (-ex * L / 2, ey * W / 2),
             "left": (ey * W / 2, ex * L / 2), "right": (-ey * W / 2, ex * L / 2)}
    vis = {f: v for f, v in faces.items() if "c_" + f in pw}
    corner_keys = [k for k in keys if k.startswith("k_")]

    def corners_of(off, half):
        out = []
        for sgn in (-1, 1):
            p_ = c + off + sgn * half
            out.append(min(corner_keys, key=lambda k: float(np.sum((pos[k] - p_) ** 2))))
        return out

    share = {k: 0 for k in corner_keys}
    for off, half in vis.values():
        for k in corners_of(off, half):
            share[k] += 1
    out_xy, out_s = [], []
    for f, (off, half) in vis.items():
        power = pw["c_" + f] + sum(pw[k] / share[k] for k in corners_of(off, half))
        length = 2 * float(np.linalg.norm(half))
        n = max(1, int(math.ceil(length / spacing)))
        t = (np.arange(n) + 0.5) / n * 2 - 1
        out_xy.append(c + off + t[:, None] * half)
        out_s.append(np.full(n, 10 * math.log10(power / n)))
    return np.concatenate(out_xy), np.concatenate(out_s)


def peak_floor_lp(power, leak, lp, peak, floor_db):
    """Element-wise log-Pfa for CLEAN's threshold max(eta, peak - floor).

    ``power`` = 1 + leak + target SNR is each cell's mean power over thermal noise and ``peak``
    the mean power of the reference peak (broadcast). The thermal threshold is eta (1 + leak);
    the relative one is peak / 10^(floor/10). Returned as log-Pfa on the cell's noise-plus-
    leakage scale, i.e. -max(eta, peak / (F (1 + leak))), for ``cell_pd``.
    """
    eta = -lp
    rel = np.asarray(peak, float) / 10 ** (floor_db / 10) / (1 + np.asarray(leak, float))
    return -np.maximum(eta, rel)


def expected_points(parts, pred, rsu, bore, bandwidth, sigma, weights, lp=None,
                    peak_floor_db=None, max_points=None, peak_floor_ref="target"):
    """Expected number of detected receive-image cells on the target, per call.

    ``parts`` = [(W, n_re, clutter)] (several parts add their energy, as in multi-grant looks).
    The target is the box's visible surface (``surface_patches``) at each node of a coarse pose
    quadrature (Gauss-Hermite on the principal position axes, plus heading). Patch returns add
    in power within the receive image's range x angle resolution cells (range step c / (2 B),
    angle step 2 / M_rx in u, cells on a fixed grid); each patch falls in its nearest cell, so a
    single reflector gives at most one point, as CLEAN returns one point per resolved peak. Each cell crosses with ``cell_pd`` at its SNR over noise plus
    mean UE leakage; the count is averaged over poses and the RCS (and calibration) nodes
    ``sigma``/``weights``, as P_D is. Without an extent the target is one point.
    ``peak_floor_db`` adds CLEAN's relative threshold: a cell must also reach the reference peak
    minus ``peak_floor_db``. The reference is the mean power of the strongest target cell at each
    pose and RCS node ("target", a single-target gate) or the larger of that and the UE's own echo
    ("global", a shared image). Using the mean of the strongest cell rather than the random
    maximum is an approximation (it makes the relative threshold slightly low). ``max_points``
    caps the expected count per pose and RCS node (CLEAN's point budget).
    Returns (expected points, info dict). The count saturates at the number of occupied cells;
    ``info["points_sum_snr_db"]`` (expected SINR summed over the target's cells, dB) does not,
    and is the companion feature for calibrating against observed CLEAN points.
    """
    lp = math.log(V.P["pfa"]) if lp is None else lp
    x, P = pred["x"], pred["P"]
    lam, E = np.linalg.eigh(0.5 * (P[:2, :2] + P[:2, :2].T))
    lam = np.maximum(lam, 0.0)
    z, wz = np.polynomial.hermite_e.hermegauss(POINTS_POSE_NODES)
    wz = wz / wz.sum()
    sh = float(pred.get("heading_std") or 0.0)
    if sh > 0 and POINTS_HEADING_NODES > 1:
        zh, wh = np.polynomial.hermite_e.hermegauss(POINTS_HEADING_NODES)
        wh = wh / wh.sum()
    else:
        zh, wh = np.zeros(1), np.ones(1)
    poses = []
    for i, a in enumerate(z):
        for j, b in enumerate(z):
            xy = x[:2] + E[:, 0] * math.sqrt(lam[0]) * a + E[:, 1] * math.sqrt(lam[1]) * b
            for k, hz in enumerate(zh):
                poses.append((xy, pred["heading"] + sh * hz, wz[i] * wz[j] * wh[k]))
    pts, sig, pw_ = [], [], []
    for xy, h, w in poses:
        if pred.get("extent") is not None:
            pxy, ps = surface_patches(xy, h, *pred["extent"], rsu)
        else:
            pxy, ps = np.zeros((0, 2)), np.zeros(0)
        if not len(pxy):
            pxy, ps = xy[None], np.zeros(1)
        pts.append(pxy)
        sig.append(ps)
        pw_.append(w)
    n_p = max(len(p) for p in pts)
    # pad to a common patch count (padding patches carry no power)
    XY = np.stack([np.vstack([p, np.repeat(p[-1:], n_p - len(p), 0)]) for p in pts])
    S = np.stack([np.concatenate([s, np.full(n_p - len(s), -np.inf)]) for s in sig])
    pw_ = np.asarray(pw_)
    u, r, th, _ = V.to_polar(XY.reshape(-1, 2), rsu, bore, V.P["rsu_h"], V.P["rc_h"])
    u, r, th = (a.reshape(XY.shape[:2]) for a in (u, r, th))
    du, dr = 2.0 / V.P["M_rx"], V.C / (2 * bandwidth)
    uc = np.arange(math.floor(u.min() / du) - 1, math.ceil(u.max() / du) + 2) * du
    rc = np.arange(math.floor(r.min() / dr) - 1, math.ceil(r.max() / dr) + 2) * dr
    UC, RC = (a.ravel() for a in np.meshgrid(uc, rc, indexing="ij"))
    # each patch's power goes to its nearest resolution cell: CLEAN returns one point per
    # resolved peak and removes the point-spread sidelobes
    iu = np.rint(u / du).astype(int) - (math.floor(u.min() / du) - 1)
    ir = np.rint(r / dr).astype(int) - (math.floor(r.min() / dr) - 1)
    cell = iu * len(rc) + ir
    coup = np.zeros(u.shape + (len(UC),))
    np.put_along_axis(coup, cell[:, :, None], 1.0, axis=2)
    snr = 0.0
    leak = 0.0
    ue_peak = 0.0
    for W, n_re, clutter in parts:
        W = np.atleast_2d(W)
        g = array_gain(W, u)[0]
        rho = np.where(np.isfinite(S), 10 ** (V.rc_snr_db(g, u, r, np.where(np.isfinite(S), S, 0.0),
                                                         th, n_re) / 10), 0.0)
        snr = snr + np.einsum("np,npc->nc", rho, coup)
        if clutter is not None:
            cu, cr, cs, ct = clutter[:4]
            cu = np.atleast_1d(cu)
            cw = np.asarray(clutter[4]) if len(clutter) > 4 else np.ones(len(cu)) / len(cu)
            cp = 10 ** (V.rc_snr_db(array_gain(W, cu)[0], cu, cr, cs, ct, n_re) / 10)
            leak = leak + (cp * cw) @ (V.rx_rel(UC[None] - cu[:, None])
                                       * delay_power(RC[None] - cr, bandwidth))
            ue_peak = ue_peak + float(cp @ cw)
    sinr = snr / (1 + leak)
    sk, wk = np.asarray(sigma, float), np.asarray(weights, float)
    if peak_floor_ref not in ("target", "global"):
        raise ValueError(peak_floor_ref)
    leak_a = np.broadcast_to(leak, snr.shape)
    count = 0.0
    for s_, w_ in zip(sk, wk):
        x = 10 ** (s_ / 10)
        if peak_floor_db is None:
            per = cell_pd(x * sinr, None, lp).sum(axis=1)
        else:
            power = 1 + leak_a + x * snr
            peak = power.max(axis=1, keepdims=True)
            if peak_floor_ref == "global":
                peak = np.maximum(peak, 1 + ue_peak)
            per = cell_pd(x * snr, leak_a, peak_floor_lp(power, leak_a, lp, peak,
                                                         peak_floor_db)).sum(axis=1)
        if max_points is not None:
            per = np.minimum(per, max_points)
        count += w_ * (per @ pw_)
    total = float(wk @ 10 ** (sk / 10)) * float(sinr.sum(axis=1) @ pw_)
    return float(count), dict(points_cells=int(len(UC)), points_poses=int(len(pw_)),
                              points_patches=int(n_p), points_du=du, points_dr_m=dr,
                              points_sum_snr_db=float(10 * np.log10(max(total, 1e-30))),
                              points_peak_floor_db=peak_floor_db, points_peak_floor_ref=peak_floor_ref)


def extended_pd(W, nodes, n_re, bandwidth, clutter=None, focus=None, threshold=None):
    """Uncertainty-averaged P_D of the extended target at the RCS stored in ``nodes``.

    With a ``threshold`` the clutter-free value (an upper bound) screens beams first; screened-out
    beams return -1. Returns (pd, clutter-free pd).
    """
    screen = None if threshold is None else (threshold, np.ones(1))
    pd, upper = extended_pd_shifts(W, nodes, n_re, bandwidth, clutter, focus, (0.0,), screen)
    return pd[0], upper[0]


class ExtendedDetector:
    """Drop-in replacement for ``detection.predictive_pd`` in the paper's batch simulation.

    The simulation predicts the dominant return's (u, r) with covariance (sigma_u, s_r, c_ur). This
    maps that prediction to the box: the predicted reflector is the dominant face centre, and the
    polar covariance becomes a horizontal one through the inverse Jacobian. Heading is the frame's
    heading, known as in the paper. ``prior`` is the paper's effective RCS at the full-band reference
    focus; the dominant scatterer RCS is lowered by the box's own corner/face contribution to that
    focus, so both models see the same effective RCS at the reference cell.
    ``set_frame`` must be called for every frame (``simulation_engine.compat_arrays(frame_hook=...)``).
    """

    resource_kernel_version = "extended-box-v1"

    def __init__(self, threshold=None, calibration_bandwidth=None):
        self.threshold = threshold
        self.cal_bw = V.P["B"] if calibration_bandwidth is None else calibration_bandwidth
        self.frame = None
        self._nodes = {}

    def set_frame(self, row, rsu, bore):
        from .rt_extended import vehicle_extent

        cls = row.get("category") or row.get("target_class")
        L, W = vehicle_extent(cls)
        keys, xy, sig, dom, body = box_scatterers(row["target_xy_m"], row["heading_rad"], L, W,
                                                  rsu, 0.0)
        u, r, th, _ = V.to_polar(xy, rsu, bore, V.P["rsu_h"], V.P["rc_h"])
        i = keys.index(dom)
        gain = 10 ** (sig / 10) * (r[i] / r) ** 4 * 10 ** (
            (2 * V.elem_dbi(np.arcsin(u), th) - 2 * V.elem_dbi(np.arcsin(u[i]), th[i])) / 10)
        focus = V.rx_rel(u[i] - u) * delay_power(r[i] - r, self.cal_bw)
        tx = array_gain(V.beam(u[i]), u)[0] / array_gain(V.beam(u[i]), [u[i]])[0, 0]
        self.frame = dict(rsu=np.asarray(rsu, float), bore=float(bore), heading=row["heading_rad"],
                          extent=(L, W), body_dom=body[i],
                          delta_db=float(10 * np.log10(np.sum(gain * focus * tx))))
        self._nodes = {}

    def nodes(self, u_hat, r_hat, sigma_u, s_r, c_ur, prior):
        key = (u_hat, r_hat, sigma_u, s_r, c_ur, prior)
        if key not in self._nodes:
            fr = self.frame
            rsu, bore = fr["rsu"], fr["bore"]
            dh = V.P["rsu_h"] - V.P["rc_h"]
            d2 = math.sqrt(max(r_hat**2 - dh**2, 1e-6))
            az = math.asin(float(np.clip(u_hat, -1, 1))) + bore
            refl = rsu + d2 * np.array([math.cos(az), math.sin(az)])
            centre = refl - fr["body_dom"] @ rot(fr["heading"]).T
            from .tracker import polar_jacobian

            J = polar_jacobian(refl, rsu, bore)
            Sp = np.array([[sigma_u**2, c_ur], [c_ur, s_r**2]])
            Ji = np.linalg.inv(J)
            if len(self._nodes) > 64:
                self._nodes.clear()
            self._nodes[key] = pose_nodes(centre, Ji @ Sp @ Ji.T, fr["heading"], 0.0, fr["extent"],
                                          rsu, bore, prior - fr["delta_db"])
        return self._nodes[key]

    def __call__(self, W, u_hat, r_hat, sigma_u, sigma_prior, theta_deg, n_re, bg=None,
                 clutter=None, s_r=None, c_ur=None):
        s_r = V.P["sigma_r"] if s_r is None else s_r
        c_ur = 0.0 if c_ur is None else c_ur
        nodes = self.nodes(u_hat, r_hat, sigma_u, s_r, c_ur, sigma_prior)
        bw = bg if bg is not None else V.P["ded_prb"] * 12 * V.P["df"]
        pd = extended_pd(W, nodes, n_re, bw, clutter, threshold=self.threshold)[0]
        if self.threshold is not None and clutter is not None and not (pd >= 0).any():
            pd = np.full(len(np.atleast_2d(W)), -1.0)
        return pd, V.db(array_gain(W, [u_hat])[:, 0])
