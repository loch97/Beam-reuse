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

from dataclasses import dataclass
from pathlib import Path
import json
import math
import numpy as np
from scipy.special import ndtr
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


def pose_nodes(mean_xy, cov_xy, heading, heading_std, extent, rsu, bore, sig_dom, h_rc=None):
    """Cell quadrature over position (principal axes of ``cov_xy``) and heading.

    The cell width on each axis keeps the induced direction step below ``ext_u_cell`` and below
    1/``ext_cells_per_sigma`` standard deviations, capped at ``ext_max_cells`` cells per axis.
    Heading nodes are added when the heading spread moves the box ends by more than
    ``ext_heading_min_disp_m``. Visibility and the dominant face are fixed at the mean pose.
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
    xy = np.concatenate(
        [centers[:, None, :] + (body @ rot(heading + heading_std * z).T)[None] for z in zh]
    )
    w = np.concatenate([wxy * x for x in wh])
    n, k = xy.shape[:2]
    u, r, th, _ = V.to_polar(xy.reshape(-1, 2), rsu, bore, V.P["rsu_h"], h_rc)
    return PoseNodes(
        u.reshape(n, k), r.reshape(n, k), th.reshape(n, k), w / w.sum(), sig, keys.index(dom), keys
    )


def array_gain(W, u):
    """|a(u)^T w|^2 / M without the byte-keyed response cache used by the batch simulation."""
    W = np.atleast_2d(W)
    A = np.exp(1j * np.pi * np.multiply.outer(np.asarray(u, float), V.N_IDX))
    return np.abs(np.tensordot(W, A, axes=([1], [-1]))) ** 2 / V.M


def cell_snr(W, nodes, n_re, bandwidth):
    """Mean SNR in each scatterer's receive cell: (beams, nodes, cells), linear."""
    tx = array_gain(W, nodes.u)
    rho = 10 ** (V.rc_snr_db(tx, nodes.u, nodes.r, nodes.sigma_dbsm, nodes.theta, n_re) / 10)
    du = nodes.u[:, :, None] - nodes.u[:, None, :]
    dr = nodes.r[:, :, None] - nodes.r[:, None, :]
    coupling = V.rx_rel(du) * delay_power(dr, bandwidth)
    return np.einsum("bnk,nck->bnc", rho, coupling, optimize=True)


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
    unit = float(np.sum(10 ** (V.rc_snr_db(1.0, 0.0, r_min, nodes.sigma_dbsm, 90.0, n_re) / 10)))
    lp = math.log(V.P["pfa"])
    scale = 10 ** (np.atleast_1d(np.asarray(shifts_db, float)) / 10)
    pd = np.exp(lp / (1 + scale[:, None] * unit * g[None, :]))
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
            out[i] += cw[q] * (np.exp(lp * (1 + L) / (1 + L + x * rho)) @ nodes.w)
    return out


def extended_pd_shifts(W, nodes, n_re, bandwidth, clutter=None, focus=None, shifts_db=(0.0,),
                       screen=None, rho=None):
    """P_D of the extended target for each RCS shift and beam: arrays (shifts, beams).

    Every scatterer RCS in ``nodes`` moves by the same shift (dB), so a shift evaluates the target
    at another dominant RCS. ``clutter`` = (cu, cr, cs, ct[, cw]) as in ``detection.predictive_pd``:
    UE-reflector directions ``cu`` with quadrature weights ``cw``, range ``cr``, RCS ``cs`` and
    elevation ``ct``. ``screen`` = (threshold, shift weights): beams whose shift-averaged
    clutter-free value (an upper bound) is below the threshold skip the clutter evaluation and
    return -1. ``rho`` passes precomputed ``cell_snr(W, nodes, n_re, bandwidth)``.
    Returns (pd, clutter-free pd).
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
    upper = np.stack([np.exp(lp / (1 + x * best)) @ nodes.w for x in scale])
    if clutter is None:
        return upper, upper
    active = np.arange(len(W))
    if screen is not None:
        threshold, weights = screen
        active = np.flatnonzero(np.asarray(weights) @ upper >= threshold - 1e-11)
    out = np.full(upper.shape, -1.0)
    if not len(active):
        return out, upper
    rho = rho[active]
    cu, cr, cs, ct = clutter[:4]
    cu = np.atleast_1d(cu)
    cw = np.asarray(clutter[4]) if len(clutter) > 4 else np.ones(len(cu)) / len(cu)
    cp = 10 ** (V.rc_snr_db(array_gain(W[active], cu), cu, cr, cs, ct, n_re) / 10)
    resp = V.rx_rel(cu_nodes[None] - cu[:, None, None]) * delay_power(cr_nodes - cr, bandwidth)
    for i, x in enumerate(scale):
        pd = np.zeros(rho.shape)
        for q in range(len(cu)):
            leak = cp[:, q, None, None] * resp[q][None]
            pd += cw[q] * np.exp(lp * (1 + leak) / (1 + leak + x * rho))
        out[i, active] = pd.max(axis=2) @ nodes.w
    return out, upper


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
