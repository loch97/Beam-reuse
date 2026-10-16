"""ULA beams, link budgets, prediction covariance and effective RCS."""

from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from scipy.special import ndtr, ndtri

PARAMS = json.loads((ROOT / "data/model_parameters.json").read_text())
P = PARAMS.copy()
C = 299792458.0
LAM = C / P["fc"]
M = P["M"]
N_IDX = np.arange(M) - (M - 1) / 2
GRID = np.arange(-32, 32) / 32
BW = 2 / M
NOISE_RX = -174 + 10 * math.log10(P["B"]) + P["nf_rx"]
NOISE_UE = -174 + 10 * math.log10(P["B"]) + P["nf_ue"]
SITES = ("center", "corner", "offset")
EVAL_SITE = "corner"
LANES_X = (-5.25, -1.75, 1.75, 8.75, 12.25)
_SECTORS = None
SECTOR_META = []


def config():
    P.update(PARAMS)


def levels():
    return (levels_v5()[0], 6)


def predictive_pd(*args, **kwargs):
    from .detection import predictive_pd as evaluate

    return evaluate(*args, **kwargs)


def beam(g, q=0.0):
    return np.exp(
        1j * np.pi * q * (N_IDX / N_IDX.max()) ** 2 - 1j * np.pi * N_IDX * g
    ) / math.sqrt(M)


def gain(W, u):
    """G_w(u) = |a(u)^T w|^2 / M for weights W (k, M) at directions u (...) -> (k, ...)."""
    W = np.atleast_2d(W)
    u = np.asarray(u, float)
    A = np.exp(1j * np.pi * np.multiply.outer(u, N_IDX))
    return np.abs(np.tensordot(W, A, axes=([1], [-1]))) ** 2 / M


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-30))


def elem_dbi(phi_rel_rad, theta_deg):
    a_h = 12 * (np.degrees(phi_rel_rad) / 65) ** 2
    a_v = 12 * ((np.asarray(theta_deg) - 90) / 65) ** 2
    return P["e_max"] - np.minimum(a_h + a_v, 30.0)


def sector_codebook():
    """{kappa: boresight phase vector} of the step-(a) phase-only sector beams that pass acceptance."""
    global _SECTORS
    if _SECTORS is None:
        base = Path(__file__).resolve().parents[1] / "data/sector_codebook"
        rows = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))["rows"]
        with np.load(base.with_suffix(".npz")) as z:
            _SECTORS = {
                r["opt"]["kappa"]: z[f"k{r['opt']['kappa']:g}"]
                for r in rows
                if r["opt"]["pass"]
            }
    return _SECTORS


def sector_tag(kappa):
    return 100 + int(round(10 * kappa))


def sector_beams(g0, u_hat, sigma_u, ue_win=None):
    """Spanning-sector beams: for each codebook kappa, the sector [c -+ kappa W0/2] (W0 = 2/M) that
    contains the UE window g0 +- 1/M and is centred as close as possible to the midpoint of the span
    of the UE window and the target window u_hat +- h, h = max(2 sigma_u, 1/M).
    R19 (sector_rule 'union', ue_win = (lo, hi) the UE design window): sectors must contain the UE
    window; centres = clipped hull midpoint + sector_centres points of the UE-containing range, plus
    the hull midpoint + sector_centres points of the (smaller) range whose sector contains the whole
    hull [min(lo, u_hat - h), max(hi, u_hat + h)], when that range is nonempty. Whether a candidate
    detects is decided by the predictive detector, not by the window."""
    h = max(2 * sigma_u, 1 / M)
    union = True
    u_lo, u_hi = ue_win if union and ue_win is not None else (g0 - 1 / M, g0 + 1 / M)
    lo, hi = (min(u_lo, u_hat - h), max(u_hi, u_hat + h))
    W, tag = ([], [])
    SECTOR_META.clear()
    for k, phi in sector_codebook().items():
        half = k * BW / 2
        c_lo, c_hi = (u_hi - half, u_lo + half)
        cs = [min(max(0.5 * (lo + hi), c_lo), c_hi)]
        if P["sector_centres"] > 1 and c_hi > c_lo:
            cs += list(np.linspace(c_lo, c_hi, P["sector_centres"]))
        if union:
            v_lo, v_hi = (hi - half, lo + half)
            if v_hi >= v_lo:
                cs += [0.5 * (lo + hi)] + (
                    list(np.linspace(v_lo, v_hi, P["sector_centres"]))
                    if P["sector_centres"] > 1 and v_hi > v_lo
                    else []
                )
        for c in cs:
            W.append(np.exp(1j * phi - 1j * np.pi * N_IDX * c) / math.sqrt(M))
            tag.append(sector_tag(k))
            SECTOR_META.append((k, c, bool(c - half <= lo + 1e-12 and c + half >= hi - 1e-12)))
    return (W, tag)


def candidates(u_ue, u_hat, k0, level, sigma_u=0.0, ue_ctrl=None):
    """Admissible weight sets A_level for a UE at u_ue with serving grid beam k0 (section 2).
    R19: ue_ctrl = (u_hat_ue, lo, hi), the controller's reported UE direction and design window
    The two-lobe base uses the reported UE direction. Sectors contain the UE window."""
    g0 = GRID[k0]
    W = [beam(g0)]
    tag = [0]
    if level >= 1:
        for dk in (-1, 1):
            if 0 <= k0 + dk < len(GRID):
                W.append(beam(GRID[k0 + dk]))
                tag.append(1)
    if level >= 2:
        for g in np.linspace(g0 - BW, g0 + BW, P["n_steer"]):
            W.append(beam(g))
            tag.append(2)
    if level >= 3:
        for g in np.linspace(g0 - BW, g0 + BW, P["n_steer"]):
            W.append(beam(g, P["q_wide"]))
            tag.append(3)
    if ue_ctrl is not None:
        b_base = beam(ue_ctrl[0])
    else:
        b_base = beam(u_ue) if P["two_lobe_base"] == "ue" else beam(g0)
    if level >= 4 and P["amp_two_lobe"]:
        for t in np.linspace(0, 1, P["n_mix"]):
            w = (1 - t) * b_base + t * beam(u_hat)
            W.append(w / (np.max(np.abs(w)) * math.sqrt(M)))
            tag.append(4)
    if level >= 5:
        b_ue, b_t = (b_base, beam(u_hat))
        for t in np.linspace(0, 1, P["n_mix_phase"]):
            w = (1 - t) * b_ue + t * b_t
            w = np.where(np.abs(w) < 1e-09 / math.sqrt(M), b_ue, w)
            W.append(np.exp(1j * np.angle(w)) / math.sqrt(M))
            tag.append(5)
    if level >= 6:
        ws, ts = sector_beams(g0, u_hat, sigma_u, None if ue_ctrl is None else ue_ctrl[1:])
        W += ws
        tag += ts
    return (np.array(W), np.array(tag))


def levels_v5():
    ks = sorted(sector_codebook())
    lv = [(0,), (0, 1), (0, 1, 2)]
    names = ["keep", "neighbour", "fine steer"]
    for i, k in enumerate(ks):
        lv.append((0, 1, 2) + tuple((sector_tag(x) for x in ks[: i + 1])))
        names.append(f"sector kappa_max {k:g}")
    lv += [(0, 1, 2, 3), (0, 1, 2, 5)]
    names += ["diag: old quadratic widening", "baseline: phase two-lobe (grid base)"]
    return (tuple(lv), tuple(names))


def gamma_s(pd, pfa):
    return 10 * math.log10(math.log(pfa) / math.log(pd) - 1)


def p_detect(rho_db):
    return np.exp(math.log(P["pfa"]) / (1 + 10 ** (np.asarray(rho_db) / 10)))


def rc_snr_db(Gw, u, r, sigma_dbsm, theta_deg, n_re):
    """Per-RC mean SNR (dB) after integrating n_re REs; transmit gain Gw (linear), focused receive."""
    phi = np.arcsin(np.clip(u, -1, 1))
    e = elem_dbi(phi, theta_deg)
    path = 10 * np.log10(LAM**2 / ((4 * math.pi) ** 3 * np.asarray(r) ** 4)) + sigma_dbsm
    return (
        P["p_ref_dbm"]
        + 10 * math.log10(M)
        + db(Gw)
        + 10 * math.log10(P["M_rx"])
        + 2 * e
        + path
        - NOISE_RX
        + 10 * math.log10(n_re)
        - P["l_proc"]
        - 10 * math.log10(1 + 10 ** (P["inr"] / 10))
    )


def rx_rel(d):
    """Relative gain of a full-aperture digital receive beam at offset d (u units)."""
    d = np.asarray(d, float)
    safe = np.where(np.abs(d) < 1e-12, 1e-12, d)
    g = (np.sin(P["M_rx"] * np.pi * safe / 2) / (P["M_rx"] * np.sin(np.pi * safe / 2))) ** 2
    return np.where(np.abs(d) < 1e-12, 1.0, g)


def pointing_nodes(sigma_u):
    """Gaussian cell quadrature with exact cell masses."""
    if sigma_u <= 0:
        z, wz = np.polynomial.hermite_e.hermegauss(P["n_z"])
        return (z, wz / wz.sum())
    zmax = P["z_max"]
    hz = min(1.0 / P["cells_per_sigma"], P["u_cell"] / sigma_u)
    n = int(math.ceil(2 * zmax / hz))
    edges = np.linspace(-zmax, zmax, n + 1)
    cdf = ndtr(edges)
    cdf[0], cdf[-1] = (0.0, 1.0)
    return (0.5 * (edges[:-1] + edges[1:]), np.diff(cdf))


def track_covariance(xy, rsu, bore, heading, with_J=False):
    """Round-7 signed model: covariance of the predicted (u, r) of a dominant return at horizontal
    position xy. Steady-state 4-state CV Kalman filter in road axes (e_par, e_perp), discrete white
    acceleration Q = sigma_a^2 [[T^4/4, T^3/2], [T^3/2, T^2]], measurement of position with
    R_road = J^-1 diag(s_um^2, s_rm^2) J^-T, s_um^2 = 6 / (pi^2 S (M_rx^2 - 1)),
    s_rm = c sqrt(3) / (sqrt(8) pi B sqrt(S)), B = prb_prev * 12 * df; then gap_k misses:
    P_k = F^{k+1} P+ F^{k+1}' + sum_j F^j Q F^j'. Returns (s_u, s_r, c_ur)."""
    T = P["T_epoch"]
    d = np.asarray(xy, float) - rsu
    h = float(np.hypot(*d))
    n = d / h
    dh = P["rsu_h"] - P["rc_h"]
    r = math.hypot(h, dh)
    a = np.array([-math.sin(bore), math.cos(bore)])
    u = float(a @ n)
    if P["psi_deg"] is not None:
        ps = math.radians(P["psi_deg"])
        e_par = math.cos(ps) * n + math.sin(ps) * np.array([-n[1], n[0]])
    else:
        e_par = np.array([math.cos(heading), math.sin(heading)])
    E = np.column_stack([e_par, [-e_par[1], e_par[0]]])
    J = np.vstack([(a - u * n) / h, d / r]) @ E
    S = 10 ** (P["snr_prev_db"] / 10)
    s_um2 = 6 / (math.pi**2 * S * (P["M_rx"] ** 2 - 1))
    s_rm = (
        C
        * math.sqrt(3)
        / (math.sqrt(8) * math.pi * P["prb_prev"] * 12 * P["df"] * math.sqrt(S))
    )
    Ji = np.linalg.inv(J)
    R = Ji @ np.diag([s_um2, s_rm**2]) @ Ji.T
    F2 = np.array([[1, T], [0, 1]])
    q = np.array([[T**4 / 4, T**3 / 2], [T**3 / 2, T**2]])
    F = np.kron(np.eye(2), F2)
    Q = np.zeros((4, 4))
    Q[:2, :2], Q[2:, 2:] = (P["sigma_a_par"] ** 2 * q, P["sigma_a_perp"] ** 2 * q)
    H = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])
    Pp = np.eye(4) * 100.0
    for _ in range(5000):
        Pm = F @ Pp @ F.T + Q
        K = Pm @ H.T @ np.linalg.inv(H @ Pm @ H.T + R)
        Pn = (np.eye(4) - K @ H) @ Pm
        if np.max(np.abs(Pn - Pp)) < 1e-14:
            Pp = Pn
            break
        Pp = Pn
    Fk = np.linalg.matrix_power(F, P["gap_k"] + 1)
    Pk = Fk @ Pp @ Fk.T + sum(
        (
            np.linalg.matrix_power(F, j) @ Q @ np.linalg.matrix_power(F, j).T
            for j in range(P["gap_k"] + 1)
        )
    )
    Sig = J @ H @ Pk @ H.T @ J.T
    out = (math.sqrt(Sig[0, 0]), math.sqrt(Sig[1, 1]), float(Sig[0, 1]))
    return out + (J[:, 0] * math.sqrt(Pk[0, 0]),) if with_J else out


def vehicle_rcs(row, rsu, sig_dom):
    L, W_ = row["truth_extent_LWH_m"][:2]
    c = np.array(row["target_xy_m"], float)
    hd = row["heading_rad"]
    ex, ey = (np.array([math.cos(hd), math.sin(hd)]), np.array([-math.sin(hd), math.cos(hd)]))
    faces = {
        "front": (c + ex * L / 2, ex, ey * W_ / 2),
        "rear": (c - ex * L / 2, -ex, ey * W_ / 2),
        "left": (c + ey * W_ / 2, ey, ex * L / 2),
        "right": (c - ey * W_ / 2, -ey, ex * L / 2),
    }
    los_best, dom = (-1, None)
    pts, sig = ({}, {})
    for name, (cf, nf, half) in faces.items():
        to_rsu = rsu - cf
        vis = float(nf @ to_rsu)
        if vis <= 0:
            continue
        cos_inc = vis / np.linalg.norm(to_rsu)
        if cos_inc > los_best:
            los_best, dom = (cos_inc, name)
        pts["c_" + name] = cf
        for s in (-1, 1):
            corner = cf + s * half
            pts["k_%.2f_%.2f" % tuple(corner)] = corner
    for k in pts:
        if k == "c_" + dom:
            sig[k] = sig_dom
        elif k.startswith("k_"):
            sig[k] = sig_dom + P["secondary_rc_offsets_db"]["corner"]
        else:
            sig[k] = sig_dom + P["secondary_rc_offsets_db"]["other_face"]
    keys = list(pts)
    xy = np.array([pts[k] for k in keys])
    return (keys, xy, np.array([sig[k] for k in keys]), "c_" + dom)


def to_polar(xy, rsu, bore, h_tx, h_rx):
    d = xy - rsu
    az = np.arctan2(d[:, 1], d[:, 0])
    d2 = np.hypot(d[:, 0], d[:, 1])
    r = np.hypot(d2, h_tx - h_rx)
    u = np.sin(az - bore)
    theta = 90 + np.degrees(np.arctan2(h_tx - h_rx, d2))
    return (u, r, theta, np.cos(az - bore))


def aspect_bin(aspect_deg):
    return "headon" if aspect_deg < 25 else "oblique"


def rcs_prior(recs, abin=None, q=20):
    """Controller prior: q-th percentile of the effective RCS pooled over classes AND aspects on the
    training sites (causal: the controller knows neither class nor aspect; the 'abin' argument is
    accepted for interface compatibility and ignored)."""
    v = [x["sigma"] for x in recs if x["site"] != EVAL_SITE]
    return float(np.percentile(v, q))


def rcs_truth(recs, key):
    """Truth for an evaluation frame: its own RT dominant-cell effective RCS."""
    return next((x["sigma"] for x in recs if x["site"] == EVAL_SITE and x["key"] == key))


def rcs_class_median(recs, cls):
    """Served-UE vehicle reflector (car) and summaries: training-site class median."""
    return float(
        np.median([x["sigma"] for x in recs if x["site"] != EVAL_SITE and x["cls"] == cls])
    )


def load_mcs():
    rows = json.loads((ROOT / "data/mcs_thresholds.json").read_text())["rows"]
    rows = sorted(rows, key=lambda r: r["mcs_index"])
    out, run = ([], -1000000000.0)
    for r in rows:
        v = r.get("raw_snr_threshold_db_at_evm")
        if v is None:
            continue
        run = max(run, float(v))
        out.append((int(r["mcs_index"]), run, float(r["spectral_efficiency"])))
    return out


def mcs_of(snr_db, lo_idx=None):
    ok = [m for m in MCS if snr_db + 1e-09 >= m[1] and (lo_idx is None or m[0] >= lo_idx)]
    return ok[-1] if ok else None


def cell_ymin(rsu):
    """Far end of the served road segment: 'grid' keeps the step-1 span (y >= -80); 'corridor' serves
    the segment in front of the panel up to the next RSU along the road (last 8 m excluded)."""
    return rsu[1] - P["isd"] + 8.0 if P["deploy"] == "corridor" else -80.0


def corridor_interference_dbm(pts, rsu, bore, rng):
    """Road corridor: co-oriented panel RSUs at rsu + (0, k ISD), k = +-1..+-grid_n, along the same road,
    LOS (same street; blockage out of scope). An active neighbour (prob. 'load') serves a uniformly
    drawn point of its own segment (our segment shifted by k ISD) with its strongest grid beam; toward
    our UE it radiates with that beam's actual array gain and its element pattern (30 dB front-to-back).
    """
    tot = np.zeros(len(pts))
    y_far, y_near = (cell_ymin(rsu), rsu[1] - 7.0)
    for k in (k for k in range(-P["grid_n"], P["grid_n"] + 1) if k):
        site = rsu + np.array([0.0, k * P["isd"]])
        if rng.random() >= P["load"]:
            continue
        srv = np.array([[rng.choice(LANES_X), k * P["isd"] + rng.uniform(y_far, y_near)]])
        us = to_polar(srv, site, bore, P["rsu_h"], P["ue_h"])[0]
        wn = beam(GRID[int(np.argmax(gain(np.array([beam(g) for g in GRID]), us)[:, 0]))])
        u, r, th, cosb = to_polar(pts, site, bore, P["rsu_h"], P["ue_h"])
        phi = np.where(
            cosb >= 0, np.arcsin(np.clip(u, -1, 1)), math.pi - np.arcsin(np.clip(u, -1, 1))
        )
        pl = 32.4 + 21 * np.log10(r) + 20 * math.log10(P["fc"] / 1000000000.0)
        tot += 10 ** (
            (
                P["p_ref_dbm"]
                + 10 * math.log10(M)
                + db(gain(wn, u)[0])
                + elem_dbi(phi, th)
                + P["g_ue"]
                - pl
            )
            / 10
        )
    return db(tot)


def ue_link(u, r, th, kind, pts, rsu, rng, bore=None):
    """Serving-grid SINR with corridor interference and vehicle penetration loss."""
    phi = np.arcsin(np.clip(u, -1, 1))
    pl = 32.4 + 21 * np.log10(r) + 20 * np.log10(P["fc"] / 1000000000.0)
    pen = np.where(kind == "vehicle", P["pen_vehicle"], 0.0)
    if P["pen_sd"] > 0:
        pen = np.where(
            kind == "vehicle",
            np.maximum(rng.normal(P["pen_vehicle"], P["pen_sd"], len(u)), 0.0),
            0.0,
        )
    G = gain(np.array([beam(g) for g in GRID]), u)
    k0 = np.argmax(G, axis=0)
    quant = db(G[k0, np.arange(len(u))])
    s = P["p_ref_dbm"] + 10 * math.log10(M) + elem_dbi(phi, th) + P["g_ue"] - pl - pen + quant
    noise = NOISE_UE
    if P["interference"]:
        i_dbm = corridor_interference_dbm(pts, rsu, bore, rng) - pen
        noise = db(10 ** (NOISE_UE / 10) + 10 ** (i_dbm / 10))
    return (s - noise, k0)


MCS = load_mcs()
