"""Exact finite-subcarrier leakage for the revised conditional point detector."""

import math
import numpy as np
from scipy.special import ndtr
from . import model as V


def delay_amplitude(dr, bandwidth):
    k = int(round(bandwidth / V.P["df"]))
    t = 2 * V.P["df"] * np.asarray(dr) / V.C
    denominator = np.sin(np.pi * t)
    out = np.ones_like(t, dtype=float)
    regular = abs(denominator) > 1e-12
    np.divide(np.sin(k * np.pi * t), k * denominator, out=out, where=regular)
    out = np.where(regular, out, np.cos((k - 1) * np.pi * np.rint(t)))
    return out


def delay_power(dr, bandwidth):
    return np.clip(delay_amplitude(dr, bandwidth) ** 2, 0.0, 1.0)


def predictive_pd(
    W,
    u_hat,
    r_hat,
    sigma_u,
    sigma_prior,
    theta_deg,
    n_re,
    bg=None,
    clutter=None,
    s_r=None,
    c_ur=None,
    _compatibility_threshold=None,
):
    W = np.atleast_2d(W)
    z, wz = V.pointing_nodes(sigma_u)
    uz = u_hat + sigma_u * z
    tx = V.gain(W, uz)
    sr = V.P["sigma_r"] if s_r is None else s_r
    cov = 0.0 if c_ur is None else c_ur
    sc = math.sqrt(max(sr**2 - (cov / sigma_u) ** 2, 0.0)) if sigma_u > 0 else sr
    mean = r_hat + (cov / sigma_u * z if sigma_u > 0 else 0.0 * z)
    if sc > 0:
        edges = np.linspace(-4, 4, V.P["n_r_cells"] + 1)
        mids = (edges[:-1] + edges[1:]) / 2
        edges[0], edges[-1] = (-np.inf, np.inf)
        mass = np.diff(ndtr(edges))
    else:
        mids, mass = (np.zeros(1), np.ones(1))
    rho1 = 10 ** (V.rc_snr_db(tx, uz, 1.0, sigma_prior, theta_deg, n_re) / 10)
    lp = math.log(V.P["pfa"])
    r = mean[None, :] + sc * mids[:, None]
    plain = np.exp(lp / (1 + rho1[:, None, :] / r[None, :, :] ** 4))
    upper = np.einsum("r,brz,z->b", mass, plain, wz, optimize=False)
    if clutter is None or bg is None:
        return (upper, V.db(tx))
    active = np.arange(len(W))
    if _compatibility_threshold is not None:
        active = np.flatnonzero(upper >= _compatibility_threshold - 1e-13)
        if not len(active):
            return (np.full(len(W), -1.0), V.db(tx))
    rho1 = rho1[active]
    cu, cr, cs, ct = clutter[:4]
    cu = np.atleast_1d(cu)
    cw = np.asarray(clutter[4]) if len(clutter) > 4 else np.ones(1)
    cp = 10 ** (V.rc_snr_db(V.gain(W[active], cu), cu, cr, cs, ct, n_re) / 10)
    leak = cp.T[:, :, None] * V.rx_rel(uz[None, None, :] - cu[:, None, None])
    pd = np.zeros(len(active))
    disturbance = np.empty_like(leak)
    denominator = np.empty_like(leak)
    for x, m in zip(mids, mass):
        r = mean + sc * x
        rho = rho1 / r[None, :] ** 4
        np.multiply(leak, delay_power(r - cr, bg)[None, None, :], out=disturbance)
        disturbance += 1
        np.add(disturbance, rho[None, :, :], out=denominator)
        np.divide(disturbance, denominator, out=disturbance)
        disturbance *= lp
        np.exp(disturbance, out=disturbance)
        pd += m * (np.einsum("q,qbz->bz", cw, disturbance, optimize=False) @ wz)
    result = np.full(len(W), -1.0)
    result[active] = pd
    return (result, V.db(tx))


def compatibility_screen(*args, **kwargs):
    return predictive_pd(*args, **kwargs, _compatibility_threshold=0.9)


predictive_pd.resource_kernel_version = "finite-subcarrier-matched-filter-v1"
