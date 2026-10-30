"""Back-end target tracker and RCS estimator feeding the beam front end.

The tracker is a constant-velocity EKF in scene coordinates, state (px, py, vx, vy). Its process
noise is the paper's discrete white acceleration with separate along-road and cross-road
intensities. Measurements are the (u, r) of the strongest detected cell. The tracker maps that cell
to the box centre through the dominant scatterer's body-frame offset. Measurement noise is the
CRB at the *measured* SNR and the *granted* bandwidth, so beam and grant choices feed back into the
next prediction's uncertainty. ``track_covariance`` in ``model.py`` is the steady-state special case
with a fixed SNR, a fixed bandwidth and a fixed number of missed epochs.
"""

import math
import numpy as np
from . import model as V
from .interfaces import TrackReport
from .target import FP, box_scatterers


def polar_jacobian(xy, rsu, bore):
    """d(u, r)/d(px, py) at reflector position ``xy``; rows (u, r).

    Analytic for the paper's u = sin(azimuth - boresight); central differences of the active
    ``model.to_polar`` otherwise (e.g. the cone angle of ``site.SiteConfig``).
    """
    from .site import cone_to_polar

    if V.to_polar is cone_to_polar:
        h, out = 1e-4, np.zeros((2, 2))
        for k in range(2):
            dp = np.zeros(2)
            dp[k] = h
            out[:, k] = (polar(np.asarray(xy, float) + dp, rsu, bore)
                         - polar(np.asarray(xy, float) - dp, rsu, bore)) / (2 * h)
        return out
    d = np.asarray(xy, float) - rsu
    a = np.array([-math.sin(bore), math.cos(bore)])
    if float(np.hypot(*d)) < 1e-3:  # reflector at the RSU mast (ghost track): use the boresight side
        d = 1e-3 * np.array([math.cos(bore), math.sin(bore)])
    h = float(np.hypot(*d))
    n = d / h
    r = math.hypot(h, V.P["rsu_h"] - V.P["rc_h"])
    return np.vstack([(a - float(a @ n) * n) / h, d / r])


def polar(xy, rsu, bore):
    u, r, _, _ = V.to_polar(np.atleast_2d(xy), rsu, bore, V.P["rsu_h"], V.P["rc_h"])
    return np.array([u[0], r[0]])


def crb_polar(snr_db, bandwidth):
    """Diagonal (u, r) measurement covariance: full-aperture ULA angle CRB and delay CRB."""
    S = 10 ** (snr_db / 10)
    s_u2 = 6 / (math.pi**2 * S * (V.P["M_rx"] ** 2 - 1))
    s_r = V.C * math.sqrt(3) / (math.sqrt(8) * math.pi * bandwidth * math.sqrt(S))
    return np.diag([s_u2, s_r**2])


def process_noise(T, heading):
    """White-acceleration Q for state (px, py, vx, vy) with road-axis intensities."""
    R = np.array([[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]])
    A = R @ np.diag([V.P["sigma_a_par"] ** 2, V.P["sigma_a_perp"] ** 2]) @ R.T
    return np.block([[T**4 / 4 * A, T**3 / 2 * A], [T**3 / 2 * A, T**2 * A]])


def transition(T):
    F = np.eye(4)
    F[0, 2] = F[1, 3] = T
    return F


def heading_of(x, P):
    """Heading and its standard deviation from the velocity estimate (road prior when slow)."""
    vx, vy = x[2], x[3]
    sp2 = vx * vx + vy * vy
    if sp2 < FP["slow_speed_mps"] ** 2:
        return FP["road_heading_rad"], FP["heading_std_slow_rad"]
    g = np.array([-vy, vx]) / sp2
    return math.atan2(vy, vx), math.sqrt(max(float(g @ P[2:, 2:] @ g), 0.0))


def predict_state(x, P, dt):
    """CV prediction over ``dt`` in epoch steps, so a gap accumulates the same Q as missed epochs."""
    if dt < -1e-9:
        raise ValueError("cannot predict backwards")
    x, P = np.array(x, float), np.array(P, float)
    if dt <= 1e-12:
        return x, P
    h, _ = heading_of(x, P)
    n = max(1, int(round(dt / V.P["T_epoch"])))
    F, Q = transition(dt / n), process_noise(dt / n, h)
    for _ in range(n):
        x, P = F @ x, F @ P @ F.T + Q
    return x, P


class CVTracker:
    """Single-target CV EKF with a box extent and a dominant-scatterer measurement model."""

    def __init__(self, track_id, t, xy, v_xy, rsu, bore, extent, cls=None, P0=None):
        self.track_id, self.t = track_id, float(t)
        self.x = np.array([*xy, *v_xy], float)
        if P0 is None:
            P0 = np.diag(
                [FP["track_init_pos_std_m"] ** 2] * 2 + [FP["track_init_vel_std_mps"] ** 2] * 2
            )
        self.P = np.array(P0, float)
        self.rsu, self.bore = np.asarray(rsu, float), float(bore)
        self.extent, self.cls = tuple(extent), cls
        self.misses = 0

    def report(self, t=None):
        """TrackReport predicted to ``t`` (no state change)."""
        x, P = self.predicted(self.t if t is None else t)
        return TrackReport(self.track_id, self.t if t is None else t, x, P, self.extent, self.cls)

    def predicted(self, t):
        return predict_state(self.x, self.P, float(t) - self.t)

    def predict(self, t):
        self.x, self.P = self.predicted(t)
        self.t = float(t)

    def scatterer_offsets(self):
        """Body offsets of the visible scatterers at the current mean pose, and the dominant index."""
        h, _ = heading_of(self.x, self.P)
        keys, xy, _, dom, _ = box_scatterers(self.x[:2], h, *self.extent, self.rsu, 0.0)
        if dom is None:
            return np.zeros((1, 2)), 0
        return xy - self.x[:2], keys.index(dom)

    def update(self, report):
        """EKF update with a SensingReport. A miss only advances time.

        The detected cell is associated with the visible scatterer of smallest Mahalanobis
        distance; its offset maps the box centre to the measured reflector.
        """
        self.predict(report.t)
        if not report.detected:
            self.misses += 1
            return False
        offs, _ = self.scatterer_offsets()
        best = None
        for off in offs:
            refl = self.x[:2] + off
            H = np.hstack([polar_jacobian(refl, self.rsu, self.bore), np.zeros((2, 2))])
            nu = np.asarray(report.z, float) - polar(refl, self.rsu, self.bore)
            S = H @ self.P @ H.T + report.R
            d2 = float(nu @ np.linalg.solve(S, nu))
            if best is None or d2 < best[0]:
                best = (d2, H, nu, S)
        _, H, nu, S = best
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ report.R @ K.T
        self.misses = 0
        return True


def glint_polar(xy, rsu, bore):
    """Extent glint of the strongest-cell position, mapped to (u, r)."""
    J = polar_jacobian(xy, rsu, bore)
    return J @ (FP["ext_glint_m"] ** 2 * np.eye(2)) @ J.T


def steady_state_polar(xy, rsu, bore, heading, snr_db, n_prb, gap_k):
    """Steady-state tracker covariance after ``gap_k`` missed epochs, mapped to (u, r).

    Uses the same filter as ``model.track_covariance`` with Cartesian rather than road-axis state
    and the measurement covariance supplied by ``crb_polar``. Returns (s_u, s_r, c_ur).
    """
    T = V.P["T_epoch"]
    J = polar_jacobian(xy, rsu, bore)
    Ji = np.linalg.inv(J)
    R = Ji @ crb_polar(snr_db, n_prb * 12 * V.P["df"]) @ Ji.T
    F, Q = transition(T), process_noise(T, heading)
    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    Pp = np.eye(4) * 100.0
    for _ in range(5000):
        Pm = F @ Pp @ F.T + Q
        K = Pm @ H.T @ np.linalg.inv(H @ Pm @ H.T + R)
        Pn = (np.eye(4) - K @ H) @ Pm
        done = np.max(np.abs(Pn - Pp)) < 1e-14
        Pp = Pn
        if done:
            break
    for _ in range(gap_k + 1):
        Pp = F @ Pp @ F.T + Q
    Sig = J @ Pp[:2, :2] @ J.T
    return math.sqrt(Sig[0, 0]), math.sqrt(Sig[1, 1]), float(Sig[0, 1])
