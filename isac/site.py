"""Site and model configuration injected into the beam front end.

The paper code reads its parameters from module globals in ``model.py`` (``P``, ``NOISE_RX``,
``to_polar`` and so on). ``SiteConfig`` describes another deployment: any RSU position, height and
boresight, radio parameters, detection calibration, cone-angle directions and road geometry.
``BeamFrontEnd(site=...)`` applies it only for the duration of each front-end call through
``SiteConfig.active()``, and the globals are restored afterwards. A field left at ``None`` keeps
the paper value, so ``SiteConfig()`` and ``site=None`` reproduce the paper pipeline exactly.

Directions: the paper uses u = sin(azimuth - boresight). With ``cone_angle=True``,
u = cos(elevation) sin(azimuth - boresight), the cone angle a horizontal ULA's phase actually sees.
``model.beam(u)`` weights then reach the full array gain M on a 32x1 panel. The difference is up to
about 1/8 beamwidth at 6 m height.

Detection calibration (applied to the target, not to UE clutter):
- ``snr_offset_db`` subtracts a fitted model-minus-RT SNR offset.
- ``snr_sigma_db`` adds a log-normal SNR uncertainty. P_D is then averaged over it with
  Gauss-Hermite nodes, i.e. a predicted rather than plug-in value.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional
import math
import numpy as np
from . import model as V


@dataclass
class SiteConfig:
    """Deployment and model parameters. ``None`` keeps the paper value (``data/model_parameters.json``).

    Geometry: ``rsu_xy`` (m), ``boresight_rad`` (scene azimuth of the panel normal), ``rsu_h``,
    ``rc_h`` (reflector height), ``ue_h``. ``cone_angle`` selects the direction convention.
    Radio: ``M``/``M_rx`` elements, ``fc`` (Hz), ``p_ref_dbm`` (the paper's reference power; boresight
    EIRP = p_ref + 10 log10 M + element gain) or ``eirp_dbm`` instead, ``nf_rx``/``nf_ue`` (dB),
    ``noise_bw`` (Hz, the bandwidth of the noise reference, separate from the subcarrier spacing
    ``df``), ``l_proc`` (dB), ``inr`` (dB), ``element_pattern`` (False: isotropic 0 dBi),
    ``e_max`` (dBi).
    Detection: ``pfa``, ``snr_offset_db``, ``snr_sigma_db``, ``sigma_hermite_nodes``.
    Resolution: ``sensing_bandwidth_hz`` is the bandwidth of the sensing range processing. It sets
    range-cell coupling and the delay CRB only; the integration gain still comes from the grant's
    REs. None uses the grant bandwidth (n_prb x 12 x df), as in the paper.
    Scene: ``rcs_prior_dbsm`` (design RCS used when the tracker sends none), ``interference``
    (corridor interference in ``model.ue_link``), ``ue_clutter``.
    ``extra`` holds any further ``model.P`` keys to override.
    """

    rsu_xy: Optional[tuple] = None
    boresight_rad: Optional[float] = None
    rsu_h: Optional[float] = None
    rc_h: Optional[float] = None
    ue_h: Optional[float] = None
    cone_angle: bool = False
    M: Optional[int] = None
    M_rx: Optional[int] = None
    fc: Optional[float] = None
    p_ref_dbm: Optional[float] = None
    eirp_dbm: Optional[float] = None
    nf_rx: Optional[float] = None
    nf_ue: Optional[float] = None
    noise_bw: Optional[float] = None
    df: Optional[float] = None
    l_proc: Optional[float] = None
    inr: Optional[float] = None
    element_pattern: bool = True
    e_max: Optional[float] = None
    pfa: Optional[float] = None
    snr_offset_db: float = 0.0
    snr_sigma_db: float = 0.0
    sigma_hermite_nodes: int = 5
    sensing_bandwidth_hz: Optional[float] = None
    rcs_prior_dbsm: Optional[float] = None
    interference: Optional[bool] = None
    ue_clutter: Optional[bool] = None
    extra: dict = field(default_factory=dict)

    def overrides(self):
        """``model.P`` keys this site sets."""
        keys = ("rsu_h", "rc_h", "ue_h", "M", "M_rx", "fc", "p_ref_dbm", "nf_rx", "nf_ue",
                "df", "l_proc", "inr", "e_max", "pfa", "interference", "ue_clutter")
        out = {k: getattr(self, k) for k in keys if getattr(self, k) is not None}
        if self.noise_bw is not None:
            out["B"] = self.noise_bw
        if self.eirp_dbm is not None:
            m = self.M if self.M is not None else V.P["M"]
            e = self.e_max if self.e_max is not None else V.P["e_max"]
            out["p_ref_dbm"] = self.eirp_dbm - 10 * math.log10(m) - (e if self.element_pattern else 0.0)
        out.update(self.extra)
        return out

    def resolution_bandwidth(self, grant_bw):
        return grant_bw if self.sensing_bandwidth_hz is None else float(self.sensing_bandwidth_hz)

    def is_default(self):
        return (not self.overrides() and not self.cone_angle and self.element_pattern
                and self.snr_offset_db == 0.0 and self.snr_sigma_db == 0.0)

    def snr_nodes(self):
        """Offsets (dB, added to the target RCS) and weights for the detection calibration."""
        if self.snr_sigma_db > 0:
            x, w = np.polynomial.hermite_e.hermegauss(self.sigma_hermite_nodes)
            return -self.snr_offset_db + self.snr_sigma_db * x, w / w.sum()
        return np.array([-self.snr_offset_db]), np.ones(1)

    @contextmanager
    def active(self):
        """Apply the site to ``model``'s globals for the duration of the block, then restore."""
        if self.is_default():
            yield self
            return
        names = ("NOISE_RX", "NOISE_UE", "LAM", "M", "N_IDX", "GRID", "BW", "to_polar", "elem_dbi")
        saved_p, saved = dict(V.P), {n: getattr(V, n) for n in names}
        try:
            V.P.update(self.overrides())
            V.M = V.P["M"]
            V.N_IDX = np.arange(V.M) - (V.M - 1) / 2
            V.GRID = np.arange(-V.M, V.M) / V.M
            V.BW = 2 / V.M
            V.LAM = V.C / V.P["fc"]
            V.NOISE_RX = -174 + 10 * math.log10(V.P["B"]) + V.P["nf_rx"]
            V.NOISE_UE = -174 + 10 * math.log10(V.P["B"]) + V.P["nf_ue"]
            if self.cone_angle:
                V.to_polar = cone_to_polar
            if not self.element_pattern:
                V.elem_dbi = lambda phi, theta: np.zeros(np.broadcast(np.asarray(phi), np.asarray(theta)).shape)
            yield self
        finally:
            V.P.clear()
            V.P.update(saved_p)
            for n, v in saved.items():
                setattr(V, n, v)


def cone_to_polar(xy, rsu, bore, h_tx, h_rx):
    """``model.to_polar`` with the cone angle u = cos(elevation) sin(azimuth - boresight)."""
    d = xy - rsu
    az = np.arctan2(d[:, 1], d[:, 0])
    d2 = np.hypot(d[:, 0], d[:, 1])
    r = np.hypot(d2, h_tx - h_rx)
    u = np.sin(az - bore) * d2 / r
    theta = 90 + np.degrees(np.arctan2(h_tx - h_rx, d2))
    return u, r, theta, np.cos(az - bore)


# -- served UEs on an arbitrary road ------------------------------------------------------------

def road_points(origin_xy, heading_rad, lane_offsets, s_range, spacing, kinds=None):
    """UE points along a straight road of any direction.

    ``origin_xy`` is a point on the road axis, ``heading_rad`` its direction, ``lane_offsets`` the
    signed lateral offsets of lanes and footways (m, left positive), and ``s_range`` = (s0, s1) the
    along-road span. ``kinds`` gives "vehicle" or "sidewalk" per lane (default: vehicle).
    Returns (xy, kind).
    """
    e = np.array([math.cos(heading_rad), math.sin(heading_rad)])
    n = np.array([-e[1], e[0]])
    s = np.arange(s_range[0], s_range[1] + 1e-9, spacing)
    kinds = kinds or ["vehicle"] * len(lane_offsets)
    xy, kind = [], []
    for off, k in zip(lane_offsets, kinds):
        for si in s:
            xy.append(np.asarray(origin_xy, float) + si * e + off * n)
            kind.append(k)
    return np.array(xy), np.array(kind)


def ue_link_site(xy, kind, rsu, bore, rng, interference_dbm=None):
    """Serving-grid SINR and grid beam for UEs anywhere around the site.

    Same link budget as ``model.ue_link`` (path loss, element pattern, UE gain, penetration loss
    with the paper's spread), but no corridor geometry. ``interference_dbm`` (scalar or per UE) is
    external interference arriving at the UE's position outside the vehicle; like the paper's
    corridor interference it is attenuated by the same penetration loss, then added to thermal
    noise. ``None`` means noise only.
    Call inside ``SiteConfig.active()`` to use the site's radio parameters.
    """
    u, r, th, _ = V.to_polar(np.atleast_2d(xy), rsu, bore, V.P["rsu_h"], V.P["ue_h"])
    phi = np.arcsin(np.clip(u, -1, 1))
    pl = 32.4 + 21 * np.log10(r) + 20 * math.log10(V.P["fc"] / 1e9)
    veh = np.asarray(kind) == "vehicle"
    pen = np.where(veh, V.P["pen_vehicle"], 0.0)
    if V.P["pen_sd"] > 0:
        pen = np.where(veh, np.maximum(rng.normal(V.P["pen_vehicle"], V.P["pen_sd"], len(u)), 0.0), 0.0)
    G = V.gain(np.array([V.beam(g) for g in V.GRID]), u)
    k0 = np.argmax(G, axis=0)
    s = (V.P["p_ref_dbm"] + 10 * math.log10(V.M) + V.elem_dbi(phi, th) + V.P["g_ue"] - pl - pen
         + V.db(G[k0, np.arange(len(u))]))
    noise = V.NOISE_UE
    if interference_dbm is not None:
        noise = V.db(10 ** (V.NOISE_UE / 10) + 10 ** ((np.asarray(interference_dbm) - pen) / 10))
    return s - noise, k0, u, r, th
