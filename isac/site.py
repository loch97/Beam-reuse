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
    Scene: ``rcs_prior_dbsm`` (design RCS used when the tracker sends none),
    ``rcs_prior_by_class`` ({cls: dBsm}, used instead when ``TrackReport.cls`` is listed), ``interference``
    (corridor interference in ``model.ue_link``), ``ue_clutter``.
    ``extra`` holds any further ``model.P`` keys to override.

    Explicit link budget (aliases; give either name, not both): ``tx_power_dbm`` = total conducted
    TX power (= ``p_ref_dbm``; boresight EIRP = tx_power + 10 log10 M + element gain),
    ``noise_figure_db`` (= ``nf_rx``), ``noise_bandwidth_hz`` (= ``noise_bw``, the noise reference
    bandwidth: per-RE SNR = P / (k T F B)), ``element_gain_dbi`` (= ``e_max``, element pattern peak,
    applied on transmit and receive). ``SiteConfig.fr2_38dbm(...)`` is the local FR2 preset.
    Fading: ``speckle_m`` is the Gamma (Nakagami) order of each cell's power for the extended
    model: 1 (None) = Rayleigh speckle, as before; larger = less fluctuation; ``float("inf")`` =
    deterministic cell power (Marcum Q). The paper's point detector stays Swerling 1.
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
    tx_power_dbm: Optional[float] = None
    noise_figure_db: Optional[float] = None
    noise_bandwidth_hz: Optional[float] = None
    element_gain_dbi: Optional[float] = None
    speckle_m: Optional[float] = None
    rcs_prior_dbsm: Optional[float] = None
    rcs_prior_by_class: Optional[dict] = None
    interference: Optional[bool] = None
    ue_clutter: Optional[bool] = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        for a, b in (("tx_power_dbm", "p_ref_dbm"), ("noise_figure_db", "nf_rx"),
                     ("noise_bandwidth_hz", "noise_bw"), ("element_gain_dbi", "e_max")):
            if getattr(self, a) is not None:
                if getattr(self, b) is not None and getattr(self, b) != getattr(self, a):
                    raise ValueError(f"give {a} or {b}, not both")
                setattr(self, b, getattr(self, a))
        if self.speckle_m is not None and not self.speckle_m > 0:
            raise ValueError("speckle_m must be positive (inf: deterministic)")

    @classmethod
    def fr2_38dbm(cls, **kw):
        """Local mimo_tracking FR2 link budget: 38 dBm TX over 190.08 MHz, NF 7 dB, 5 dBi
        elements (the PSD of a 41 dBm / 400 MHz FR2 BS; source: a summary of the TR 38.901
        calibration tables, not checked against the spec), sensing at full-band resolution.
        Any field can be overridden by keyword."""
        base = dict(tx_power_dbm=38.0, noise_figure_db=7.0, noise_bandwidth_hz=190.08e6,
                    element_gain_dbi=5.0, sensing_bandwidth_hz=190.08e6)
        base.update(kw)
        return cls(**base)

    @classmethod
    def mimo_v6(cls, **kw):
        """Local mimo_tracking calibration v6 (2026-10-29, after the carrier-phase fix).

        ``fr2_38dbm`` plus the detection calibration fitted on 918 track dwells of the
        simple_street_canyon_with_cars validation scene (40 worlds, 32x1 panel, CLEAN detector,
        footprint + 1 m matching), front end 8fd8a63:
        - ``pfa`` = exp(-10^1.3) (2.2e-9), the false-alarm rate of CLEAN's 13 dB threshold;
        - ``snr_offset_db`` 19.4 and ``snr_sigma_db`` 4.3: the "swerling_spread" fit (single
          Swerling curve with a log-normal SNR error; NLL 314.1 vs 331.3 for a plain 18.9 dB
          offset). Taken to use the CLEAN Pfa, like the 18.9 dB fit; check this with the fit;
        - ``rcs_prior_by_class``: the design RCS -1.1 dBsm shifted by the per-class offsets of the
          plain fit relative to its pooled 18.9 dB (car 23.7, bus 17.7, truck 17.0 dB), i.e.
          car -5.9, bus +0.1, truck +0.8 dBsm. Cyclists and pedestrians were never detected and
          have no entry (they fall back to ``rcs_prior_dbsm``).
        The offset was fitted to a summary SNR with a single-cell curve, not to the front end's
        own any-cell, pose-averaged event, so it is an approximate calibration: verify it in
        closed loop or refit it against ``BeamDecision.pd_offset_curve``. It is specific to that
        scene, detector and matching rule, hence a preset and not a default.
        """
        base = dict(pfa=math.exp(-10 ** 1.3), snr_offset_db=19.4, snr_sigma_db=4.3,
                    rcs_prior_dbsm=-1.1,
                    rcs_prior_by_class={"car": -1.1 - (23.7 - 18.9), "bus": -1.1 - (17.7 - 18.9),
                                        "truck": -1.1 - (17.0 - 18.9)})
        base.update(kw)
        return cls.fr2_38dbm(**base)

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
        if self.speckle_m is not None:
            out["speckle_m"] = float(self.speckle_m)
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


def young_track_extent(min_extent=(4.5, 1.9), min_hits=3, class_extents=None):
    """Extent prior for ``BeamFrontEnd(extent_prior=...)``.

    Returns a function of a TrackReport giving the box extent to plan with. A track with fewer
    than ``min_hits`` hits (``TrackReport.hits``; unknown counts as young) gets at least
    ``min_extent`` per axis; with ``class_extents`` ({cls: (L, W)}, e.g. from
    ``rt_extended.vehicle_extent``) a known class also gets at least its class box. A track
    without extent gets the prior box. Older tracks keep their own extent.
    """
    def prior(track):
        ext = track.extent
        young = track.hits is None or track.hits < min_hits
        if not young:
            return ext
        floor = np.asarray(min_extent, float)
        if class_extents and track.cls in class_extents:
            floor = np.maximum(floor, np.asarray(class_extents[track.cls], float))
        if ext is None:
            return tuple(float(v) for v in floor)
        return tuple(float(v) for v in np.maximum(np.asarray(ext, float), floor))

    return prior
