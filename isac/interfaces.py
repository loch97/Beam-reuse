"""Messages exchanged between the beam front end and the tracking, scheduling and sensing modules.

Data flow per sensing opportunity::

    tracker  --TrackReport-->  front end  <--GrantReport--  scheduler
                                   |
                               BeamDecision  -->  scheduler (MCS override) and sensing back end
                                   |
    tracker  <--SensingReport--  sensing back end

All positions are horizontal RSU-scene coordinates in metres, times in seconds. Polar quantities use
the paper's convention: u = sin(azimuth - boresight), r = 3-D range to the reflector height.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class TrackReport:
    """Tracker output for one target at time ``t``.

    ``x`` = (px, py, vx, vy) and ``P`` its 4x4 covariance. ``extent`` = (L, W) of the box hypothesis
    and ``cls`` the class label, both optional. ``heading``/``heading_std`` override the
    velocity-derived heading when the tracker estimates orientation separately.
    """

    track_id: int
    t: float
    x: np.ndarray
    P: np.ndarray
    extent: Optional[tuple] = None
    cls: Optional[str] = None
    heading: Optional[float] = None
    heading_std: Optional[float] = None


@dataclass
class RcsReport:
    """Tracker-side RCS knowledge for the dominant scatterer, in dBsm.

    ``design_dbsm`` is the value the front end plans with (a conservative lower confidence bound
    once enough looks exist, otherwise the calibration prior). ``looks`` is the effective number of
    detections behind the estimate.
    """

    design_dbsm: float
    mean_dbsm: Optional[float] = None
    looks: float = 0.0
    source: str = "calibration"


@dataclass
class GrantReport:
    """One downlink grant, as measured and decided by the scheduler.

    ``sinr_db`` is the measured (CQI/SRS) post-scheduling SINR on the serving grid beam ``k0``;
    ``mcs`` the MCS index the scheduler would use on it. ``ue_range_m`` comes from timing advance and
    positions the UE reflector as clutter. ``ue_u``/``ue_u_std`` give an SRS direction estimate; when
    absent the UE direction is only known to lie in the grid-beam cell (U1).
    """

    ue_id: int
    t: float
    k0: int
    n_prb: int
    n_sym: int
    sinr_db: float
    mcs: int
    ue_range_m: float
    ue_theta_deg: float
    ue_kind: str = "vehicle"
    ue_u: Optional[float] = None
    ue_u_std: Optional[float] = None
    max_mcs_drop: int = 3


@dataclass
class BeamDecision:
    """Front-end output for one sensing opportunity."""

    mode: str  # "reuse", "dedicated" or "skip"
    t: float
    grant: Optional[int] = None
    ue_id: Optional[int] = None
    weights: Optional[np.ndarray] = None
    tag: Optional[int] = None
    pd_pred: float = float("nan")
    ue_loss_db: float = 0.0
    mcs: Optional[int] = None
    dse: float = 0.0
    cost_bits: float = 0.0
    n_re: int = 0
    bandwidth: float = 0.0
    u_hat: float = float("nan")
    r_hat: float = float("nan")
    sigma_u: float = float("nan")
    s_r: float = float("nan")
    rcs_design_dbsm: float = float("nan")
    n_candidates: int = 0
    info: dict = field(default_factory=dict)


@dataclass
class SensingReport:
    """Sensing back-end result for an executed decision.

    ``z`` = (u, r) of the strongest detected cell, ``R`` its measurement covariance (CRB at the
    measured SNR and the grant bandwidth, plus extent glint). ``snr_db`` is the per-cell SNR estimate
    after integration and ``unit_snr_db`` the SNR the same beam would give for a 0 dBsm dominant
    scatterer at the reported cell, so that ``snr_db - unit_snr_db`` estimates the dominant RCS.
    """

    t: float
    detected: bool
    z: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    snr_db: Optional[float] = None
    unit_snr_db: Optional[float] = None
