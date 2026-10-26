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
class SensingRequirement:
    """Back-end measurement-quality need for one look, in addition to P_D >= tau.

    ``min_cell_snr_db``: the predicted SNR (over noise plus UE leakage) of the strongest scatterer
    cell, averaged in dB over pose and RCS uncertainty, must reach this value.
    ``min_cells`` with ``quality_snr_db``: the expected number of scatterer cells whose SNR reaches
    ``quality_snr_db`` must be at least ``min_cells``.
    ``quality_logistic`` = (a, b) with ``min_quality_prob``: an externally fitted calibration
    P(quality | pred_cell_snr_db) = 1 / (1 + exp(-(a + b * pred_cell_snr_db))) must reach
    ``min_quality_prob``. ``pred_cell_snr_db`` is the front end's (calibrated) value.
    Fields left at None impose nothing.
    """

    min_cell_snr_db: Optional[float] = None
    quality_snr_db: Optional[float] = None
    min_cells: Optional[float] = None
    quality_logistic: Optional[tuple] = None
    min_quality_prob: Optional[float] = None

    def active(self):
        return self.min_cell_snr_db is not None or (
            self.min_cells is not None and self.quality_snr_db is not None) or (
            self.quality_logistic is not None and self.min_quality_prob is not None)

    def quality_prob(self, pred_cell_snr_db):
        """P(back end measures well | predicted strongest-cell SNR) from the external logistic
        calibration ``quality_logistic`` = (a, b): 1 / (1 + exp(-(a + b * snr_db))); NaN if unset."""
        if self.quality_logistic is None:
            return np.full(np.shape(pred_cell_snr_db), np.nan)
        a, b = self.quality_logistic
        return 1.0 / (1.0 + np.exp(-(a + b * np.asarray(pred_cell_snr_db, float))))


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
    requirement: Optional[SensingRequirement] = None


@dataclass
class RcsReport:
    """Tracker-side knowledge of the dominant-scatterer RCS at the planning time, in dBsm.

    ``nodes_dbsm``/``weights`` are exact-mass nodes of the predictive RCS distribution; the front
    end averages P_D over them (``rcs_mode="predictive"``). ``design_dbsm`` is its lower
    ``rcs_confidence`` quantile, used alone by ``rcs_mode="design"`` and when no nodes are given.
    ``looks`` counts the looks that entered the posterior.
    """

    design_dbsm: float
    mean_dbsm: Optional[float] = None
    looks: float = 0.0
    source: str = "calibration"
    nodes_dbsm: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    sd_dbsm: Optional[float] = None


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

    mode: str  # "reuse", "reuse_multi", "dedicated" or "skip"
    t: float
    grant: Optional[int] = None
    ue_id: Optional[int] = None
    weights: Optional[np.ndarray] = None
    tag: Optional[int] = None
    pd_pred: float = float("nan")
    pd_curve_sigma: Optional[np.ndarray] = None
    pd_curve: Optional[np.ndarray] = None
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
    rcs_mean_dbsm: float = float("nan")
    n_candidates: int = 0
    info: dict = field(default_factory=dict)
    pred_cell_snr_db: float = float("nan")
    exp_quality_cells: float = float("nan")
    requirement_met: Optional[bool] = None
    candidates: Optional[dict] = None
    resolution_bw: float = float("nan")
    quality_prob: float = float("nan")
    parts: Optional[list] = None
    pd_offset_curve: Optional[tuple] = None


@dataclass
class SensingReport:
    """Sensing back-end result for an executed decision.

    ``z`` = (u, r) of the strongest detected cell, ``R`` its measurement covariance (CRB at the
    measured SNR and the grant bandwidth, plus extent glint). ``stat`` is that cell's power over
    the CFAR disturbance floor, ``eta`` the threshold on the same scale and ``floor_db`` the floor
    over thermal noise (10 log10(1 + UE leakage)). ``snr_db`` = 10 log10(stat - 1). These describe
    the strongest crossing cell and serve the tracker.

    ``rcs_detected``/``rcs_stat``/``rcs_floor_db`` are reported on every look for the designated
    cell, the dominant scatterer's receive cell. That cell is chosen independently of the data, so
    its crossing and power are the RCS evidence. The strongest-cell values above are
    selection-biased and are not used for RCS.
    """

    t: float
    detected: bool
    z: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    snr_db: Optional[float] = None
    stat: Optional[float] = None
    eta: Optional[float] = None
    floor_db: Optional[float] = None
    rcs_detected: Optional[bool] = None
    rcs_stat: Optional[float] = None
    rcs_floor_db: Optional[float] = None
