"""Hierarchical RCS estimation from detection events and threshold-truncated detection powers.

Model, from the population down to one look (all RCS values in dBsm, dominant scatterer):

    class c          sigma_bar | c ~ N(mu_c, tau^2 + se_c^2)        (between-target spread)
    target           sigma_i = sigma_bar + delta_i                   (aspect / pose deviation)
                     delta_i AR(1): stationary sd omega, correlation rho_frame per frame_dt
    look i           cell power ~ Exp(1 + 10^((sigma_i + a_i) / 10))  (Swerling 1, noise-normalised)

``RcsPopulation.fit`` estimates mu_c, tau, omega and rho_frame from calibration records with an
inverse-gamma hyperprior on tau^2. The calibration has few targets per class, so tau is shrunk
toward the hyperprior mean. The class is optional; an unknown class gives a mixture prior.

``RcsEstimator`` is a grid filter over (sigma_bar, delta). Each executed look contributes

    p(event, s | sigma) = P_D(sigma)^det (1 - P_D(sigma))^(1-det) * [ f(s | s > eta, sigma) ]^det

where P_D(sigma) is the front end's predicted detection probability of the executed beam as a
function of the dominant RCS. It is averaged over the pose uncertainty at decision time and
includes UE leakage. f(s | s > eta) = exp(-(s - eta) / m) / m is the exponential density of the
reported cell statistic truncated at the detection threshold, with m = 1 + 10^((sigma + a) / 10)
and a the unit-RCS cell SNR at the posterior pose. The product of the two factors is the full
likelihood of what the back end reports. Using only the truncated factor (``likelihood="truncated"``)
ignores misses but stays unbiased. Plain detection averages are biased high because only
threshold crossings are kept. Beam choices depend only on past observations, so they do not bias
the likelihood.
"""

from collections import defaultdict
from dataclasses import dataclass
import json
import math
import numpy as np
from scipy.special import ndtr
from . import model as V
from .interfaces import RcsReport
from .target import FP, ROOT


def _frame_dt():
    rows = json.loads((ROOT / "data/scene_frames.json").read_text())["rows"]
    by_traj = defaultdict(list)
    for r in rows:
        by_traj[r["key"].rsplit("__", 1)[0]].append(r["t_s"])
    return float(np.median(np.concatenate([np.diff(sorted(t)) for t in by_traj.values()])))


@dataclass
class RcsPopulation:
    """Class-level prior and target-level variability of the dominant RCS (dB)."""

    mu: dict
    prior_sd: dict
    weight: dict
    tau: float
    omega: float
    rho_frame: float
    frame_dt: float

    @classmethod
    def fit(cls, records, exclude_site=V.EVAL_SITE, frame_dt=None):
        """Variance components from calibration records (one record = one frame of one target).

        Targets are (site, key without frame suffix). omega^2 is the pooled within-target
        variance and rho_frame the pooled lag-1 correlation of within-target residuals.
        tau^2 combines the method-of-moments between-target estimate (df = targets - classes)
        with an inverse-gamma hyperprior of mean ``rcs_tau0_db``^2 and shape ``rcs_tau_a0``.
        """
        groups = defaultdict(list)
        for r in records:
            if r["site"] == exclude_site:
                continue
            stem, frame = r["key"].rsplit("__", 1)
            groups[(r["site"], stem, r.get("cls", r.get("category")))].append(
                (int(frame.lstrip("f")), r["sigma"])
            )
        means, ss_w, df_w, num, den = defaultdict(list), 0.0, 0, 0.0, 0.0
        n_frames = []
        for (_, _, c), rows in groups.items():
            s = np.array([v for _, v in sorted(rows)])
            e = s - s.mean()
            means[c].append(s.mean())
            ss_w, df_w = ss_w + float(e @ e), df_w + len(s) - 1
            num, den = num + float(e[:-1] @ e[1:]), den + float(e @ e)
            n_frames.append(len(s))
        omega2 = ss_w / df_w if df_w else FP["rcs_omega0_db"] ** 2
        rho = float(np.clip(num / den, 0.0, 0.99)) if den > 0 else 0.0
        nbar = float(np.mean(n_frames))
        mu = {c: float(np.mean(m)) for c, m in means.items()}
        ss_b = sum(float(np.sum((np.array(m) - mu[c]) ** 2)) for c, m in means.items())
        df_b = sum(len(m) for m in means.values()) - len(means)
        tau2_mm = max(ss_b / df_b - omega2 / nbar, 0.0) if df_b > 0 else 0.0
        a0 = FP["rcs_tau_a0"]
        b0 = (a0 - 1) * FP["rcs_tau0_db"] ** 2
        tau2 = (b0 + 0.5 * df_b * tau2_mm) / (a0 + 0.5 * df_b - 1)
        sd = {c: math.sqrt(tau2 + (tau2 + omega2 / nbar) / len(m)) for c, m in means.items()}
        n = sum(len(m) for m in means.values())
        w = {c: len(m) / n for c, m in means.items()}
        return cls(mu, sd, w, math.sqrt(tau2), math.sqrt(omega2), rho,
                   _frame_dt() if frame_dt is None else frame_dt)

    def class_weights(self, cls=None):
        if cls in self.mu:
            return {cls: 1.0}
        return dict(self.weight)


def _cell_masses(edges, mean, sd):
    c = ndtr((edges - mean) / sd)
    c[..., 0], c[..., -1] = 0.0, 1.0
    return np.diff(c, axis=-1)


@dataclass
class RcsLook:
    """Evidence from one executed look, assembled by the front end.

    ``pd_sigma``/``pd`` is the predicted P_D of the executed beam versus dominant RCS. For a
    detection, ``stat`` is the reported cell power over the disturbance floor, ``eta`` the
    threshold on that scale and ``unit_db`` the unit-RCS cell SNR over the floor at the posterior
    pose.
    """

    t: float
    detected: bool
    pd_sigma: np.ndarray
    pd: np.ndarray
    stat: float = None
    eta: float = None
    unit_db: float = None


class RcsEstimator:
    """Grid posterior over (sigma_bar, delta) for one track."""

    def __init__(self, population, cls=None, t0=0.0, likelihood=None):
        self.pop = population
        self.likelihood = FP["rcs_likelihood"] if likelihood is None else likelihood
        if self.likelihood not in ("event+truncated", "truncated"):
            raise ValueError(self.likelihood)
        weights = population.class_weights(cls)
        self.cls = cls if cls in population.mu else None
        step = FP["rcs_grid_db"]
        lo = min(population.mu[c] - 6 * population.prior_sd[c] for c in weights)
        hi = max(population.mu[c] + 6 * population.prior_sd[c] for c in weights)
        edges = np.arange(math.floor(lo / step) * step, hi + step, step)
        self.sbar = 0.5 * (edges[:-1] + edges[1:])
        prior_b = sum(w * _cell_masses(edges, population.mu[c], population.prior_sd[c])
                      for c, w in weights.items())
        om = population.omega
        if om > 0:
            nd = FP["rcs_delta_nodes"]
            dedges = np.linspace(-FP["rcs_delta_z"] * om, FP["rcs_delta_z"] * om, nd + 1)
            self.delta = 0.5 * (dedges[:-1] + dedges[1:])
            self._dedges = dedges
            prior_d = _cell_masses(dedges, 0.0, om)
        else:
            self.delta, self._dedges, prior_d = np.zeros(1), None, np.ones(1)
        self.p = np.outer(prior_b, prior_d)
        self.p /= self.p.sum()
        self.t = float(t0)
        self.looks = self.detections = 0

    # -- dynamics -------------------------------------------------------------------------------
    def _transition(self, dt):
        """AR(1) transition of delta over dt as exact cell masses, rows = from."""
        om = self.pop.omega
        phi = self.pop.rho_frame ** (dt / self.pop.frame_dt)
        sd = om * math.sqrt(max(1 - phi * phi, 1e-12))
        return _cell_masses(self._dedges[None, :], phi * self.delta[:, None], sd)

    def _advanced(self, t):
        dt = float(t) - self.t
        if dt < -1e-9:
            raise ValueError("cannot advance backwards")
        if dt <= 1e-12 or self._dedges is None:
            return self.p
        return self.p @ self._transition(dt)

    def advance(self, t):
        self.p = self._advanced(t)
        self.t = max(self.t, float(t))

    # -- inference ------------------------------------------------------------------------------
    def sigma(self):
        return self.sbar[:, None] + self.delta[None, :]

    def log_likelihood(self, look):
        s = self.sigma()
        ll = np.zeros_like(s)
        if self.likelihood == "event+truncated":
            pd = np.clip(np.interp(s, look.pd_sigma, look.pd), 1e-12, 1 - 1e-12)
            ll += np.log(pd) if look.detected else np.log1p(-pd)
        if look.detected and look.stat is not None:
            m = 1 + 10 ** ((s + look.unit_db) / 10)
            ll += -(look.stat - look.eta) / m - np.log(m)
        return ll

    def update(self, look):
        self.advance(look.t)
        ll = self.log_likelihood(look)
        if not np.any(ll):
            return
        p = self.p * np.exp(ll - ll.max())
        total = p.sum()
        if not total > 0:
            raise FloatingPointError("RCS posterior vanished")
        self.p = p / total
        self.looks += 1
        self.detections += int(look.detected)

    def predictive(self, t):
        """(sigma values, masses) of sigma_t = sigma_bar + delta_t, flattened."""
        p = self._advanced(t)
        return self.sigma().ravel(), p.ravel()

    def report(self, t=None):
        """RcsReport at time t with exact-mass nodes of the predictive sigma distribution."""
        s, p = self.predictive(self.t if t is None else t)
        order = np.argsort(s, kind="stable")
        s, p = s[order], p[order]
        cdf = np.cumsum(p)
        k = FP["rcs_nodes"]
        nodes, weights = [], []
        lo = 0
        for j in range(1, k + 1):
            hi = int(np.searchsorted(cdf, j / k - 1e-12)) + 1 if j < k else len(s)
            hi = max(hi, lo + 1)
            if lo >= len(s):
                break
            w = p[lo:hi].sum()
            if w > 0:
                nodes.append(float(p[lo:hi] @ s[lo:hi] / w))
                weights.append(float(w))
            lo = hi
        mean = float(p @ s)
        design = float(s[min(int(np.searchsorted(cdf, 1 - FP["rcs_confidence"])), len(s) - 1)])
        return RcsReport(
            design_dbsm=design,
            mean_dbsm=mean,
            looks=float(self.looks),
            source="hierarchical" if self.looks else "hierarchical-prior",
            nodes_dbsm=np.array(nodes),
            weights=np.array(weights) / np.sum(weights),
            sd_dbsm=float(math.sqrt(max(p @ (s - mean) ** 2, 0.0))),
        )
