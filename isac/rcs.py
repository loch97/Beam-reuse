"""Hierarchical RCS estimation from detection events and threshold-truncated detection powers.

Model, from the population down to one look (all RCS values in dBsm, dominant scatterer):

    class c          sigma_bar | c ~ N(mu_c, tau^2 + se_c^2)        (between-target spread)
    target           sigma_i = sigma_bar + delta_i                   (aspect / pose deviation)
                     delta_i AR(1): stationary sd omega, correlation rho_frame per frame_dt
    look i           cell power ~ Exp(1 + 10^((sigma_i + a_i) / 10))  (Swerling 1, noise-normalised)

``RcsPopulation.fit`` estimates mu_c, tau, omega and rho_frame from calibration records with an
inverse-gamma hyperprior on tau^2. The calibration has few targets per class, so tau is shrunk
toward the hyperprior mean. The class is optional; an unknown class gives a mixture prior.

``RcsEstimator`` is a grid filter over (sigma_bar, delta). The evidence of a look is the
*designated cell*: the dominant scatterer's receive cell. The cell is fixed before the look, and
the back end reports whether it crossed the threshold and its power s on every look. A look then
contributes

    p(det, s | sigma) = P_d(sigma)^det (1 - P_d(sigma))^(1-det) * [ f(s | s > eta, sigma) ]^det

Both factors come from the cell's unit-RCS SNR over the disturbance floor that the back end
measured (CFAR), a_n at posterior pose node n with weight w_n, and m_n = 1 + 10^((sigma + a_n)/10):

    P_d(sigma) = sum_n w_n exp(-eta / m_n)                      (predicted crossing probability)
    f(s | s > eta, sigma) = sum_n w_n exp(-s / m_n) / m_n / P_d(sigma)

so a crossing contributes sum_n w_n exp(-s/m_n)/m_n and a miss contributes 1 - P_d. Conditioning
on the measured floor matters: the front end's decision-time curve (``BeamDecision.pd_curve``)
averages the UE leakage over the UE's unknown position in its grid cell. When the actual UE
leaks more or less than that average, the curve gives a wrong event probability and biases the
RCS by up to 2 dB. The curve is used only when a look carries no unit-SNR nodes. A single point
pose can sit in a pattern null and make one look dominate, which is why the pose is integrated.

The strongest crossing cell, which the tracker uses, is not used here: choosing the maximum biases
its power upward. Likewise, the union of cell crossings is not E[max_c P_c]. Plain averages of
detected powers are biased high because only threshold crossings are kept; the truncated factor
removes that bias, and ``likelihood="truncated"`` uses it alone. Beam choices depend only on past
observations, so they do not bias the likelihood.
"""

from collections import defaultdict
from dataclasses import dataclass
import json
import math
import numpy as np
from scipy.special import ndtr, logsumexp
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

    ``detected`` says whether the designated cell crossed the threshold ``eta``. For a crossing,
    ``stat`` is the cell power over the measured disturbance floor. ``unit_db``/``unit_w`` are
    nodes and weights of the cell's unit-RCS SNR over that floor at the posterior pose; they are
    given for misses too. ``pd_sigma``/``pd`` is the decision-time crossing curve, a fallback for
    looks without unit nodes.
    """

    t: float
    detected: bool
    pd_sigma: np.ndarray
    pd: np.ndarray
    stat: float = None
    eta: float = None
    unit_db: np.ndarray = None
    unit_w: np.ndarray = None


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
        event = self.likelihood == "event+truncated"
        if look.unit_db is None:
            if event:
                pd = np.clip(np.interp(s, look.pd_sigma, look.pd), 1e-12, 1 - 1e-12)
                ll += np.log(pd) if look.detected else np.log1p(-pd)
            return ll
        a = np.atleast_1d(np.asarray(look.unit_db, float))
        w = np.ones(len(a)) if look.unit_w is None else np.asarray(look.unit_w, float)
        lw = np.log(w / w.sum())
        m = 1 + 10 ** ((s[..., None] + a) / 10)
        log_pd = logsumexp(lw - look.eta / m, axis=-1)
        if look.detected:
            ll += logsumexp(lw - look.stat / m - np.log(m), axis=-1)
            if not event:
                ll -= log_pd
        elif event:
            ll += np.log(-np.expm1(np.minimum(log_pd, -1e-15)))
        return ll

    def update(self, look):
        """Fold in one RcsLook (``None``: no RCS evidence from this look)."""
        if look is None:
            return
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
