"""Closed-loop demonstration: tracker -> beam front end -> sensing back end -> tracker.

A vehicle drives along the evaluation lane toward the corner RSU. Every ``period`` epochs the
scheduler reports its grants (measured SINR, MCS, PRBs, timing-advance range of each UE). The front
end plans with the current track and RCS design value. The simulated back end executes the beam
on the true multi-scatterer target and returns a detection report. The tracker and the RCS
estimator are then updated. A miss enlarges the next prediction's uncertainty.

The back end draws one complex Gaussian amplitude per scatterer and forms every scatterer cell
coherently through the signed receive-array and delay responses. Its noise-plus-UE-leakage is
Gaussian with the model's power. The strongest threshold crossing gives the measured cell, at CRB
accuracy for its measured SNR and the grant's bandwidth.

    python -m isac.closed_loop --epochs 80 --period 4 --target-model extended
"""

from pathlib import Path
import argparse
import json
import math
import numpy as np
from . import model as V, simulation_engine as LS
from .detection import delay_amplitude
from .frontend import BeamFrontEnd
from .interfaces import GrantReport, SensingReport
from .target import box_scatterers, array_gain, FP
from .tracker import CVTracker, RcsEstimator, crb_polar, glint_polar

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/closed_loop"
EXTENT = {"car": (4.7295, 1.884), "bus": (11.995, 2.642), "truck": (8.0, 2.5)}


def configure():
    """The paper's broad_U1 settings; the tracker replaces ``gap_k`` and the fixed SNR."""
    V.config()
    V.P.update(deploy="corridor", pen_sd=5.0, isd=200.0, load=0.5, sector_centres=9,
               ue_model="U1", sector_rule="union", eps_ue=0.01, psi_deg=None)


class Scheduler:
    """Uniform scheduler over the served lane points; reports measured link state per grant."""

    def __init__(self, rsu, bore, rng, n_grants=5, max_mcs_drop=3, srs_std=None):
        self.rsu, self.bore, self.rng = rsu, bore, rng
        self.n, self.dm, self.srs_std = n_grants, max_mcs_drop, srs_std
        pts, u, r, th, kind = LS.lane_points(rsu, bore, 8.0, V.cell_ymin(rsu))
        _, r_rc, th_rc, _ = V.to_polar(pts, rsu, bore, V.P["rsu_h"], V.P["rc_h"])
        self.pts, self.u, self.r, self.th, self.kind = pts, u, r, th, kind
        self.r_rc, self.th_rc = r_rc, th_rc

    def grants(self, t):
        sinr, k0 = V.ue_link(self.u, self.r, self.th, self.kind, self.pts, self.rsu, self.rng, self.bore)
        mcs = [V.mcs_of(s) for s in sinr]
        elig = np.flatnonzero([m is not None and m[0] >= V.P["mcs_floor"] for m in mcs])
        pick = self.rng.choice(elig, size=min(self.n, len(elig)), replace=False)
        out = []
        for j in pick:
            srs = None
            if self.srs_std:
                srs = float(self.u[j] + self.srs_std * self.rng.normal())
            out.append(
                GrantReport(
                    ue_id=int(j), t=t, k0=int(k0[j]), n_prb=int(self.rng.choice(V.P["prbs"])),
                    n_sym=V.P["symbols"], sinr_db=float(sinr[j]), mcs=mcs[j][0],
                    ue_range_m=float(self.r_rc[j]), ue_theta_deg=float(self.th_rc[j]),
                    ue_kind=str(self.kind[j]), ue_u=srs, ue_u_std=self.srs_std,
                    max_mcs_drop=self.dm,
                )
            )
        return out


def rx_amp(d):
    d = np.asarray(d, float)
    m = V.P["M_rx"]
    safe = np.where(np.abs(d) < 1e-12, 1e-12, d)
    return np.where(np.abs(d) < 1e-12, 1.0, np.sin(m * np.pi * safe / 2) / (m * np.sin(np.pi * safe / 2)))


class SensingBackend:
    """Executes a decision on the true target."""

    def __init__(self, rsu, bore, rng, ue_vehicle_rcs):
        self.rsu, self.bore, self.rng, self.ue_rcs = rsu, bore, rng, ue_vehicle_rcs

    def execute(self, d, truth, sched=None, grant=None):
        keys, xy, sig, dom, _ = box_scatterers(truth["xy"], truth["heading"], *truth["extent"],
                                               self.rsu, truth["rcs"])
        u, r, th, _ = V.to_polar(xy, self.rsu, self.bore, V.P["rsu_h"], V.P["rc_h"])
        W = d.weights[None]
        rho = 10 ** (V.rc_snr_db(array_gain(W, u)[0], u, r, sig, th, d.n_re) / 10)
        amp = np.sqrt(rho / 2) * (self.rng.normal(size=len(u)) + 1j * self.rng.normal(size=len(u)))
        coup = rx_amp(u[:, None] - u[None, :]) * delay_amplitude(r[:, None] - r[None, :], d.bandwidth)
        leak = np.zeros(len(u))
        if grant is not None and V.P["ue_clutter"]:
            j = grant.ue_id
            cs = self.ue_rcs if grant.ue_kind == "vehicle" else V.P["sigma_ped_dbsm"]
            cu, cr, ct = sched.u[j], sched.r_rc[j], sched.th_rc[j]
            cp = 10 ** (V.rc_snr_db(array_gain(W, [cu])[0, 0], cu, cr, cs, ct, d.n_re) / 10)
            leak = cp * V.rx_rel(u - cu) * np.clip(delay_amplitude(r - cr, d.bandwidth) ** 2, 0, 1)
        noise = np.sqrt((1 + leak) / 2) * (self.rng.normal(size=len(u)) + 1j * self.rng.normal(size=len(u)))
        stat = np.abs(coup @ amp + noise) ** 2 / (1 + leak)
        hit = stat > -math.log(V.P["pfa"])
        if not hit.any():
            return SensingReport(d.t, False)
        c = int(np.argmax(np.where(hit, stat, -np.inf)))
        snr_db = 10 * math.log10(max(stat[c] - 1, 1e-3))
        R = crb_polar(snr_db, d.bandwidth) + glint_polar(xy[c], self.rsu, self.bore)
        z = np.array([u[c], r[c]]) + np.linalg.cholesky(R) @ self.rng.normal(size=2)
        dh = V.P["rsu_h"] - V.P["rc_h"]
        z_th = 90 + math.degrees(math.atan2(dh, math.sqrt(max(z[1] ** 2 - dh**2, 0.0))))
        unit = V.rc_snr_db(array_gain(W, [z[0]])[0, 0], z[0], z[1], 0.0, z_th, d.n_re)
        return SensingReport(d.t, True, z, R, snr_db, float(unit))


def truth_trajectory(n, T, rng, start=(5.25, -95.0), speed=8.0, sigma_a=0.5):
    xy, v = np.array(start, float), np.array([0.0, speed])
    out = []
    for k in range(n):
        out.append(dict(t=k * T, xy=xy.copy(), v=v.copy(), heading=math.atan2(v[1], v[0])))
        a = np.array([0.05 * rng.normal(), sigma_a * rng.normal()])
        xy, v = xy + v * T + 0.5 * a * T * T, v + a * T
    return out


def run(epochs=80, period=4, seed=1, target_model="extended", cls="car", n_grants=5,
        max_mcs_drop=3, srs_std=None, verbose=True):
    configure()
    rng = np.random.default_rng(seed)
    meta, rsu, bore = LS.geometry()
    recs = json.loads((ROOT / "data/reference/main_calibration.json").read_text())["records"]
    prior = float(np.percentile([x["sigma"] for x in recs if x["site"] != V.EVAL_SITE],
                                FP["rcs_prior_quantile"]))
    truth_rcs = float(np.median([x["sigma"] for x in recs if x["site"] == V.EVAL_SITE and x["cls"] == cls]))
    ue_rcs = V.rcs_class_median(recs, "car")
    T = V.P["T_epoch"]
    traj = truth_trajectory(epochs, T, rng)
    t0 = traj[0]
    trk = CVTracker(0, 0.0, t0["xy"] + rng.normal(size=2), t0["v"] + rng.normal(size=2), rsu, bore,
                    EXTENT[cls], cls)
    rcs_est = RcsEstimator(prior)
    fe = BeamFrontEnd(rsu, bore, target_model, ue_vehicle_rcs_dbsm=ue_rcs)
    sched = Scheduler(rsu, bore, rng, n_grants, max_mcs_drop, srs_std)
    back = SensingBackend(rsu, bore, rng, ue_rcs)
    log = []
    for k in range(period, epochs, period):
        tr = traj[k]
        if tr["xy"][1] > rsu[1] - 7.0:
            break
        truth = dict(xy=tr["xy"], heading=tr["heading"], extent=EXTENT[cls], rcs=truth_rcs)
        grants = sched.grants(tr["t"])
        rcs = rcs_est.report()
        d = fe.plan(trk.report(), grants, rcs, t=tr["t"])
        g = grants[d.grant] if d.mode == "reuse" else None
        rep = back.execute(d, truth, sched, g)
        trk.update(rep)
        rcs_est.update(rep)
        err = float(np.hypot(*(trk.x[:2] - tr["xy"])))
        row = dict(t=tr["t"], mode=d.mode, tag=d.tag, pd_pred=d.pd_pred, detected=rep.detected,
                   sigma_u=d.sigma_u, s_r=d.s_r, cost_bits=d.cost_bits, dse=d.dse,
                   ue_loss_db=d.ue_loss_db, rcs_design=rcs.design_dbsm, rcs_source=rcs.source,
                   snr_db=rep.snr_db, pos_err_m=err, n_candidates=d.n_candidates)
        log.append(row)
        if verbose:
            print(f"t={tr['t']:5.1f}s {d.mode:9s} tag={d.tag:4d} pd={d.pd_pred:.3f} "
                  f"det={int(rep.detected)} s_u={d.sigma_u:.4f} rcs={rcs.design_dbsm:6.2f}({rcs.source}) "
                  f"err={err:5.2f}m", flush=True)
    summary = summarize(log)
    return dict(config=dict(epochs=epochs, period=period, seed=seed, target_model=target_model,
                            cls=cls, n_grants=n_grants, max_mcs_drop=max_mcs_drop,
                            srs_std=srs_std, prior_dbsm=prior, truth_rcs_dbsm=truth_rcs),
                summary=summary, log=log)


def summarize(log):
    if not log:
        return {}
    det = np.array([r["detected"] for r in log], float)
    pd = np.array([r["pd_pred"] for r in log])
    reuse = np.array([r["mode"] == "reuse" for r in log])
    return dict(
        opportunities=len(log),
        reuse_fraction=float(reuse.mean()),
        detection_rate=float(det.mean()),
        mean_predicted_pd=float(pd.mean()),
        reuse_detection_rate=float(det[reuse].mean()) if reuse.any() else None,
        reuse_mean_predicted_pd=float(pd[reuse].mean()) if reuse.any() else None,
        mean_cost_bits=float(np.mean([r["cost_bits"] for r in log])),
        rms_pos_err_m=float(np.sqrt(np.mean([r["pos_err_m"] ** 2 for r in log]))),
        final_rcs_source=log[-1]["rcs_source"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--target-model", choices=["extended", "point"], default="extended")
    p.add_argument("--cls", choices=sorted(EXTENT), default="car")
    p.add_argument("--grants", type=int, default=5)
    p.add_argument("--max-mcs-drop", type=int, default=3)
    p.add_argument("--srs-std", type=float, default=None,
                   help="UE direction std from SRS (u units); default: grid-cell knowledge (U1)")
    a = p.parse_args()
    res = run(a.epochs, a.period, a.seed, a.target_model, a.cls, a.grants, a.max_mcs_drop, a.srs_std)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{a.target_model}_{a.cls}_seed{a.seed}.json"
    dest.write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(json.dumps(res["summary"], indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
