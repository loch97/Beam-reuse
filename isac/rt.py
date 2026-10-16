"""Trace or score the frozen 48-trajectory RT experiment."""

from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import argparse
import concurrent.futures as cf
import multiprocessing as mp
import hashlib
import time

OUT = ROOT / "outputs/rt"
BASE = ROOT / "data/reference"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def worker_init(variant):
    import mitsuba as mi

    mi.set_variant(variant)


def trace_case(task):
    case, samples, seed, variant = task
    from . import rt_scene as T
    from sionna.rt import PathSolver, Receiver
    from . import rt_scoring as S

    S.config()
    protocol = json.loads((ROOT / "data/rt/protocol.json").read_text())
    trajectory = protocol["trajectories"][case["trajectory"]]
    source = json.loads((ROOT / "data/propagation.json").read_text())
    source["raytracing"]["rt_seed"] = seed
    stem = f"{case['key']}_n{samples}_s{seed}"
    folder = OUT / "paths"
    folder.mkdir(exist_ok=True)
    path = folder / (stem + ".npz")
    info = folder / (stem + ".json")
    start = time.time()
    cue = json.loads(
        (ROOT / "data/rt/reference" / f"{case['key']}_n4000000_s32.json").read_text()
    )
    scene, initial, centre, cfg = T.setup(case, trajectory["materials"][1], source)
    r = case["row"]
    T.pose_vehicle_only(scene, initial, r["target_xy_m"], r["heading_rad"], centre)
    served, antenna, dims = T.add_entity(scene, case, trajectory["materials"][1])
    data = T.solve_one(scene, r, initial, centre, cfg, source, samples)
    ids = [int(scene.objects[n].object_id) for n in served]
    data["served"] = np.isin(data["objects"], ids).any(axis=0)
    total_paths = len(data["a"])
    environment_paths = int((~data["target"] & ~data["served"]).sum())
    keep = data["target"] | data["served"]
    data = {k: v[:, keep] if k == "objects" else v[keep] for k, v in data.items()}
    scene.remove("rx")
    scene.add(Receiver("ue", position=antenna.tolist()))
    paths = PathSolver()(
        scene,
        max_depth=3,
        los=True,
        specular_reflection=True,
        diffuse_reflection=True,
        refraction=False,
        diffraction=False,
        synthetic_array=True,
        samples_per_src=samples,
        seed=seed,
    )
    a = (np.asarray(paths.a[0]) + 1j * np.asarray(paths.a[1])).reshape(-1)
    tau = np.asarray(paths.tau).reshape(-1)
    valid = np.asarray(paths.valid).reshape(-1).astype(bool) & (abs(a) > 0) & np.isfinite(tau)
    for name, value in [
        ("a", a),
        ("tau", tau),
        ("phi_t", np.asarray(paths.phi_t).reshape(-1)),
        ("theta_t", np.asarray(paths.theta_t).reshape(-1)),
    ]:
        data["comm_" + name] = value[valid]
    data["comm_antenna_xyz"] = antenna
    np.savez_compressed(path, **data)
    metadata = dict(
        key=case["key"],
        trajectory=case["trajectory"],
        samples=samples,
        seed=seed,
        variant=variant,
        target_paths=int(data["target"].sum()),
        served_only_paths=int((data["served"] & ~data["target"]).sum()),
        joint_paths=int((data["target"] & data["served"]).sum()),
        comm_paths=int(valid.sum()),
        total_paths=total_paths,
        unscored_environment_paths=environment_paths,
        antenna_xyz=antenna.tolist(),
        served_dims=dims,
        cue=cue,
        elapsed_s=time.time() - start,
        protocol_sha256=sha(ROOT / "data/rt/protocol.json"),
        path_sha256=sha(path),
        code_sha256=sha(__file__),
    )
    write(info, metadata)
    print(
        json.dumps(
            {
                k: metadata[k]
                for k in ("key", "target_paths", "served_only_paths", "comm_paths", "elapsed_s")
            }
        ),
        flush=True,
    )
    return metadata


def score_case(task):
    case, samples, seed = task
    from threadpoolctl import threadpool_limits
    from . import rt_scoring as S
    from .detection import predictive_pd, delay_power
    from .calibration import array_power
    from .rt_scoring import communication_power, echo_power
    from .quadrature import cached_gain

    S.config()
    V = S.V
    V.gain = cached_gain
    stem = f"{case['key']}_n{samples}_s{seed}"
    dest = OUT / "scores" / (stem + ".json")
    path = OUT / "paths" / (stem + ".npz")
    meta = json.loads(path.with_suffix(".json").read_text())
    protocol = json.loads((ROOT / "data/rt/protocol.json").read_text())
    traj = protocol["trajectories"][case["trajectory"]]
    with np.load(path) as z:
        data = dict(z)
    control_data = data
    if samples != 4000000 or seed != 32:
        with np.load(OUT / "paths" / f"{case['key']}_n4000000_s32.npz") as z:
            control_data = dict(z)
    rsu = np.array(case["row"]["rsu_xy_m"])
    bore = traj["boresight_rad"]
    cue = meta["cue"]
    prior = json.loads((BASE / "rt_calibration.json").read_text())["prior_dbsm"]
    car_rcs = V.rcs_class_median(
        json.loads((BASE / "main_calibration.json").read_text())["records"], "car"
    )
    sig = V.P["sigma_ped_dbsm"] if case["entity"] == "pedestrian" else car_rcs
    cu, cr, ct, _ = V.to_polar(np.array([case["ue_xy"]]), rsu, bore, 7.5, 1.5)
    cu, cr, ct = map(float, (cu[0], cr[0], ct[0]))
    _, _, theta, _ = V.to_polar(np.array([case["row"]["target_xy_m"]]), rsu, bore, 7.5, 1.5)
    theta = float(theta[0])
    su, sr, cov = V.track_covariance(
        np.array(case["row"]["target_xy_m"]), rsu, bore, case["row"]["heading_rad"]
    )
    chol = np.linalg.cholesky([[su**2, cov], [cov, sr**2]])
    grid = np.array([V.beam(g) for g in V.GRID])
    levels = V.levels()[0]
    records = []
    states = []
    beamfreeze = []
    with threadpool_limits(limits=1):
        for report in case["reports"]:
            bg = report["prb"] * 12 * V.P["df"]
            nre = report["prb"] * 12 * V.P["symbols"]
            gridpower = communication_power(control_data, grid, bg, bore)
            k0 = int(np.argmax(gridpower))
            g0 = float(V.GRID[k0])
            ib = float(
                V.corridor_interference_dbm(
                    np.array([case["ue_xy"]]),
                    rsu,
                    bore,
                    np.random.default_rng(report["interference_seed"]),
                )[0]
            )
            noiseI = float(V.db(10 ** (V.NOISE_UE / 10) + 10 ** (ib / 10)))
            gamma0 = float(
                V.P["p_ref_dbm"]
                + 10 * np.log10(V.M)
                + V.P["g_ue"]
                + V.db(gridpower[k0])
                - noiseI
            )
            mcs = V.mcs_of(gamma0)
            eligible = mcs is not None and mcs[0] >= V.P["mcs_floor"]
            state = dict(
                case=case["key"],
                trajectory=case["trajectory"],
                report=report["id"],
                prb=report["prb"],
                gamma0=gamma0,
                eligible=eligible,
                serving_grid=g0,
                reference_available=cue["available"],
            )
            states.append(state)
            if not eligible:
                continue
            uh, rh = np.array([cue["u"], cue["r"]]) + chol @ report["z"]
            budget = gamma0 - mcs[1]
            W, tags = V.candidates(
                cu, float(uh), k0, 6, sigma_u=su, ue_ctrl=(g0, g0 - 1 / 64, g0 + 1 / 64)
            )
            tq = 2 * (np.arange(V.P["n_ue_q"]) + 0.5) / V.P["n_ue_q"] - 1
            loss = np.quantile(
                V.db(V.gain(V.beam(g0), g0 + tq / 64))[0][None, :]
                - V.db(V.gain(W, g0 + tq / 64)),
                0.99,
                axis=1,
                method="inverted_cdf",
            )
            ev = np.flatnonzero(loss <= budget + 1e-09)
            prediction = np.full(len(W), -1.0)
            x, cw = np.polynomial.legendre.leggauss(V.P["n_ue_clut"])
            if len(ev):
                prediction[ev] = predictive_pd(
                    W[ev],
                    float(uh),
                    float(rh),
                    su,
                    prior,
                    theta,
                    nre,
                    bg,
                    (g0 + x / 64, cr, sig, ct, cw / 2),
                    s_r=sr,
                    c_ur=cov,
                )[0]
            selected = []
            for family, lv in S.FAMILIES.items():
                adm = ev[np.isin(tags[ev], levels[lv])]
                state[family + "_compatible"] = bool(len(adm) and prediction[adm].max() >= 0.9)
                if not len(adm):
                    continue
                for side in (-1, 1):
                    pool = adm[(prediction[adm] - 0.9) * side >= 0]
                    if len(pool):
                        selected.append(
                            (
                                "boundary",
                                family,
                                side,
                                int(pool[np.argmin(abs(prediction[pool] - 0.9))]),
                            )
                        )
                selected.append(("action", family, 0, int(adm[np.argmax(prediction[adm])])))
            if not selected:
                continue
            indices = sorted({v[3] for v in selected})
            pos = {v: k for k, v in enumerate(indices)}
            ws = W[indices]
            beamfreeze.append(
                dict(
                    report=report["id"],
                    uhat=float(uh),
                    rhat=float(rh),
                    sigma_u=su,
                    sigma_r=sr,
                    cov_ur=cov,
                    indices=indices,
                    selection=selected,
                    beam_real=ws.real.tolist(),
                    beam_imag=ws.imag.tolist(),
                    prediction=prediction[indices].tolist(),
                )
            )
            signal, served = echo_power(data, ws, nre, bg, {"u": cue["u"], "r": cue["r"]})
            proxy = (
                10 ** (V.rc_snr_db(V.gain(ws, cu), cu, cr, sig, ct, nre) / 10)
                * array_power(cue["u"] - cu, V.P["M_rx"])
                * delay_power(cue["r"] - cr, bg)
            )
            pd = lambda c: np.exp(np.log(V.P["pfa"]) / (1 + signal / (1 + c)))
            pointpd, entitypd = (pd(proxy), pd(served))
            wp = communication_power(data, ws, bg, bore)
            gam = V.P["p_ref_dbm"] + 10 * np.log10(V.M) + V.P["g_ue"] + V.db(wp) - noiseI
            for selector, family, side, idx in selected:
                j = pos[idx]
                records.append(
                    dict(
                        case=case["key"],
                        trajectory=case["trajectory"],
                        frame=case["frame"],
                        report=report["id"],
                        target_class=case["target_class"],
                        entity=case["entity"],
                        selector=selector,
                        family=family,
                        side=side,
                        candidate=idx,
                        prb=report["prb"],
                        prediction=float(prediction[idx]),
                        point_ue_pd=float(pointpd[j]),
                        entity_ue_pd=float(entitypd[j]),
                        target_power=float(signal[j]),
                        point_ue_power=float(proxy[j]),
                        entity_ue_power=float(served[j]),
                        gamma0=gamma0,
                        rt_comm_sinr_db=float(gam[j]),
                        rt_comm_loss_db=float(gamma0 - gam[j]),
                        control_loss_db=float(loss[idx]),
                        ue_budget=float(budget),
                        mcs=mcs[0],
                        comm_pass=bool(gam[j] >= mcs[1]),
                    )
                )
    result = dict(
        case=case,
        metadata=meta,
        states=states,
        records=records,
        beam_selection=beamfreeze,
        scoring_code_sha256=sha(__file__),
    )
    write(dest, result)
    print("SCORED", case["key"], len(records), flush=True)
    return dict(key=case["key"], probes=len(records))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["trace", "score"])
    p.add_argument("--samples", type=int, default=16000000)
    p.add_argument("--seed", type=int, default=32)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--variant",
        choices=["cuda_ad_mono_polarized", "llvm_ad_mono_polarized"],
        default="cuda_ad_mono_polarized",
    )
    p.add_argument("--cases", nargs="+")
    a = p.parse_args()
    protocol = json.loads((ROOT / "data/rt/protocol.json").read_text())
    cases = protocol["cases"]
    if a.cases is not None:
        cases = [c for c in cases if c["key"] in a.cases]
    (OUT / "paths").mkdir(parents=True, exist_ok=True)
    if a.command == "trace":
        tasks = [(c, a.samples, a.seed, a.variant) for c in cases]
        with cf.ProcessPoolExecutor(
            max_workers=a.workers,
            mp_context=mp.get_context("spawn"),
            initializer=worker_init,
            initargs=(a.variant,),
        ) as pool:
            list(pool.map(trace_case, tasks))
    else:
        with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(score_case, [(c, a.samples, a.seed) for c in cases]))


if __name__ == "__main__":
    main()
