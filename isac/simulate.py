"""Run the six paper conditions directly, without retries or lock files."""

from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import argparse
import concurrent.futures as cf
import time
from threadpoolctl import threadpool_limits
from . import model as V, simulation_engine as LS, scheduling as VR
from .quadrature import cached_gain, PointTruth, UELossCache
from .detection import predictive_pd, compatibility_screen
from .scheduling import expectations, LEVELS

OUT = ROOT / "outputs"
STRESS = ["s1", "s2", "b+2", "b-2"]
CONDITIONS = ["broad_U1", "inter_U1", "narrow_U1", "broad_U0", "inter_U0", "narrow_U0"]


def run(condition, seed, frames=None, skip_robustness=False, target_model="point"):
    """One condition and seed block. ``target_model="extended"`` replaces the point detector by
    the multi-scatterer box (``target.ExtendedDetector``) in both the dedicated and the reuse
    screens and writes to ``outputs/simulation_extended``; everything else, including the random
    draws, is identical, so the two models are paired by seed block."""
    t0 = time.time()
    unc, ue = condition.split("_")
    assert unc in ["narrow", "inter", "broad"] and ue in ["U0", "U1"]
    V.config()
    V.gain = cached_gain
    V.predictive_pd = predictive_pd
    screen, hook = compatibility_screen, None
    if target_model == "extended":
        from .target import ExtendedDetector

        plain, screen = ExtendedDetector(), ExtendedDetector(threshold=0.9)
        V.predictive_pd = plain

        def hook(row, rsu, bore):
            plain.set_frame(row, rsu, bore)
            screen.set_frame(row, rsu, bore)
    elif target_model != "point":
        raise ValueError(target_model)
    V.P.update(
        deploy="corridor",
        pen_sd=5.0,
        comm_per_draw=True,
        isd=200.0,
        load=0.5,
        dm_list=(0, 3, 16),
        n_draws=1,
        sector_centres=9,
        unc_model="track",
        gap_k={"narrow": 0, "inter": 9, "broad": 19}[unc],
        psi_deg=None if unc == "narrow" else 90.0,
        prune_admissible=True,
        ue_model=ue,
        sector_rule="union",
        eps_ue=0.01,
    )
    recs = json.loads((ROOT / "data/reference/main_calibration.json").read_text())["records"]
    robust = condition == "broad_U1" and (not skip_robustness)
    with threadpool_limits(limits=1):
        A = LS.compat_arrays(
            10.0,
            1.0,
            8.0,
            recs,
            seed,
            n_frames=frames,
            eval_levels=LEVELS,
            ue_loss_fn=UELossCache(),
            extra=PointTruth() if robust else None,
            predictive_screen_fn=screen,
            frame_hook=hook,
        )
        out = dict(
            condition=condition,
            seed=seed,
            frames=A["C"].shape[0],
            draws=1,
            dm=[0, 3, 16],
            N=list(range(1, 11)),
            metrics=expectations(A),
            kernel_version=(predictive_pd if target_model == "point" else screen).resource_kernel_version,
            target_model=target_model,
            robustness=robust,
        )
        assert not (A["C"][:, :, 2] & ~A["C"][:, :, 11]).any()
        assert not (A["C"][:, :, 2] & ~A["C"][:, :, 13]).any()
        assert np.all(np.diff(A["C"].astype(int), axis=3) >= 0)
        if robust:
            VR.STRESS = STRESS
            out["stress"] = VR.summarize(A, eval_levels=(2, 11, 13))
            out["cascade_S10_T"] = VR.cascade(A, 11, 13)
        dest = OUT / ("simulation" if target_model == "point" else "simulation_extended")
        dest.mkdir(parents=True, exist_ok=True)
        stem = f"{condition}_seed{seed}_f{out['frames']}"
        arrays = {k: A[k][:, :, LEVELS] for k in ["C", "DSE", "PDP", "LOSS", "TSEL"]}
        arrays.update({k: A[k] for k in ["FDED", "SE0", "N_RE"]})
        if robust:
            arrays.update({f"X_{s}": A["X"][s][:, :, LEVELS] for s in STRESS})
            arrays.update({f"XDED_{s}": A["XDED"][s] for s in STRESS})
        np.savez_compressed(dest / (stem + ".npz"), levels=LEVELS, **arrays)
        out["elapsed_s"] = time.time() - t0
        tmp = dest / (stem + ".json")
        tmp.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print("DONE", stem, round(out["elapsed_s"], 1), flush=True)
    return out


def run_task(task):
    return run(*task)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=CONDITIONS, nargs="+", default=CONDITIONS)
    p.add_argument("--seeds", type=int, default=64)
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--target-model", choices=["point", "extended"], default="point")
    p.add_argument("--skip-robustness", action="store_true")
    a = p.parse_args()
    tasks = [(c, 2026100600 + i, a.frames, a.skip_robustness, a.target_model)
             for c in a.condition for i in range(a.seeds)]
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(run_task, tasks):
            print(result["condition"], result["seed"])


if __name__ == "__main__":
    main()
