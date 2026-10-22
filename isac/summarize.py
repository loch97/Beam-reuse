"""Complete-set paired block inference for revised resource and stress figures."""

import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .simulate import OUT
from .simulate import CONDITIONS
from .scheduling import POLICIES

METRICS = ["need", "dedicated", "unserved", "adaptation", "displacement", "cost"]
STRESS = ["s1", "s2", "b+2", "b-2"]


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-model", choices=["point", "extended"], default="point")
    tm = p.parse_args().target_model
    suffix = "" if tm == "point" else "_extended"
    seeds = list(range(2026100600, 2026100664))
    folder = OUT / ("simulation" + suffix)
    paths = {(c, s): folder / f"{c}_seed{s}_f12.json" for c in CONDITIONS for s in seeds}
    missing = [str(p.name) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(
            f"Incomplete: {len(missing)} of 384 blocks; no final inference written."
        )
    records = {key: json.loads(p.read_text()) for key, p in paths.items()}
    assert all((r["frames"] == 12 and r["draws"] == 1 for r in records.values()))
    idx = np.random.default_rng(20261006).integers(0, 64, (20000, 64))
    counts = np.stack([(idx == j).sum(axis=1) / 64 for j in range(64)], axis=1)
    resource = dict(
        config=dict(
            seed_count=64,
            frames_per_seed=12,
            draws_per_frame=1,
            configurations_per_condition=768,
            total_condition_configurations=4608,
            dm=[0, 3, 16],
            N=list(range(1, 11)),
            bootstrap_resamples=20000,
            scope="Paired seed-block inference conditional on the fixed two trajectories and corrected model",
        ),
        conditions={},
        paired={},
    )
    for s in seeds:
        reference = None
        for c in CONDITIONS:
            with np.load(folder / f"{c}_seed{s}_f12.npz") as z:
                a = z["C"]
                assert a.shape == (12, 1, 4, 3, 115)
                assert not (a[:, :, 1] & ~a[:, :, 2]).any() and (
                    not (a[:, :, 1] & ~a[:, :, 3]).any()
                )
                assert np.all(np.diff(a.astype(int), axis=3) >= 0)
                if reference is None:
                    reference = (z["N_RE"].copy(), z["SE0"].copy())
                else:
                    assert np.array_equal(reference[0], z["N_RE"]) and np.array_equal(
                        reference[1], z["SE0"]
                    )
    resource["per_state_audit"] = dict(
        blocks=384,
        paired_prbs_and_channels=True,
        nested_steering_candidates=True,
        nested_mcs_allowance=True,
    )
    for c in CONDITIONS:
        rows = [records[c, s] for s in seeds]
        data = np.array(
            [[[r["metrics"][p][m] for m in METRICS] for p in POLICIES] for r in rows]
        )
        assert data.shape == (64, 5, 6, 3, 10) and np.isfinite(data).all()
        assert np.max(abs(data[:, :, 0] - data[:, :, 1] - data[:, :, 2])) < 1e-12
        assert np.max(abs(data[:, :, 5] - data[:, :, 3] - data[:, :, 4])) < 1e-10
        assert np.all(np.diff(data[:, :, 1], axis=-1) <= 1e-12)
        assert np.all(data[:, 1, 1] <= data[:, 0, 1] + 1e-12)
        assert np.all(data[:, 2, 1] <= data[:, 1, 1] + 1e-12)
        assert np.all(data[:, 3, 1] <= data[:, 1, 1] + 1e-12)
        with threadpool_limits(1):
            boot = (counts @ data.reshape(64, -1)).reshape(20000, 5, 6, 3, 10)
        mean = data.mean(axis=0)
        ci = np.quantile(boot, [0.025, 0.975], axis=0)
        resource["conditions"][c] = {
            p: {
                m: dict(mean=mean[j, k].tolist(), ci=np.moveaxis(ci[:, j, k], 0, -1).tolist())
                for k, m in enumerate(METRICS)
            }
            for j, p in enumerate(POLICIES)
        }
        resource["paired"][c] = {}
        for name, a, b in [
            ("broadening_minus_steering", 2, 1),
            ("multibeam_minus_broadening", 3, 2),
            ("sequential_minus_multibeam", 4, 3),
        ]:
            resource["paired"][c][name] = {}
            for metric in ["dedicated", "cost"]:
                k = METRICS.index(metric)
                d = boot[:, a, k, :, 4] - boot[:, b, k, :, 4]
                resource["paired"][c][name][metric] = dict(
                    mean=(mean[a, k, :, 4] - mean[b, k, :, 4]).tolist(),
                    ci=np.quantile(d, [0.025, 0.975], axis=0).T.tolist(),
                )
    (OUT / f"resource_summary{suffix}.json").write_text(
        json.dumps(resource, indent=2) + "\n", encoding="utf-8"
    )
    robust = dict(
        config=dict(
            seeds=64,
            frames_per_seed=12,
            draws_per_frame=1,
            frame_draw_configurations=768,
            bootstrap_resamples=20000,
            independent_unit="seed block, including shared PRB allocation",
            scope="Frozen corrected decisions; conditional point-target Gaussian disturbance model",
        ),
        conditions=STRESS,
        scores={},
        paired_broadening_minus_multibeam={},
    )
    rows = [records["broad_U1", s] for s in seeds]
    boots = {}
    for label, key in [
        ("Beam steering", "L2_dm0"),
        ("Broadened beam", "L11_dm0"),
        ("Multibeam", "L13_dm0"),
        ("Sequential", None),
    ]:
        pds = []
        nums = []
        dens = []
        deds = []
        for row in rows:
            if key:
                r = row["stress"][key]
                w = np.array([np.mean(r[c]["reuse_weight_frames"]) for c in STRESS])
                n = np.array(
                    [
                        r[c]["P_out"] * weight if weight > 0 else 0.0
                        for c, weight in zip(STRESS, w)
                    ]
                )
                p = np.array([r[c]["P_det"] for c in STRESS])
                ded = r["o_ded"]
            else:
                r = row["cascade_S10_T"]["dm0"]
                w = np.full(4, r["w1"] + r["w2"])
                n = np.array(
                    [
                        (r["w1"] * r[c]["P_out_tier1"] if r["w1"] > 0 else 0.0)
                        + (r["w2"] * r[c]["P_out_tier2"] if r["w2"] > 0 else 0.0)
                        for c in STRESS
                    ]
                )
                p = np.array([r[c]["P_det"] for c in STRESS])
                ded = r["o_std"]
            pds.append(p)
            nums.append(n)
            dens.append(w)
            deds.append(ded)
        pds, nums, dens, deds = map(np.asarray, [pds, nums, dens, deds])
        bp = pds[idx].mean(axis=1)
        bo = nums[idx].sum(axis=1) / dens[idx].sum(axis=1)
        boots[label] = dict(system_miss=1 - bp, P_out_opp=bo)
        robust["scores"][label] = dict(
            P_det=pds.mean(axis=0).tolist(),
            P_out_opp=(nums.sum(axis=0) / dens.sum(axis=0)).tolist(),
            system_miss_ci=np.quantile(1 - bp, [0.025, 0.975], axis=0).T.tolist(),
            P_out_opp_ci=np.quantile(bo, [0.025, 0.975], axis=0).T.tolist(),
            o_ded=float(deds.mean()),
            o_ded_ci=np.quantile(deds[idx].mean(axis=1), [0.025, 0.975]).tolist(),
        )
        assert (
            abs(
                robust["scores"][label]["o_ded"]
                - resource["conditions"]["broad_U1"][label]["dedicated"]["mean"][0][4]
            )
            < 1e-12
        )
    for metric in ["system_miss", "P_out_opp"]:
        delta = boots["Broadened beam"][metric] - boots["Multibeam"][metric]
        a, b = (robust["scores"]["Broadened beam"], robust["scores"]["Multibeam"])
        estimate = (
            np.array(b["P_det"]) - np.array(a["P_det"])
            if metric == "system_miss"
            else np.array(a["P_out_opp"]) - np.array(b["P_out_opp"])
        )
        robust["paired_broadening_minus_multibeam"][metric] = dict(
            estimate=estimate.tolist(), ci=np.quantile(delta, [0.025, 0.975], axis=0).T.tolist()
        )
    (OUT / f"robustness_summary{suffix}.json").write_text(
        json.dumps(robust, indent=2) + "\n", encoding="utf-8"
    )
    print("Complete full-set inference: 384 blocks and 64 frozen-robustness blocks.")
    for label in POLICIES:
        r = resource["conditions"]["broad_U1"][label]
        print(
            label,
            "dedicated_pct",
            100 * r["dedicated"]["mean"][0][4],
            "cost",
            r["cost"]["mean"][0][4],
        )
    print("Stress", json.dumps(robust["scores"], indent=2))


if __name__ == "__main__":
    main()
