"""Paired seed-block comparison of the extended and point target models.

Both models are run on the same seed blocks (``simulate --target-model point|extended``), so the
random reports, grants, penetration losses and interference are shared. For each condition,
policy, MCS-drop budget and number of scheduled UEs, the extended-minus-point difference of each
resource metric is averaged over the blocks. A 95% interval comes from 20,000 paired resamples of
whole seed blocks, the same resampling as ``summarize.py``. Robustness scores (broad_U1) are
compared in the same way.

Blocks missing for either model are left out and the report says how many were used; the final
figures need all 64.

    python -m isac.compare_models [--dm 3] [--n 5]
"""

import argparse
import json
from pathlib import Path
import numpy as np
from .simulate import OUT, CONDITIONS
from .scheduling import POLICIES

METRICS = ["need", "unserved", "cost"]
SEEDS = list(range(2026100600, 2026100664))
FOLDERS = dict(point=OUT / "simulation", extended=OUT / "simulation_extended")


def load(model, condition, seed):
    p = FOLDERS[model] / f"{condition}_seed{seed}_f12.json"
    return json.loads(p.read_text()) if p.exists() else None


def paired(diff, idx):
    """Mean and 95% interval of the block mean under paired block resampling."""
    boot = diff[idx].mean(axis=1)
    return dict(mean=float(diff.mean()), lo=float(np.percentile(boot, 2.5)),
                hi=float(np.percentile(boot, 97.5)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dm", type=int, default=3, help="MCS-drop budget (0, 3 or 16)")
    p.add_argument("--n", type=int, default=5, help="scheduled UEs N (1..10)")
    a = p.parse_args()
    q, n = [0, 3, 16].index(a.dm), a.n - 1
    out = dict(dm=a.dm, N=a.n, conditions={})
    for c in CONDITIONS:
        pairs = [(load("point", c, s), load("extended", c, s)) for s in SEEDS]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if not pairs:
            continue
        idx = np.random.default_rng(20261006).integers(0, len(pairs), (20000, len(pairs)))
        res = dict(blocks=len(pairs), policies={})
        for pol in POLICIES:
            r = {}
            for m in METRICS:
                x = np.array([pt["metrics"][pol][m][q][n] for pt, _ in pairs])
                y = np.array([ex["metrics"][pol][m][q][n] for _, ex in pairs])
                r[m] = dict(point=float(x.mean()), extended=float(y.mean()), diff=paired(y - x, idx))
            res["policies"][pol] = r
        both = [(x, y) for x, y in pairs if "stress" in x and "stress" in y]
        if both:
            idx2 = np.random.default_rng(20261006).integers(0, len(both), (20000, len(both)))
            rob = {}
            for key in both[0][0]["stress"]:
                if not key.endswith(f"dm{a.dm}"):
                    continue
                rob[key] = {}
                for k in ("s1", "s2", "b+2", "b-2"):
                    x = np.array([pt["stress"][key][k]["P_det"] for pt, _ in both])
                    y = np.array([ex["stress"][key][k]["P_det"] for _, ex in both])
                    rob[key][k] = dict(point=float(x.mean()), extended=float(y.mean()),
                                       diff=paired(y - x, idx2))
            res["robustness_P_det"] = dict(blocks=len(both), scores=rob)
        out["conditions"][c] = res
    dest = OUT / "model_comparison.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    for c, res in out["conditions"].items():
        print(f"== {c}: {res['blocks']} paired blocks (dm={a.dm}, N={a.n}); extended - point [95% CI]")
        for pol, r in res["policies"].items():
            cells = [f"{m} {r[m]['point']:8.3f}->{r[m]['extended']:8.3f} "
                     f"[{r[m]['diff']['lo']:+.3f}, {r[m]['diff']['hi']:+.3f}]" for m in METRICS]
            print(f"  {pol:15s} " + " | ".join(cells))
        if "robustness_P_det" in res:
            rb = res["robustness_P_det"]
            print(f"  robustness P_det ({rb['blocks']} blocks):")
            for key, d in rb["scores"].items():
                print("    " + key + " " + " ".join(
                    f"{k} {v['point']:.3f}->{v['extended']:.3f}" for k, v in d.items()))
    if len(out["conditions"]) < len(CONDITIONS) or any(
            r["blocks"] < 64 for r in out["conditions"].values()):
        print("PARTIAL: not all 6 x 64 paired blocks are available")
    print("wrote", dest)


if __name__ == "__main__":
    main()
