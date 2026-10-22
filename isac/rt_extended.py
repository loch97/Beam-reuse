"""RT check of the target geometry: point reflector versus multi-scatterer box.

For each traced case it compares the beam dependence of the RT target power at the case's
reference focus with the two models' predictions, over steered, broadened and sector beams around
the focus and several grant bandwidths. Each model's level (dominant RCS) is fitted on the beams
within ``--fit-db`` of the RT peak, so only the shape is compared.

RT paths are split by geometry:
- *direct* paths leave and return in the same direction (|u_t - u_r| < ``--mono-tol``). These test
  the scatterer geometry.
- *transmit multipath* paths reach the target over another reflector and return directly. Neither
  model contains them, so their share per beam is reported separately.

Errors are reported within ``--window-db`` of the RT peak, where the detection decision is made.
Positive under-prediction means RT is stronger than the model (conservative); over-prediction is
the unsafe direction.

    python -m isac.rt_extended                      # bundled example cache
    python -m isac.rt_extended --paths outputs/rt/paths --fit-offsets
"""

from functools import lru_cache
from pathlib import Path
import argparse
import json
import math
import numpy as np
from . import model as V, rt_scoring as S
from .detection import delay_power
from .target import array_gain, box_scatterers

ROOT = Path(__file__).resolve().parents[1]
PRBS = (1, 4, 32)


@lru_cache(maxsize=None)
def vehicle_extent(category):
    """(L, W) of a vehicle from the bounding box of its RT meshes."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for f in sorted((ROOT / "assets" / f"complex_low_poly_{category}" / "meshes").glob("*.ply")):
        raw = f.read_bytes()
        end = raw.index(b"end_header\n") + len(b"end_header\n")
        head = raw[:end].decode()
        n = int(next(l for l in head.splitlines() if l.startswith("element vertex")).split()[-1])
        props = sum(l.startswith("property") for l in head.split("element face")[0].splitlines())
        v = np.frombuffer(raw[end:end + 4 * n * props], dtype="<f4").reshape(n, props)[:, :3]
        lo, hi = np.minimum(lo, v.min(0)), np.maximum(hi, v.max(0))
    return float(hi[0] - lo[0]), float(hi[1] - lo[1])


def beams(u0):
    """Steered, quadratically broadened and accepted phase-only sector beams around u0."""
    gs = np.linspace(u0 - 0.25, u0 + 0.25, 101)
    W = [V.beam(g) for g in gs] + [V.beam(g, V.P["q_wide"]) for g in gs[::4]]
    for phi in V.sector_codebook().values():
        W += [np.exp(1j * phi - 1j * np.pi * V.N_IDX * c) / math.sqrt(V.M)
              for c in np.linspace(u0 - 0.1, u0 + 0.1, 5)]
    return np.array(W)


def focus_point(focus, rsu, bore):
    d2 = math.sqrt(max(focus["r"] ** 2 - (V.P["rsu_h"] - V.P["rc_h"]) ** 2, 0.0))
    az = math.asin(focus["u"]) + bore
    return np.array([rsu + d2 * np.array([math.cos(az), math.sin(az)])])


def model_power(W, xy, sig, focus, rsu, bore, prb):
    """Per-beam model power at the receive focus; scatterer RCS in dBsm."""
    nre, bg = prb * 12 * V.P["symbols"], prb * 12 * V.P["df"]
    u, r, th, _ = V.to_polar(xy, rsu, bore, V.P["rsu_h"], V.P["rc_h"])
    rho = 10 ** (V.rc_snr_db(array_gain(W, u), u, r, sig, th, nre) / 10)
    return rho @ (V.rx_rel(focus["u"] - u) * delay_power(focus["r"] - r, bg))


def rt_power(data, W, focus, prb, mask):
    d = dict(data)
    d["target"] = np.asarray(data["target"], bool) & mask
    d["served"] = np.zeros_like(np.asarray(data["served"], bool))
    return S.echo_power(d, W, prb * 12 * V.P["symbols"], prb * 12 * V.P["df"], focus)[0]


def shape_errors(y, m, fit_db, window_db):
    """Level fitted on the beams within fit_db of the peak; dB errors within window_db."""
    fit = y >= y.max() * 10 ** (-fit_db / 10)
    lvl = float(np.mean(10 * np.log10(y[fit] / m[fit])))
    win = y >= y.max() * 10 ** (-window_db / 10)
    e = 10 * np.log10(y[win] / m[win]) - lvl
    return lvl, e


def load_case(paths, case, samples, seed):
    stem = f"{case['key']}_n{samples}_s{seed}"
    npz, meta = Path(paths) / (stem + ".npz"), Path(paths) / (stem + ".json")
    if not npz.exists():
        return None
    with np.load(npz) as z:
        data = {k: z[k] for k in z.files}
    return data, json.loads(meta.read_text())


def validate_case(case, data, meta, args, offsets=None):
    row = case["row"]
    rsu, bore = np.array(row["rsu_xy_m"], float), math.radians(row["boresight_az_deg"])
    focus = dict(u=meta["cue"]["u"], r=meta["cue"]["r"])
    L, Wd = vehicle_extent(case["target_class"])
    keys, xy, sig, dom, _ = box_scatterers(row["target_xy_m"], row["heading_rad"], L, Wd, rsu, 0.0,
                                           offsets)
    W = beams(focus["u"])
    direct = np.abs(np.asarray(data["ut"]) - np.asarray(data["ur"])) < args.mono_tol
    geoms = dict(point=(focus_point(focus, rsu, bore), np.zeros(1)), box=(xy, sig))
    out = dict(case=case["key"], cls=case["target_class"], cue_sigma=meta["cue"]["sigma"],
               n_beams=len(W), prb={})
    for prb in args.prbs:
        y_d, y_a = rt_power(data, W, focus, prb, direct), rt_power(data, W, focus, prb, ~direct | direct)
        share = 1 - y_d / np.maximum(y_a, 1e-300)
        win = y_a >= y_a.max() * 10 ** (-args.window_db / 10)
        res = dict(multipath_share_max=float(share[win].max()),
                   multipath_share_mean=float(share[win].mean()))
        for name, (pxy, psig) in geoms.items():
            m = model_power(W, pxy, psig, focus, rsu, bore, prb)
            for lab, y in (("direct", y_d), ("all", y_a)):
                lvl, e = shape_errors(y, m, args.fit_db, args.window_db)
                res[f"{name}_{lab}"] = dict(level_dbsm=lvl, rms_db=float(np.sqrt(np.mean(e**2))),
                                            under_max_db=float(e.max()), over_max_db=float(-e.min()),
                                            errors_db=e.tolist())
        out["prb"][str(prb)] = res
    return out


def summarize(results, prbs):
    s = {}
    for prb in map(str, prbs):
        rows = [r["prb"][prb] for r in results]
        d = dict(multipath_share_max=float(np.median([x["multipath_share_max"] for x in rows])))
        for key in ("point_direct", "box_direct", "point_all", "box_all"):
            e = np.concatenate([x[key]["errors_db"] for x in rows])
            d[key] = dict(rms_db=float(np.sqrt(np.mean(e**2))), p95_abs_db=float(np.percentile(abs(e), 95)),
                          over_p95_db=float(np.percentile(-e, 95)), n=int(len(e)))
        s[prb] = d
    return s


def fit_offsets(cases, args):
    """Corner and other-face offsets that minimise the pooled direct-path rms of the box model."""
    from scipy.optimize import minimize

    def loss(x):
        off = {"dominant_face": 0.0, "corner": x[0], "other_face": x[1]}
        e = []
        for case, data, meta in cases:
            r = validate_case(case, data, meta, args, off)
            e += [v for p in r["prb"].values() for v in p["box_direct"]["errors_db"]]
        return float(np.sqrt(np.mean(np.square(e))))

    best = minimize(loss, [-6.0, -10.0], method="Nelder-Mead", options=dict(xatol=0.1, fatol=1e-3))
    return dict(corner=float(best.x[0]), other_face=float(best.x[1]), rms_db=float(best.fun))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paths", default=str(ROOT / "data/example_rt"))
    p.add_argument("--samples", type=int, default=16000000)
    p.add_argument("--seed", type=int, default=32)
    p.add_argument("--prbs", type=int, nargs="+", default=list(PRBS))
    p.add_argument("--mono-tol", type=float, default=0.01)
    p.add_argument("--fit-db", type=float, default=10.0)
    p.add_argument("--window-db", type=float, default=20.0)
    p.add_argument("--fit-offsets", action="store_true")
    p.add_argument("--out", default=str(ROOT / "outputs/rt_extended.json"))
    a = p.parse_args()
    S.config()
    protocol = json.loads((ROOT / "data/rt/protocol.json").read_text())
    cases = []
    for case in protocol["cases"]:
        got = load_case(a.paths, case, a.samples, a.seed)
        if got is not None:
            cases.append((case, *got))
    if not cases:
        raise SystemExit(f"no path caches under {a.paths}")
    results = [validate_case(c, d, m, a) for c, d, m in cases]
    out = dict(cases=len(results), summary=summarize(results, a.prbs), results=results)
    if a.fit_offsets:
        out["fitted_offsets"] = fit_offsets(cases, a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(results)} case(s) from {a.paths}")
    for prb, d in out["summary"].items():
        print(f"PRB {prb:>2}: transmit-multipath share (max over beams, median over cases) "
              f"{d['multipath_share_max']:.2f}")
        for key in ("point_direct", "box_direct", "point_all", "box_all"):
            x = d[key]
            print(f"   {key:13s} rms {x['rms_db']:5.2f} dB  |e| p95 {x['p95_abs_db']:5.2f} dB  "
                  f"over-prediction p95 {x['over_p95_db']:5.2f} dB  (n={x['n']})")
    if a.fit_offsets:
        print("fitted offsets", out["fitted_offsets"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
