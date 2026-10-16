"""Summarize completed RT scores, separating boundary probes and family actions."""

from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]


def tally(rows):
    passing = [r for r in rows if r["prediction"] >= 0.9]
    rejected = [r for r in rows if r["prediction"] < 0.9]
    return dict(
        probes=len(rows),
        predicted_pass=len(passing),
        predicted_reject=len(rejected),
        entity_fail_accepted=sum(r["entity_ue_pd"] < 0.9 for r in passing),
        point_fail_accepted=sum(r["point_ue_pd"] < 0.9 for r in passing),
        entity_pass_rejected=sum(r["entity_ue_pd"] >= 0.9 for r in rejected),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=16000000)
    parser.add_argument("--seed", type=int, default=32)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "data/rt/protocol.json").read_text())
    folder = ROOT / "outputs/rt"
    scores = [
        json.loads(
            (folder / "scores" / f"{c['key']}_n{args.samples}_s{args.seed}.json").read_text()
        )
        for c in protocol["cases"]
    ]
    rows = [r for s in scores for r in s["records"]]
    summary = dict(
        status="complete",
        samples_per_source=args.samples,
        completed_scenes=len(scores),
        trajectories_completed=48,
        boundary=tally([r for r in rows if r["selector"] == "boundary"]),
        action=tally([r for r in rows if r["selector"] == "action"]),
    )
    for name, value in [
        (f"all_records_n{args.samples}.json", rows),
        (f"summary_n{args.samples}.json", summary),
    ]:
        (folder / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
