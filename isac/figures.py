"""Reproduce the five paper figures from the archived inputs and summaries."""

from pathlib import Path
import argparse
import importlib
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-paper", action="store_true")
    parser.add_argument("--simulation-results", type=Path)
    parser.add_argument("--rt-results", type=Path)
    args = parser.parse_args()
    out = ROOT / "outputs/figures"
    out.mkdir(parents=True, exist_ok=True)
    importlib.import_module(".plot_system", __package__)
    from . import plot_scene, plot_feasibility, plot_resources, plot_validation

    plot_scene.main()
    plot_feasibility.main()
    # Both inputs must be named explicitly when plotting new results.
    if args.simulation_results is not None:
        plot_resources.INPUT = args.simulation_results / "resource_summary.json"
        plot_validation.BASE = args.simulation_results
    if args.rt_results is not None:
        plot_validation.RT = args.rt_results
    plot_resources.main()
    shutil.copy2(out / "fig4_three_panel.pdf", out / "fig_demand.pdf")
    shutil.copy2(out / "fig4_three_panel.png", out / "fig_demand.png")
    plot_validation.main()
    if args.install_paper:
        for name in [
            "fig_system_overview",
            "fig_scene",
            "fig_regime_map",
            "fig_demand",
            "fig_robustness",
        ]:
            shutil.copy2(out / f"{name}.pdf", ROOT / "paper/figs" / f"{name}.pdf")
    print("Figures written to", out)


if __name__ == "__main__":
    main()
