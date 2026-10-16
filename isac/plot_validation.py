"""Simulated robustness and RT boundary-probe comparison."""

from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch, Patch
from matplotlib.colors import BoundaryNorm, ListedColormap
from .plot_style import apply
from .plot_layout import align_three_columns, save_aligned_result

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/reference"
RT = ROOT / "data/rt"
OUT = ROOT / "outputs/figures"
PAPER = ROOT / "paper"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    robust = json.loads((BASE / "robustness_summary.json").read_text())
    summary = json.loads((RT / "summary_n16000000.json").read_text())
    assert robust["config"]["seeds"] == 64
    assert summary["status"] == "complete" and summary["completed_scenes"] == 144
    rows = [
        r
        for r in json.loads((RT / "all_records_n16000000.json").read_text())
        if r["selector"] == "boundary"
    ]
    assert len(rows) == 926
    assert sum((r["prediction"] >= 0.9 for r in rows)) == 146
    assert sum((r["prediction"] >= 0.9 and r["entity_ue_pd"] < 0.9 for r in rows)) == 9
    OUT.mkdir(exist_ok=True)
    apply()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 1.94))
    colors = ["#a1d99b", "#fdae6b", "#9ecae1", "#bcb0cf"]
    policies = ["Beam steering", "Broadened beam", "Multibeam", "Sequential"]
    x = np.arange(4)
    scores = robust["scores"]
    bar_data = {}
    for i, (label, color) in enumerate(zip(policies, colors)):
        r = scores[label]
        bar_data[label] = {}
        for ax, values, key in zip(
            axes[:2],
            [100 * (1 - np.array(r["P_det"])), 100 * np.array(r["P_out_opp"])],
            ["system_miss_ci", "P_out_opp_ci"],
        ):
            ci = 100 * np.array(r[key])
            err = np.maximum(0, np.array([values - ci[:, 0], ci[:, 1] - values]))
            ax.bar(
                x + (i - 1.5) * 0.19,
                values,
                0.18,
                color=color,
                edgecolor=".25",
                lw=0.4,
                yerr=err,
                error_kw=dict(elinewidth=0.5, capsize=1.1, capthick=0.5),
            )
            bar_data[label][key] = dict(mean_percent=values.tolist(), ci_percent=ci.tolist())
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(color=".88", lw=0.35)
        ax.tick_params(length=2, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:2]:
        ax.set_xticks(x, ["Nominal", "$s=2$", "$\\beta=+2$", "$\\beta=-2$"])
    axes[0].set_ylabel("$1-P_{\\rm det}$ (%)", labelpad=2)
    axes[0].set_ylim(
        0,
        max(
            10,
            2 * np.ceil(100 * max((np.max(r["system_miss_ci"]) for r in scores.values())) / 2),
        ),
    )
    axes[1].set_ylabel("$P_{\\rm out}^{\\rm opp}$ (%)", labelpad=2)
    axes[1].set_ylim(
        0,
        max(
            50,
            10 * np.ceil(100 * max((np.max(r["P_out_opp_ci"]) for r in scores.values())) / 10),
        ),
    )
    zoom = axes[1].inset_axes([0.16, 0.56, 0.3, 0.36])
    for i, (label, color) in enumerate(zip(policies, colors)):
        r = scores[label]
        v = 100 * r["P_out_opp"][0]
        lo, hi = 100 * np.array(r["P_out_opp_ci"][0])
        zoom.bar(
            (i - 1.5) * 0.19,
            v,
            0.18,
            color=color,
            edgecolor=".25",
            lw=0.3,
            yerr=[[max(0, v - lo)], [max(0, hi - v)]],
            error_kw=dict(elinewidth=0.4, capsize=0.8, capthick=0.4),
        )
    top = max(0.1, 100 * max((r["P_out_opp_ci"][0][1] for r in scores.values())) * 1.2)
    zoom.set_ylim(0, top)
    zoom.set_xlim(-0.55, 0.55)
    zoom.set_xticks([])
    zoom.yaxis.set_major_locator(plt.MaxNLocator(3))
    zoom.tick_params(axis="y", labelsize=5.8, length=1.5, pad=1)
    zoom.set_title("Nominal (zoom, %)", fontsize=6.2, pad=1.5)
    zoom.spines[["top", "right"]].set_visible(False)
    axes[1].add_patch(
        Rectangle((-0.43, 0), 0.86, top, fill=False, edgecolor=".45", lw=0.5, ls="--", zorder=4)
    )
    for source_x, inset_x in [(-0.43, 0), (0.43, 1)]:
        axes[1].add_artist(
            ConnectionPatch(
                xyA=(source_x, top),
                coordsA=axes[1].transData,
                xyB=(inset_x, 0),
                coordsB=zoom.transAxes,
                color=".45",
                lw=0.5,
                ls="--",
                zorder=2,
            )
        )
    edges = np.linspace(0, 1, 11)
    counts, _, _ = np.histogram2d(
        [r["prediction"] for r in rows], [r["entity_ue_pd"] for r in rows], bins=(edges, edges)
    )
    assert int(counts.sum()) == len(rows)
    count_edges = [1, 5, 20, 50, 100, 200]
    count_cmap = ListedColormap(["#eeeaf2", "#d5cddd", "#bcb0cf", "#9483ab", "#67547e"])
    mesh = axes[2].pcolormesh(
        edges,
        edges,
        np.ma.masked_equal(counts.T, 0),
        cmap=count_cmap,
        norm=BoundaryNorm(count_edges, count_cmap.N),
        shading="flat",
        zorder=1,
    )
    axes[2].grid(False)
    axes[2].plot([0, 1], [0, 1], color=".5", lw=0.6, zorder=2)
    axes[2].axhline(0.9, color=".15", ls="--", lw=0.65, zorder=3)
    axes[2].axvline(0.9, color=".15", ls="--", lw=0.65, zorder=3)
    axes[2].set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
    cax = axes[2].inset_axes([1.045, 0, 0.07, 1])
    cb = fig.colorbar(mesh, cax=cax, ticks=count_edges, spacing="uniform", drawedges=True)
    cb.ax.set_yticklabels([str(v) for v in count_edges])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=6.5, length=1.5, pad=1)
    cb.ax.set_title("Count", fontsize=7, pad=3)
    cb.outline.set_linewidth(0.35)
    axes[2].set_xlabel("Predicted $\\bar p_D$", labelpad=1)
    axes[2].set_ylabel("RT $p_D$", labelpad=1)
    fig.subplots_adjust(left=0.065, right=0.955, bottom=0.31, top=0.83, wspace=0.44)
    align_three_columns(axes)
    labels = ["(a) Overall miss probability", "(b) Opportunity outage", "(c) Theory versus RT"]
    for ax, label in zip(axes, labels):
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.055, label, ha="center", va="bottom", fontsize=8)
    fig.legend(
        [Patch(facecolor=c, edgecolor=".25", linewidth=0.4) for c in colors],
        ["Beam steering", "Broadened beam", "Multibeam", "Sequential policy"],
        ncol=4,
        loc="upper center",
        frameon=False,
        fontsize=7.5,
        handlelength=1.5,
        columnspacing=1.4,
        bbox_to_anchor=(0.5, 1.015),
    )
    for ext in ["pdf", "png"]:
        save_aligned_result(fig, OUT / f"fig_robustness.{ext}", dpi=250)
    plt.close(fig)
    (OUT / "figure_data.json").write_text(
        json.dumps(
            dict(
                simulation_ab=bar_data,
                rt_c=rows,
                rt_histogram=dict(
                    edges=edges.tolist(),
                    counts_prediction_by_rt=counts.astype(int).tolist(),
                    color_boundaries=count_edges,
                    empty_bin_color="white",
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    (OUT / "figure_provenance.json").write_text(
        json.dumps(
            dict(
                simulation_ab_source=str(BASE / "robustness_summary.json"),
                simulation_ab_sha256=sha(BASE / "robustness_summary.json"),
                rt_c_source=str(RT / "all_records_n16000000.json"),
                rt_c_sha256=sha(RT / "all_records_n16000000.json"),
                rt_samples_per_source=16000000,
                rt_boundary_probes=926,
                note="Panels a,b preserve original simulation means and confidence intervals. Panel c pools correlated boundary probes selected nearest the threshold on each side within each of three beam families, rather than Sequential-policy actions. Counts use discrete purple-gray levels with white empty bins. No outcomes omitted.",
                script_sha256=sha(__file__),
                installed_pdf_sha256=sha(OUT / "fig_robustness.pdf"),
            ),
            indent=2,
        )
        + "\n"
    )
    print("Figure 5 installed: original simulation a,b plus 926 RT probes in c", flush=True)


if __name__ == "__main__":
    main()
