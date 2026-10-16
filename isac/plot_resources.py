"""Dedicated sensing demand, communication cost and grant-count effects."""

import json
import shutil
import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .plot_style import apply

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/figures"
INPUT = ROOT / "data/reference/resource_summary.json"
FAMS = [
    ("Unmodified", "#666666", "#eeeeee", "o"),
    ("Beam steering", "#388443", "#a1d99b", "s"),
    ("Broadened beam", "#d46b18", "#fdae6b", "^"),
    ("Multibeam", "#397fa8", "#9ecae1", "D"),
    ("Sequential", "#766a94", "#bcb0cf", "v"),
]


def error(ax, x, y, ci, scale=1, capsize=1.4):
    lo, hi = np.asarray(ci) * scale
    ax.errorbar(
        x,
        y,
        yerr=[[max(0, y - lo)], [max(0, hi - y)]],
        fmt="none",
        ecolor=".2",
        elinewidth=0.6,
        capsize=capsize,
        capthick=0.6,
        zorder=5,
    )


def main():
    cost_legend_outside = reader_labels = align_panels = True
    data = json.loads(INPUT.read_text())
    assert data["config"]["seed_count"] == 64
    apply()
    plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.65})
    plt.rcParams["hatch.linewidth"] = 0.35
    fig, axs = plt.subplots(
        1, 3, figsize=(7.05, 2.3), gridspec_kw={"width_ratios": [1.2, 1, 1.1]}
    )
    ax = axs[0]
    maximum = 0
    for i, c in enumerate(["narrow", "inter", "broad"]):
        for u, ue in enumerate(["U0", "U1"]):
            for j, (p, _, fill, _) in enumerate(FAMS[1:4]):
                row = data["conditions"][c + "_" + ue][p]["dedicated"]
                v = 100 * np.array(row["mean"])[:, 4]
                ci = np.array(row["ci"])[0, 4]
                x = i + (u - 0.5) * 0.42 + (j - 1) * 0.13
                ax.bar(
                    x,
                    v[0],
                    0.12,
                    color=fill,
                    edgecolor=".35",
                    lw=0.35,
                    hatch="///" if u else None,
                )
                error(ax, x, v[0], ci, 100, capsize=0.9)
                ax.plot([x - 0.048, x + 0.048], [v[1], v[1]], color="k", lw=0.8)
                ax.plot(
                    [x - 0.048, x + 0.048],
                    [v[2], v[2]],
                    color="k",
                    lw=0.8,
                    ls=(0, (1.5, 1.2)),
                    zorder=6,
                )
                maximum = max(maximum, 100 * ci[1])
    ymax = max(40, 10 * np.ceil(maximum / 10))
    ax.set_xticks(range(3), ["Narrow", "Interm.", "Broad"])
    ax.set(xlabel="Target uncertainty", ylabel="Dedicated sensing (%)", ylim=(0, ymax))
    ax.set_yticks(np.arange(0, ymax + 1, 10))
    ue_legend_location = dict(loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.legend(
        handles=[
            Patch(facecolor="white", edgecolor=".35", lw=0.35, label="U0"),
            Patch(facecolor="white", edgecolor=".35", lw=0.35, hatch="///", label="U1"),
        ],
        **ue_legend_location,
        ncol=2,
        frameon=False,
        fontsize=6.5,
        handlelength=1.2,
        columnspacing=0.6,
    )
    ax = axs[1]
    maximum = 0
    for j, (p, _, fill, _) in enumerate(FAMS[1:]):
        row = data["conditions"]["broad_U1"][p]
        for q in range(3):
            x = q + (j - 1.5) * 0.2
            displacement = row["displacement"]["mean"][q][4]
            adaptation = row["adaptation"]["mean"][q][4]
            ax.bar(x, displacement, 0.184, color=fill, edgecolor=".3", lw=0.3)
            ax.bar(
                x,
                adaptation,
                0.184,
                bottom=displacement,
                color=fill,
                edgecolor=".3",
                lw=0.3,
                hatch="\\\\\\",
            )
            ci = row["cost"]["ci"][q][4]
            error(ax, x, displacement + adaptation, ci)
            maximum = max(maximum, ci[1])
    ymax = max(1500, 500 * np.ceil(maximum / 500))
    ax.set_xticks(range(3), ["$\\Delta m=0$", "$\\Delta m=3$", "Floor"])
    ax.set(xlabel="MCS allowance", ylabel="Communication cost (bit/interval)", ylim=(0, ymax))
    ax.set_yticks(np.arange(0, ymax + 1, 500))
    cost_legend_location = dict(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax.legend(
        handles=[
            Patch(facecolor="white", edgecolor=".35", lw=0.35, label="Displaced payload"),
            Patch(
                facecolor="white", edgecolor=".35", lw=0.35, hatch="\\\\\\", label="Rate loss"
            ),
        ],
        **cost_legend_location,
        frameon=False,
        fontsize=6.5,
        handlelength=1.2,
    )
    ax = axs[2]
    ns = np.arange(1, 11)
    for p, line, fill, marker in FAMS:
        row = data["conditions"]["broad_U1"][p]["dedicated"]
        v = 100 * np.array(row["mean"])[0]
        ci = 100 * np.array(row["ci"])[0]
        ax.fill_between(ns, ci[:, 0], ci[:, 1], color=line, alpha=0.28, lw=0)
        sequential = p == "Sequential"
        marks = [0, 2, 4, 6, 8] if sequential else [1, 3, 5, 7, 9] if p == "Multibeam" else None
        ax.plot(
            ns,
            v,
            color=line,
            lw=1.25,
            ls="--" if sequential else "-",
            marker=marker,
            ms=4.0 if sequential else 3.4,
            mfc="white" if sequential else fill,
            mec=line,
            mew=0.65,
            markevery=marks,
            label="Sequential policy" if sequential else p,
        )
    ax.set(
        xlim=(0.7, 10.3),
        ylim=(0, 100),
        xlabel="Scheduled grants $N$",
        ylabel="Dedicated sensing (%)",
    )
    ax.set_xticks([1, 2, 3, 5, 8, 10])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1, 1.06),
        frameon=False,
        fontsize=6.5,
        handlelength=1.5,
        labelspacing=0.15,
        borderaxespad=0.3,
    )
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=".87", lw=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7, length=2, pad=1)
        ax.xaxis.label.set_size(7.5)
        ax.yaxis.label.set_size(7.5)
    handles = [
        Patch(
            facecolor=fill,
            edgecolor=".35",
            lw=0.35,
            label="Sequential policy" if p == "Sequential" else p,
        )
        for p, _, fill, _ in FAMS[1:]
    ]
    handles += [
        Line2D([], [], color="k", lw=0.8, label="$\\Delta m=3$"),
        Line2D([], [], color="k", lw=0.8, ls=(0, (1.5, 1.2)), label="Floor MCS"),
    ]
    fig.legend(
        handles=handles[:4],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        fontsize=7.5,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.4,
    )
    axs[0].add_artist(axs[0].get_legend())
    axs[0].legend(
        handles=handles[4:],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=6.5,
        frameon=False,
        handlelength=1.1,
        labelspacing=0.15,
        borderaxespad=0.3,
    )
    for ax, label in zip(
        axs,
        [
            "(a) Dedicated sensing, N = 5",
            "(b) Communication loss, N = 5",
            "(c) Effect of grant count",
        ],
    ):
        ax.text(0.5, -0.4, label, transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
    fig.subplots_adjust(left=0.063, right=0.995, top=0.8, bottom=0.32, wspace=0.45)
    from .plot_layout import align_three_columns, save_aligned_result

    align_three_columns(axs)
    for ext in ["pdf", "png"]:
        save_aligned_result(fig, OUT / f"fig4_three_panel.{ext}", dpi=260)
    plt.close(fig)


if __name__ == "__main__":
    main()
