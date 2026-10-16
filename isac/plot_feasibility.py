"""Feasibility maps from the archived deterministic candidate masks."""

from __future__ import annotations
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "outputs/figures"
LABELS = ("Neither", "Multibeam only", "Broadened beam only", "Both", "Beam steering")
COLORS = ("#eeeeee", "#9ecae1", "#fdae6b", "#8c6bb1", "#a1d99b")


def main():
    z = np.load(ROOT / "data/regime_masks.npz")
    ds, xis = (z["ds"], z["xis"])
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    from .plot_style import apply

    apply()
    fig, axes = plt.subplots(1, 2, figsize=(3.9, 1.55), sharey=True)
    cmap = ListedColormap(COLORS)
    for ax, b in zip(axes, (4, 8)):
        key = f"U1_e0.01_a8_b{b}"
        L2, T = (z[key + "_L2"], z[key + "_T"])
        S = z[key + "_S"] | z[key + "_Scover"]
        lab = np.zeros(L2.shape, int)
        lab[T] = 1
        lab[S] = 2
        lab[S & T] = 3
        lab[L2] = 4
        ax.pcolormesh(
            ds,
            xis,
            lab,
            cmap=cmap,
            norm=BoundaryNorm(np.arange(-0.5, 5.5), 5),
            shading="nearest",
            rasterized=True,
        )
        scr = z[key + "_tail_bound"]
        if scr.any() and (not scr.all()):
            ax.contour(
                ds,
                xis,
                scr.astype(float),
                levels=[0.5],
                colors="k",
                linestyles="--",
                linewidths=0.8,
            )
        ax.set_title(f"Allowed UE gain loss: {b} dB", fontsize=8, pad=2)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 1)
        ax.tick_params(length=2, pad=1)
        ax.set_xlabel("Angular separation $d$", fontsize=7, labelpad=1)
    axes[0].set_ylabel("Target std. $\\sigma_u/W_0$")
    handles = [
        Patch(facecolor=c, edgecolor="0.6", lw=0.3, label=l)
        for c, l in zip(COLORS[::-1], LABELS[::-1])
    ]
    handles.append(Line2D([0], [0], color="k", ls="--", lw=0.8, label="Full-span screen"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=6.5,
        frameon=False,
        handlelength=1.2,
        columnspacing=0.8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.8)
    PAPER.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER / "fig_regime_map.pdf", dpi=300)
    fig.savefig(PAPER / "fig_regime_map.png", dpi=220)
    for b in (4, 8):
        key = f"U1_e0.01_a8_b{b}"
        L2, T = (z[key + "_L2"], z[key + "_T"])
        S = z[key + "_S"] | z[key + "_Scover"]
        n = L2.size
        print(
            f"b={b}: steer {L2.sum()}/{n}, S beyond steer {(S & ~L2).sum()}, T beyond steer {(T & ~L2).sum()}, S only {(S & ~T & ~L2).sum()}, T only {(T & ~S & ~L2).sum()}, both {(S & T & ~L2).sum()}, neither {(~S & ~T & ~L2).sum()}"
        )


if __name__ == "__main__":
    main()
