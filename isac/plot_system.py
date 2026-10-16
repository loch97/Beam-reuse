"""Roadside geometry and gain patterns for the frozen illustrative beams."""

from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from . import model as V

OUT = ROOT / "outputs/figures/fig_system_overview"
OUT.parent.mkdir(parents=True, exist_ok=True)
M = V.M
COL = dict(L0="#7a7a7a", L2="#2a6fdb", S="#e07b00", T="#1b9e77")
INK, INK2, GRAY, LANE, TGT = ("#0b0b0b", "#52514e", "#c3c2b7", "#eeede8", "#c0392b")
z = np.load(ROOT / "data/illustration.npz")
rsu, xy = (z["rsu"], z["xy"])
i, k0 = (int(z["i"]), int(z["k0"]))
ut, su, bore = (float(z["ut"]), float(z["su"]), float(z["bore"]))
UE_XY = z["ue_xy"]
best = {k: dict(w=z["w_" + k]) for k in COL}
row = json.loads((ROOT / "data/illustration_row.json").read_text())


def tx_gain_dbi(w, u):
    return (
        10 * math.log10(M)
        + V.db(V.gain(w, u))[0]
        + V.elem_dbi(np.arcsin(np.clip(u, -1, 1)), 90.0)
    )


def rot(p):
    p = np.atleast_2d(p)
    return np.column_stack([-(p[:, 1] - rsu[1]), p[:, 0]])


plt.rcParams.update(
    {
        "font.size": 7.5,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "font.family": "serif",
    }
)
from .plot_style import apply

apply()
plt.rcParams.update(
    {
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    }
)
fig, (ax, bx) = plt.subplots(
    1, 2, figsize=(7.16, 1.95), gridspec_kw=dict(width_ratios=[1.35, 1])
)
for x0, x1 in ((-7.0, -3.5), (-3.5, 0.0), (0.0, 3.5), (3.5, 7.0), (7.0, 10.5)):
    ax.add_patch(Rectangle((-2, x0), 90, x1 - x0, color=LANE, lw=0, zorder=0))
    ax.plot([-2, 88], [x1, x1], color="white", lw=0.8, ls=(0, (4, 4)), zorder=1)
phis = np.linspace(-math.pi / 2, math.pi / 2, 1441)
R_MAX, DR = (80.0, 20.0)
peak = tx_gain_dbi(best["L0"]["w"], np.array([V.GRID[k0]]))[0]


def footprint(w):
    g = tx_gain_dbi(w, np.sin(phis))
    rr = R_MAX * np.clip((g - peak + DR) / DR, 0, 1)
    pts = rsu + np.column_stack([rr * np.cos(bore + phis), rr * np.sin(bore + phis)])
    return rot(np.vstack([rsu, pts, rsu]))


ax.add_patch(
    Polygon(
        footprint(best["L0"]["w"]), closed=True, fc=COL["L0"], ec="none", alpha=0.25, zorder=2
    )
)
ax.add_patch(
    Polygon(
        footprint(best["S"]["w"]), closed=True, fc=COL["S"], ec="none", alpha=0.18, zorder=3
    )
)
ax.add_patch(
    Polygon(footprint(best["S"]["w"]), closed=True, fc="none", ec=COL["S"], lw=1.1, zorder=4)
)
ax.add_patch(
    Polygon(
        footprint(best["T"]["w"]),
        closed=True,
        fc="none",
        ec=COL["T"],
        lw=0.9,
        ls=(0, (3, 2)),
        zorder=4,
    )
)
ax.add_patch(
    Polygon(
        footprint(best["L2"]["w"]), closed=True, fc="none", ec=COL["L2"], lw=1.0, zorder=4.2
    )
)
center = np.asarray(row["target_xy_m"], float)
length, width = row["truth_extent_LWH_m"][:2]
hd = row["heading_rad"]
ex, ey = (np.array([math.cos(hd), math.sin(hd)]), np.array([-math.sin(hd), math.cos(hd)]))
body = np.array(
    [
        center + a * length * ex / 2 + b * width * ey / 2
        for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))
    ]
)
ax.add_patch(Polygon(rot(body), closed=True, fc="#f3c4bd", ec=TGT, lw=0.8, zorder=5))
tg = xy[i]
arc_u = np.linspace(ut - 2 * su, ut + 2 * su, 40)
arc = rsu + np.linalg.norm(tg - rsu) * np.column_stack(
    [np.cos(bore + np.arcsin(arc_u)), np.sin(bore + np.arcsin(arc_u))]
)
s_, l_ = rot(np.array(UE_XY))[0]
ax.add_patch(
    Rectangle((s_ - 2.25, l_ - 0.95), 4.5, 1.9, fc="#d9d8d0", ec=INK2, lw=0.8, zorder=5)
)
ax.plot(*rot(rsu)[0], marker="s", color=INK, ms=4.5, zorder=7)
BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
ax.annotate(
    "RSU", rot(rsu)[0], xytext=(4, 8), textcoords="offset points", color=INK, fontsize=7
)
ax.annotate(
    "served UE",
    rot(np.array(UE_XY))[0],
    xytext=(-6, 6),
    textcoords="offset points",
    color=INK2,
    fontsize=6.8,
    bbox=BOX,
    zorder=8,
)
ax.annotate(
    "target",
    rot(tg)[0],
    xytext=(-12, 14),
    textcoords="offset points",
    color=TGT,
    fontsize=6.8,
    bbox=BOX,
    zorder=8,
)
ax.set_xlim(-2, 86)
ax.set_ylim(-9.5, 13)
ax.set_aspect("equal")
ax.set_anchor("S")
ax.set_xlabel("Distance along the road (m)")
ax.set_xticks([0, 20, 40, 60, 80])
ax.set_ylabel("Lateral position (m)")
ax.set_yticks([-7, 0, 7])
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.text(0.01, 0.97, "(a)", transform=ax.transAxes, va="top", fontsize=8)
uu = np.linspace(-0.26, 0.36, 2400)
th = np.degrees(np.arcsin(uu))
for f, lab, ls in (
    ("L0", "Scheduled beam", "-"),
    ("L2", "Beam steering", "-"),
    ("S", "Broadened beam", "-"),
    ("T", "Multibeam pattern", (0, (3, 2))),
):
    b = best[f]
    bx.plot(th, tx_gain_dbi(b["w"], uu) - peak, color=COL[f], lw=1.0, ls=ls, label=lab)
tw = (ut - 2 * su, ut + 2 * su)
ue_window = np.degrees(np.arcsin([V.GRID[k0] - 1 / 64, V.GRID[k0] + 1 / 64]))
bx.axvspan(*ue_window, color=COL["L0"], alpha=0.18, lw=0, zorder=0.5)
for edge in ue_window:
    bx.axvline(edge, color=COL["L0"], lw=0.6, ls=(0, (3, 2)), zorder=0.8)
bx.axvspan(*np.degrees(np.arcsin(tw)), color=TGT, alpha=0.12, lw=0, zorder=0.5)
for edge in np.degrees(np.arcsin(tw)):
    bx.axvline(edge, color=TGT, lw=0.6, ls=(0, (3, 2)), zorder=0.8)
bx.text(
    math.degrees(math.asin(V.GRID[k0])),
    2.8,
    "UE\nwindow",
    color=INK2,
    fontsize=6,
    ha="center",
    va="top",
)
target_window = np.degrees(np.arcsin(tw))
bx.text(
    float(np.mean(target_window)),
    2.8,
    "Target\nwindow",
    color=TGT,
    fontsize=6,
    ha="center",
    va="top",
)
bx.set_xlim(th.min(), th.max())
bx.set_ylim(-28, 3.5)
bx.set_xlabel("Azimuth from boresight (deg)")
bx.set_xticks([-10, 0, 10, 20])
bx.set_ylabel("Relative TX gain (dB)")
bx.set_yticks([-20, -10, 0])
bx.grid(False)
for sp in ("top", "right"):
    bx.spines[sp].set_visible(False)
bx.text(0.01, 0.97, "(b)", transform=bx.transAxes, va="top", fontsize=8)
ax.legend(
    *bx.get_legend_handles_labels(),
    loc="lower center",
    ncol=2,
    frameon=False,
    fontsize=6.3,
    bbox_to_anchor=(0.5, 1.1),
    borderaxespad=0
)
fig.tight_layout(pad=0.3, w_pad=2.0)
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=220)
print("saved", OUT)
