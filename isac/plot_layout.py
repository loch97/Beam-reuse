"""Common horizontal geometry for the three-panel paper result figures."""

from matplotlib.transforms import Bbox

LEFT, RIGHT, WSPACE = (0.075, 0.94, 0.48)


def align_three_columns(axes):
    width = (RIGHT - LEFT) / (3 + 2 * WSPACE)
    for i, ax in enumerate(axes):
        old = ax.get_position()
        ax.set_position([LEFT + i * width * (1 + WSPACE), old.y0, width, old.height])


def save_aligned_result(fig, path, dpi):
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    bbox = Bbox.from_extents(0, tight.y0 - 0.025, fig.get_figwidth(), tight.y1 + 0.025)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox, pad_inches=0)
