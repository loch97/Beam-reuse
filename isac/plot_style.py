"""Register the supplied paper fonts without a local TeX installation."""

from pathlib import Path
import matplotlib as mpl
from matplotlib import font_manager


def apply():
    fonts = Path(__file__).resolve().parents[1] / "assets/fonts"
    for style in ["Regular", "Italic", "Bold", "BoldItalic"]:
        font_manager.fontManager.addfont(str(fonts / f"PaperNimbusRoman-{style}.otf"))
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Paper Nimbus Roman"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
