"""Copy the frozen scene illustration, whose annotations are vector graphics."""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "outputs/figures"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "assets/evaluation_scene.pdf", out / "fig_scene.pdf")


if __name__ == "__main__":
    main()
