"""Shared style for the CatFORGE Results figures (Nature Catalysis widths: 88 mm single, 180 mm double column).

Reproducibility package: style of the computation scripts in figures/compute/. Their first-draft figures are not part
of the manuscript; when the figure scripts in figures/ run them, save() is replaced and nothing is written. Run on
their own, they save the drafts to output/compute_drafts/."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "output" / "compute_drafts"
MM = 1 / 25.4
DOUBLE, SINGLE = 180 * MM, 88 * MM
DS_COL = {"drm": "#0072B2", "wgs": "#D55E00", "ocm": "#009E73"}
DS_NAME = {"drm": "DRM", "wgs": "WGS", "ocm": "OCM"}
INK, MUTED, GRID = "#1A1A1A", "#5A6770", "#E3E7EA"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "lines.linewidth": 1.1, "savefig.dpi": 600, "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.edgecolor": "#333333", "axes.labelcolor": INK, "xtick.color": "#333333", "ytick.color": "#333333",
})


def panel(ax, letter, title=None, x=-0.16, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)
    if title:
        ax.set_title(title, loc="left", fontsize=7.2, color=INK, pad=4)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.02)
    print("saved", OUT / f"{name}.pdf")
