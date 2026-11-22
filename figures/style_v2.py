"""Shared style for the CatFORGE figures, version 2 (Nature Catalysis, 183 mm double column).

Design only: this module holds no data. Usage
    import style_v2 as S                        # applies the rcParams on import
    fig = S.figure(60)                          # 183 mm wide, 60 mm high (final print size)
    ax = S.axes_mm(fig, 14, 6, 44, 40)          # left, top, width, height in mm from the figure's top-left corner
    ...
    S.panel_letters(fig, [[(ax, "a"), (ax2, "b")]])   # one list per row; a stacked panel may pass a list of axes
    S.save(fig, "Fig2_synthetic_fidelity_vs_usefulness")   # pdf + svg + png (600 dpi) in output/figures

Reproducibility package: paths are relative to the package root. The "v1" scripts of the authors' working folders are
the computation scripts in figures/compute/ (they compute every plotted value and draw a first-draft figure that is
never saved); the figure scripts in figures/ plot exactly the values those scripts compute (see v1_dry_run).
Set CATFORGE_FIG_OUT to save figures elsewhere and CATFORGE_RESULTS_DIR to plot Figs 2-3 from re-run results.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.collections import LineCollection, PathCollection, PolyCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# ----------------------------------------------------------------------------------------------------------------------
# paths (package root = parent of figures/)
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("CATFORGE_FIG_OUT", ROOT / "output" / "figures")).resolve()
V1_SCRIPTS = Path(__file__).resolve().parent / "compute"              # computation scripts ("v1")
V1_NUMBERS = ROOT / "results" / "reference_numbers"                     # stored reference numbers
NUMBERS_OUT = ROOT / "output" / "numbers"                               # regenerated numbers (written by the figure scripts)
CACHE_DIR = ROOT / "output" / "cache"                                   # dry-run results (reproduce.py clears it per run)
ASSETS = Path(__file__).resolve().parent / "assets"

# ----------------------------------------------------------------------------------------------------------------------
# geometry (all design at final print size)
MM = 1 / 25.4                 # inches per mm
PT_PER_MM = 72 / 25.4
WIDTH_MM = 183.0              # double column
MAX_HEIGHT_MM = 170.0

# type sizes (pt)
TICK_PT = 6
LABEL_PT = 7                  # axis labels, legends, direct labels, panel headers
SMALL_PT = 6                  # secondary annotations and value labels
LETTER_PT = 8                 # panel letters, bold lowercase

# line widths (pt)
AXIS_LW = 0.5
DATA_LW = 1.1
THIN_LW = 0.75
ERR_LW = 0.6
CAP_PT = 1.5
GRID_LW = 0.3
REF_LW = 0.6                  # zero / identity reference lines

# ----------------------------------------------------------------------------------------------------------------------
# colour system (Okabe-Ito based, colour-blind safe)
CATFORGE = "#D55E00"          # vermilion: CatFORGE only (filled = current, open = before repair)
GREY_DARK = "#4D4D4D"
GREY_MID = "#8C8C8C"
GREY_LIGHT = "#BDBDBD"
BEST_OTHER = "#0072B2"        # best competitor, where a comparison needs one
DATASET = {"drm": "#0072B2", "wgs": "#E69F00", "ocm": "#009E73"}
DATASET_NAME = {"drm": "DRM", "wgs": "WGS", "ocm": "OCM"}
RESPONSE = {"ch4": "#0072B2", "co2": "#56B4E9", "h2co": "#CC79A7"}
OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
             "blue": "#0072B2", "vermilion": "#D55E00", "purple": "#CC79A7"}
INK = "#1A1A1A"               # text and axes
MUTED = "#6B6B6B"             # secondary annotation text
GRID = "#E5E5E5"
REF = "#4D4D4D"               # reference lines (zero, identity)
WORSE_SHADE = "#F3F3F3"       # background of a 'worse than reference' region
BAND_ALPHA = 0.20             # uncertainty bands (alpha of the line colour)

# ----------------------------------------------------------------------------------------------------------------------
RC = {
    "font.family": "Arial", "font.size": LABEL_PT, "text.color": INK,
    "mathtext.default": "regular", "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial", "axes.unicode_minus": True,
    "axes.labelsize": LABEL_PT, "axes.titlesize": LABEL_PT, "axes.titleweight": "normal", "axes.titlelocation": "left",
    "axes.titlepad": 4.0, "axes.labelpad": 3.0, "axes.labelcolor": INK, "axes.edgecolor": INK, "axes.linewidth": AXIS_LW,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True, "axes.grid": False,
    "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
    "grid.color": GRID, "grid.linewidth": GRID_LW, "grid.linestyle": "-",
    "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT, "xtick.color": INK, "ytick.color": INK,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": AXIS_LW, "ytick.major.width": AXIS_LW, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "lines.linewidth": DATA_LW, "lines.markersize": 3.5, "lines.markeredgewidth": 0.6, "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "butt", "patch.linewidth": 0.5, "hatch.linewidth": 0.5,
    "errorbar.capsize": 0.0, "scatter.edgecolors": "face",
    "legend.fontsize": LABEL_PT, "legend.frameon": False, "legend.borderaxespad": 0.3, "legend.borderpad": 0.2,
    "legend.handlelength": 1.6, "legend.handleheight": 0.7, "legend.handletextpad": 0.5, "legend.labelspacing": 0.35,
    "legend.columnspacing": 1.2, "legend.markerscale": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "catforge-v2",
    "savefig.dpi": 600, "figure.dpi": 150, "path.simplify": False,
}
plt.rcParams.update(RC)


# ----------------------------------------------------------------------------------------------------------------------
# small helpers
def tint(color, frac):
    """Opaque mix of `color` with white; frac = share of the colour (1 = colour, 0 = white)."""
    c = np.array(mcolors.to_rgb(color))
    return tuple(1 - frac * (1 - c))


def shade(color, frac):
    """Opaque mix of `color` with black; frac = share of the colour."""
    c = np.array(mcolors.to_rgb(color))
    return tuple(frac * c)


def figure(height_mm, width_mm=WIDTH_MM):
    if height_mm > MAX_HEIGHT_MM + 1e-6:
        raise ValueError(f"figure height {height_mm} mm exceeds {MAX_HEIGHT_MM} mm")
    return plt.figure(figsize=(width_mm * MM, height_mm * MM))


def axes_mm(fig, left, top, width, height, **kw):
    """Add axes at an exact position given in mm from the figure's top-left corner."""
    W, H = fig.get_figwidth() / MM, fig.get_figheight() / MM
    return fig.add_axes([left / W, 1 - (top + height) / H, width / W, height / H], **kw)


def header(ax, text, pad=4.0):
    """Optional panel header: short, sentence case, left-aligned, not bold, 7 pt."""
    return ax.set_title(text, loc="left", fontsize=LABEL_PT, pad=pad, color=INK)


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, lw=GRID_LW, zorder=0)
    ax.set_axisbelow(True)


def ref_line(ax, value=0.0, orientation="h", **kw):
    """Reference line (zero, ratio 1, ...) behind the data."""
    kw = {"color": REF, "lw": REF_LW, "zorder": 1.5, **kw}
    ln = ax.axhline(value, **kw) if orientation == "h" else ax.axvline(value, **kw)
    ln.set_gid("ref")
    return ln


def fixed_ticks(ax, axis, ticks, labels=None, minor=False):
    """Set ticks at exact positions with clean labels (no needless decimals, true minus)."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_ticks(ticks, labels=labels if labels is not None else [num(t) for t in ticks])
    if not minor:
        a.set_minor_locator(matplotlib.ticker.NullLocator())


def num(v, nd=None):
    """Format a number for a tick label: no trailing zeros, true minus sign."""
    if nd is None:
        s = f"{v:.6f}".rstrip("0").rstrip(".")
    else:
        s = f"{v:.{nd}f}"
    if s in ("-0", "\u22120"):
        s = "0"
    return s.replace("-", "\u2212")


# ----------------------------------------------------------------------------------------------------------------------
# panel letters
def _renderer(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _panel_bbox(fig, axes, renderer):
    from matplotlib.transforms import Bbox
    bbs = []
    for ax in (axes if isinstance(axes, (list, tuple)) else [axes]):
        bb = ax.get_tightbbox(renderer)
        if bb is not None:
            bbs.append(bb)
    return Bbox.union(bbs).transformed(fig.transFigure.inverted())


def panel_letters(fig, rows, top_mm=None, dx_mm=0.0):
    """Place bold lowercase letters (8 pt). `rows` is a list of rows; each row a list of (axes or [axes...], letter).
    Each letter sits at the left edge of its panel's outer bounding box (tick and axis labels included); all letters
    in a row share one height: the top of the tallest element in that row (or `top_mm` from the figure top)."""
    r = _renderer(fig)
    H = fig.get_figheight() / MM
    W = fig.get_figwidth() / MM
    out = []
    for i, row in enumerate(rows):
        bbs = [(_panel_bbox(fig, axs, r), letter) for axs, letter in row]
        if top_mm is not None:
            y = 1 - (top_mm[i] if isinstance(top_mm, (list, tuple)) else top_mm) / H
        else:
            y = max(bb.y1 for bb, _ in bbs)
        for bb, letter in bbs:
            t = fig.text(bb.x0 + dx_mm / W, y, letter, fontsize=LETTER_PT, fontweight="bold", ha="left", va="top",
                         color=INK)
            t.set_gid("panel-letter")
            out.append(t)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# layout checks
def _ticklabels(axis):
    labs = []
    try:
        ticks = axis._update_ticks()
    except Exception:  # pragma: no cover
        ticks = axis.get_major_ticks()
    for t in ticks:
        for lab in (t.label1, t.label2):
            if lab.get_visible() and lab.get_text():
                labs.append(lab)
    return labs


def _all_texts(fig):
    texts = [t for t in fig.texts if t.get_visible() and t.get_text()]
    for leg in fig.legends:
        texts += [t for t in leg.get_texts() if t.get_visible() and t.get_text()]
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        texts += [t for t in ax.texts if t.get_visible() and t.get_text()]
        for t in (ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label):
            if t.get_visible() and t.get_text():
                texts.append(t)
        texts += _ticklabels(ax.xaxis) + _ticklabels(ax.yaxis)
        for leg in _legends(ax):
            texts += [t for t in leg.get_texts() if t.get_visible() and t.get_text()]
    return texts


def _legends(ax):
    """All visible legends of an axes (the current one and any added with ax.add_artist)."""
    from matplotlib.legend import Legend
    legs = [c for c in ax.get_children() if isinstance(c, Legend) and c.get_visible()]
    return legs


def _data_points_display(ax, step_px=1.5):
    """Display-space points along data lines and at scatter offsets (reference lines and grid excluded)."""
    pts = []
    for ln in ax.lines:
        if not ln.get_visible() or ln.get_gid() == "ref" or ln.get_gid() == "nocheck":
            continue
        xy = ln.get_transform().transform(np.column_stack([ln.get_xdata(orig=False), ln.get_ydata(orig=False)]).astype(float))
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if len(xy) == 0:
            continue
        if ln.get_linestyle() not in ("None", "", " ", "none") and len(xy) > 1:
            seg = []
            for p, q in zip(xy[:-1], xy[1:]):
                n = max(2, int(np.hypot(*(q - p)) / step_px) + 1)
                seg.append(p + (q - p) * np.linspace(0, 1, n)[:, None])
            xy = np.vstack(seg)
        pts.append(xy)
    for c in ax.collections:
        if not c.get_visible() or c.get_gid() in ("nocheck", "ref"):
            continue
        if isinstance(c, PathCollection):
            off = c.get_offsets()
            if len(off):
                pts.append(c.get_offset_transform().transform(np.asarray(off, float)))
        elif isinstance(c, LineCollection):
            for seg in c.get_segments():
                xy = c.get_transform().transform(np.asarray(seg, float))
                if len(xy) > 1:
                    s = []
                    for p, q in zip(xy[:-1], xy[1:]):
                        n = max(2, int(np.hypot(*(q - p)) / step_px) + 1)
                        s.append(p + (q - p) * np.linspace(0, 1, n)[:, None])
                    pts.append(np.vstack(s))
    return np.vstack(pts) if pts else np.zeros((0, 2))


def layout_report(fig, verbose=True, data_margin_px=1.0):
    """Check (1) every text is inside the canvas, (2) no two texts overlap, (3) no in-axes text or legend sits on data
    (line vertices along the path, scatter centres, error-bar segments). Returns a list of issue strings."""
    r = _renderer(fig)
    fb = fig.bbox
    texts = _all_texts(fig)
    issues = []
    ext = []
    from matplotlib.text import Text
    for t in texts:
        try:
            bb = Text.get_window_extent(t, r)
        except Exception:
            continue
        ext.append((t, bb))
        if bb.x0 < fb.x0 - 0.5 or bb.y0 < fb.y0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y1 > fb.y1 + 0.5:
            issues.append(f"CLIPPED: '{t.get_text()}' {bb.bounds}")
    for i in range(len(ext)):
        for j in range(i + 1, len(ext)):
            a, b = ext[i][1], ext[j][1]
            ov_w = min(a.x1, b.x1) - max(a.x0, b.x0)
            ov_h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ov_w > 0.5 and ov_h > 0.5:
                issues.append(f"OVERLAP: '{ext[i][0].get_text()}' x '{ext[j][0].get_text()}' ({ov_w:.1f}x{ov_h:.1f} px)")
    # in-axes texts and legends against data
    for ax in fig.axes:
        pts = _data_points_display(ax)
        if len(pts) == 0:
            continue
        from matplotlib.text import Text
        # Text.get_window_extent: the text box only (an Annotation's own extent would include its leader line)
        cands = [(t, Text.get_window_extent(t, r)) for t in ax.texts
                 if t.get_visible() and t.get_text() and t.get_gid() != "nocheck"]
        for leg in _legends(ax):
            if leg.get_gid() != "nocheck":          # 'nocheck' = checked by eye (rectangular box vs ragged text)
                cands.append((leg, leg.get_window_extent(r)))
        for t, bb in cands:
            m = data_margin_px
            inside = ((pts[:, 0] > bb.x0 - m) & (pts[:, 0] < bb.x1 + m) & (pts[:, 1] > bb.y0 - m) & (pts[:, 1] < bb.y1 + m))
            if inside.any():
                name = t.get_text() if hasattr(t, "get_text") else "legend"
                issues.append(f"ON DATA: '{name}' covers {int(inside.sum())} data sample(s)")
    if verbose:
        print("layout check:", "OK" if not issues else f"{len(issues)} issue(s)")
        for s in issues:
            print("   ", s)
    return issues


# ----------------------------------------------------------------------------------------------------------------------
# saving
def save(fig, name, out_dir=OUT_DIR, check=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w_mm, h_mm = fig.get_figwidth() / MM, fig.get_figheight() / MM
    assert abs(w_mm - WIDTH_MM) < 0.01 or w_mm < WIDTH_MM, f"width {w_mm:.2f} mm"
    assert h_mm <= MAX_HEIGHT_MM + 1e-6, f"height {h_mm:.2f} mm"
    issues = layout_report(fig) if check else []
    meta_pdf = {"Creator": "CatFORGE figures v2 (matplotlib)", "CreationDate": None, "ModDate": None}
    fig.savefig(out_dir / f"{name}.pdf", metadata=meta_pdf)
    fig.savefig(out_dir / f"{name}.svg", metadata={"Date": None, "Creator": "CatFORGE figures v2 (matplotlib)"})
    fig.savefig(out_dir / f"{name}.png", dpi=600, metadata={"Software": "CatFORGE figures v2 (matplotlib)"})
    shown = out_dir / name
    try:
        shown = shown.relative_to(ROOT)
    except ValueError:
        pass
    print(f"saved {shown}.{{pdf,svg,png}}  ({w_mm:.1f} x {h_mm:.1f} mm)")
    return issues


# ----------------------------------------------------------------------------------------------------------------------
# reuse of the v1 data pipelines without touching v1 outputs
def extract_artists(fig):
    """Numeric content of every axes: lines, scatter offsets, fills, error-bar segments, bar rectangles, texts."""
    out = []
    for ax in fig.axes:
        d = {"lines": [], "scatter": [], "fills": [], "segments": [], "bars": [], "texts": [t.get_text() for t in ax.texts],
             "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(), "xscale": ax.get_xscale(), "yscale": ax.get_yscale()}
        for ln in ax.lines:
            d["lines"].append({"x": np.asarray(ln.get_xdata(), float), "y": np.asarray(ln.get_ydata(), float),
                               "data": ln.get_transform() is ax.transData, "label": ln.get_label(), "gid": ln.get_gid()})
        for c in ax.collections:
            if isinstance(c, PathCollection):
                d["scatter"].append({"xy": np.asarray(c.get_offsets(), float), "gid": c.get_gid(), "label": c.get_label(),
                                     "c": None if c.get_array() is None else np.asarray(c.get_array(), float)})
            elif isinstance(c, LineCollection):
                d["segments"].append({"segs": [np.asarray(s, float) for s in c.get_segments()], "gid": c.get_gid()})
            elif isinstance(c, PolyCollection):
                d["fills"].append({"verts": [np.asarray(p.vertices, float) for p in c.get_paths()], "gid": c.get_gid()})
        for p in ax.patches:
            if isinstance(p, Rectangle):
                d["bars"].append((p.get_x(), p.get_y(), p.get_width(), p.get_height()))
        out.append(d)
    return out


_BOOT = r'''
import sys, io, pickle, runpy, pathlib, contextlib, types
script, outpath = sys.argv[1], sys.argv[2]
args = sys.argv[3:]
sys.path.insert(0, str(pathlib.Path(script).parent))
sys.path.insert(0, STYLE_DIR)
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
matplotlib.figure.Figure.savefig = lambda *a, **k: None          # nothing written by the v1 script
import figstyle
cap = {}
def _save(fig, name):
    cap["fig"], cap["name"] = fig, name
figstyle.save = _save
writes = {}
pathlib.Path.write_text = lambda self, data, *a, **k: writes.__setitem__(str(self), data) or len(data)
sys.argv = [script] + args
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path(script, run_name="__main__")
import style_v2
art = style_v2.extract_artists(cap["fig"]) if "fig" in cap else None
keep = {}
for k, v in g.items():
    if k.startswith("__") or isinstance(v, (types.ModuleType, types.FunctionType, type)):
        continue
    if type(v).__module__.startswith("matplotlib"):
        continue
    try:
        pickle.dumps(v)
        keep[k] = v
    except Exception:
        pass
with open(outpath, "wb") as f:
    pickle.dump({"globals": keep, "artists": art, "stdout": buf.getvalue(), "writes": writes, "name": cap.get("name")}, f)
'''


def v1_dry_run(script_name, args=(), refresh=False):
    """Run a v1 figure script in a separate process with every file write disabled (figure saving and
    Path.write_text are intercepted), and return its module globals (picklable ones), the numeric content of its
    figure, its printed output and any text it tried to write. Results are cached in the temp folder."""
    script = V1_SCRIPTS / script_name
    key = hashlib.sha1(script.read_bytes() + (V1_SCRIPTS / "figstyle.py").read_bytes() + json.dumps(list(args)).encode()
                       + Path(__file__).read_bytes()).hexdigest()[:16]
    cache = CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    pk = cache / f"{script.stem}_{key}.pkl"
    if refresh or os.environ.get("V1_REFRESH") == "1" or not pk.exists():
        boot = _BOOT.replace("STYLE_DIR", repr(str(Path(__file__).parent)))
        res = subprocess.run([sys.executable, "-c", boot, str(script), str(pk), *args], capture_output=True, text=True,
                             cwd=str(ROOT))
        if res.returncode != 0:
            raise RuntimeError(f"v1 dry run of {script_name} failed:\n{res.stderr[-4000:]}")
    with open(pk, "rb") as f:
        return pickle.load(f)


def v1_numbers_text(name):
    return (V1_NUMBERS / name).read_text(encoding="utf-8")


def write_numbers(name, text):
    """Package addition: save the numbers regenerated in this run (output/numbers/<name>) for comparison."""
    NUMBERS_OUT.mkdir(parents=True, exist_ok=True)
    (NUMBERS_OUT / name).write_bytes(text.encode("utf-8"))
    print(f"regenerated numbers: output/numbers/{name}")


def assert_close(a, b, what, tol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, f"{what}: shape {a.shape} != {b.shape}"
    ok = np.allclose(a, b, rtol=0, atol=tol, equal_nan=True)
    assert ok, f"{what}: max |diff| = {np.nanmax(np.abs(a - b))}"
    return True


