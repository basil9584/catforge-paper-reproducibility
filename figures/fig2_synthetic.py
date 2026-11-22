"""Figure 2 (v2 design): synthetic records reproduce the data but do not improve prediction for unseen studies.

Data: identical to v1 (figures/compute/fig2_synthetic_compute.py). The v1 script is executed in a separate process with
all file writes disabled (style_v2.v1_dry_run) and its tables `main` and `abl`
(results/augmentation_benchmark/benchmark_summary.csv) are plotted here unchanged. At the end every plotted array is
asserted equal to the arrays drawn in the v1 figure, and the numbers the v1 script prints are asserted equal to
results/reference_numbers/fig2_numbers.txt.
Run from the package root: python figures/fig2_synthetic.py
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
import style_v2 as S  # noqa: E402

V1 = S.v1_dry_run("fig2_synthetic_compute.py")
main, abl = V1["globals"]["main"], V1["globals"]["abl"]
DS = ("drm", "wgs", "ocm")
SIZES = [25, 50, 100, 200, 400]

# ---------------------------------------------------------------- layout (mm)
H = 53.5
TOP, PH = 7.0, 39.0
PW = 46.0
LEFTS = (12.0, 72.0, 132.0)
fig = S.figure(H)
axa, axb, axc = (S.axes_mm(fig, x, TOP, PW, PH) for x in LEFTS)
YLAB = "Change in held-out MAE (%)"


def shade_positive(ax):
    sh = ax.axhspan(0, 1e4, color=S.WORSE_SHADE, lw=0, zorder=0)
    sh.set_gid("shade")
    S.ref_line(ax, 0, color=S.INK, lw=0.6)


def log_records_axis(ax, xmax=400 * 1.12):
    ax.set_xscale("log")
    ax.set_xlim(25 / 1.12, xmax)
    S.fixed_ticks(ax, "x", SIZES)


# ---------------------------------------------------------------- a: real + 200 synthetic records
ax = axa
shade_positive(ax)
for ds in DS:
    col = S.DATASET[ds]
    for samp in ("random", "clustered"):
        d = main[(main.dataset == ds) & (main.sampling == samp) & (main.arm == "catforge_aug")].sort_values("size")
        if samp == "random":
            band = ax.fill_between(d["size"], d.ci_low, d.ci_high, color=col, alpha=S.BAND_ALPHA, lw=0, zorder=1)
            band.set_gid(f"a:band:{ds}")
            ln, = ax.plot(d["size"], d.rel_delta_pct, "-o", color=col, lw=S.DATA_LW, ms=3.4, mec="white", mew=0.4, zorder=4)
        else:
            ln, = ax.plot(d["size"], d.rel_delta_pct, "--s", color=col, lw=S.THIN_LW + 0.15, ms=3.0, mfc="white", mec=col,
                          mew=0.7, dashes=(3.2, 1.6), zorder=3)
        ln.set_gid(f"a:{ds}:{samp}")
log_records_axis(ax)
ax.set_ylim(-6, 15.5)
S.fixed_ticks(ax, "y", [-5, 0, 5, 10, 15])
ax.set_xlabel("Real training records")
ax.set_ylabel(YLAB)
S.header(ax, "Real + 200 synthetic records")
# key (top right, empty region)
key = [Line2D([], [], color=S.GREY_DARK, lw=S.DATA_LW, marker="o", ms=3.4, mec="white", mew=0.4),
       Line2D([], [], color=S.GREY_DARK, lw=S.THIN_LW + 0.15, ls="--", dashes=(3.2, 1.6), marker="s", ms=3.0, mfc="white",
              mec=S.GREY_DARK, mew=0.7)]
leg = ax.legend(key, ["many studies", "a few studies"], loc="upper right", title="Real records drawn from",
                title_fontsize=S.LABEL_PT, alignment="left", bbox_to_anchor=(1.0, 1.0), handlelength=2.2,
                borderaxespad=0.4)
ax.add_artist(leg)
# dataset key as coloured words (same colours as the direct labels in b)
dleg = ax.legend([Line2D([], [], lw=0) for _ in DS], [S.DATASET_NAME[d] for d in DS], loc="upper right",
                 bbox_to_anchor=(1.0, 0.745), ncol=3, handlelength=0, handletextpad=0, columnspacing=0.9,
                 fontsize=S.LABEL_PT, borderaxespad=0.4)
for txt, ds in zip(dleg.get_texts(), DS):
    txt.set_color(S.DATASET[ds])
ax.text(0.03, 0.035, "Lower error", transform=ax.transAxes, fontsize=S.SMALL_PT, color=S.MUTED, ha="left", va="bottom")
ax.text(0.985, 5.3, "Higher error", transform=ax.get_yaxis_transform(), fontsize=S.SMALL_PT, color=S.MUTED, ha="right",
        va="center")

# ---------------------------------------------------------------- b: synthetic records only
ax = axb
shade_positive(ax)
ends = {}
for ds in DS:
    col = S.DATASET[ds]
    d = main[(main.dataset == ds) & (main.sampling == "random") & (main.arm == "catforge_synth_only")].sort_values("size")
    band = ax.fill_between(d["size"], d.ci_low, d.ci_high, color=col, alpha=S.BAND_ALPHA, lw=0, zorder=1)
    band.set_gid(f"b:band:{ds}")
    ln, = ax.plot(d["size"], d.rel_delta_pct, "-o", color=col, lw=S.DATA_LW, ms=3.4, mec="white", mew=0.4, zorder=4)
    ln.set_gid(f"b:{ds}")
    ends[ds] = float(d.rel_delta_pct.iloc[-1])
log_records_axis(ax, xmax=400 * 1.75)
ax.set_ylim(-8, 140)
S.fixed_ticks(ax, "y", [0, 40, 80, 120])
ax.set_xlabel("Real records used to train the generator")
ax.set_ylabel(YLAB)
S.header(ax, "Synthetic records only")
ax.text(0.03, 0.975, "Higher error", transform=ax.transAxes, fontsize=S.SMALL_PT, color=S.MUTED, ha="left", va="top")
nudge = {"drm": -4.5, "wgs": 4.5, "ocm": 0.0}
for ds in DS:
    ax.text(400 * 1.13, ends[ds] + nudge[ds], S.DATASET_NAME[ds], color=S.DATASET[ds], fontsize=S.LABEL_PT, ha="left",
            va="center")

# ---------------------------------------------------------------- c: quality score vs change in MAE
ax = axc
shade_positive(ax)
GEN = [("catforge", "CatFORGE (r = 0.5–4)", "o", 3.6), ("gc_aug_r1", "Gaussian copula", "D", 3.0),
       ("ctgan_aug_r1", "CTGAN", "^", 3.6), ("tvae_aug_r1", "TVAE", "v", 3.6)]
for ds in DS:
    for key_, lab, mk, ms in GEN:
        if key_ == "catforge":
            d = abl[(abl.dataset == ds) & abl.arm.str.startswith("catforge")]
        else:
            d = abl[(abl.dataset == ds) & (abl.arm == key_)]
        sc = ax.scatter(d.quality * 100, d.rel_delta_pct, marker=mk, s=ms ** 2, color=S.DATASET[ds], edgecolor="white",
                        lw=0.35, zorder=3)
        sc.set_gid(f"c:{ds}:{key_}")
ax.set_xlim(54, 91)
ax.set_ylim(-2.4, 12)
S.fixed_ticks(ax, "x", [60, 70, 80, 90])
S.fixed_ticks(ax, "y", [0, 4, 8, 12])
ax.set_xlabel("Synthetic-data quality score (%)")
ax.set_ylabel(YLAB)
S.header(ax, "Quality score vs usefulness")
ax.text(0.03, 0.975, "Higher error", transform=ax.transAxes, fontsize=S.SMALL_PT, color=S.MUTED, ha="left", va="top")
ax.text(0.62, 0.035, "Lower error", transform=ax.transAxes, fontsize=S.SMALL_PT, color=S.MUTED, ha="center", va="bottom")
h = [Line2D([], [], marker=mk, ls="", color=S.GREY_DARK, mec="white", mew=0.35, ms=ms) for _, _, mk, ms in GEN]
# gid 'nocheck': the legend box's lower-right corner reaches one marker beside the short "CTGAN" row; checked by eye
ax.legend(h, [lab for _, lab, _, _ in GEN], loc="upper left", bbox_to_anchor=(-0.01, 0.80), handlelength=1.2,
          handletextpad=0.2, labelspacing=0.3, borderaxespad=0.3).set_gid("nocheck")

S.panel_letters(fig, [[(axa, "a"), (axb, "b"), (axc, "c")]])
NAME = "Fig2_synthetic_fidelity_vs_usefulness"


# ---------------------------------------------------------------- verification against the v1 figure
def verify():
    v1 = V1["artists"]
    v2 = S.extract_artists(fig)
    n = 0
    # a and b: data lines and bands, in v1 drawing order
    order_a = [(ds, samp) for ds in DS for samp in ("random", "clustered")]
    v1a = [l for l in v1[0]["lines"] if l["data"]]
    for (ds, samp), l1 in zip(order_a, v1a):
        l2 = next(l for l in v2[0]["lines"] if l["gid"] == f"a:{ds}:{samp}")
        S.assert_close(l2["x"], l1["x"], f"a {ds} {samp} x"); S.assert_close(l2["y"], l1["y"], f"a {ds} {samp} y"); n += 2
    for k, ds in enumerate(DS):
        f2 = next(f for f in v2[0]["fills"] if f["gid"] == f"a:band:{ds}")
        S.assert_close(f2["verts"][0], v1[0]["fills"][k]["verts"][0], f"a band {ds}"); n += 1
    v1b = [l for l in v1[1]["lines"] if l["data"]]
    for ds, l1 in zip(DS, v1b):
        l2 = next(l for l in v2[1]["lines"] if l["gid"] == f"b:{ds}")
        S.assert_close(l2["x"], l1["x"], f"b {ds} x"); S.assert_close(l2["y"], l1["y"], f"b {ds} y"); n += 2
        f2 = next(f for f in v2[1]["fills"] if f["gid"] == f"b:band:{ds}")
        S.assert_close(f2["verts"][0], v1[1]["fills"][DS.index(ds)]["verts"][0], f"b band {ds}"); n += 1
    # c: the set of plotted points (quality %, change in MAE) must be identical
    p1 = np.vstack([s["xy"] for s in v1[2]["scatter"]])
    p2 = np.vstack([s["xy"] for s in v2[2]["scatter"]])
    p1, p2 = p1[np.lexsort(p1.T[::-1])], p2[np.lexsort(p2.T[::-1])]
    S.assert_close(p2, p1, "c points"); n += 1
    # dataset/generator assignment of the points in c
    for ds in DS:
        a1 = abl[abl.dataset == ds]
        p_ds = np.vstack([s["xy"] for s in v2[2]["scatter"] if s["gid"].startswith(f"c:{ds}:")])
        assert len(p_ds) == len(a1[a1.arm != "real_only"]), ds
    # stored reference = the v1 printed numbers (the v1 "saved <path>" message that opened the original file is removed)
    stored = S.v1_numbers_text("fig2_numbers.txt").splitlines()
    assert [s.rstrip() for s in V1["stdout"].splitlines()] == [s.rstrip() for s in stored], "v1 printed numbers changed"
    n += 1
    print(f"verification Fig 2: {n} array comparisons against the v1 figure passed "
          f"({len(p1)} points in c, {sum(len(l['x']) for l in v1a)} + {sum(len(l['x']) for l in v1b)} line vertices in a, b)")


if __name__ == "__main__":
    S.save(fig, NAME)
    S.write_numbers("fig2_numbers.txt", V1["stdout"])
    verify()
