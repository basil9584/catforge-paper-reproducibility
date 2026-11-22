"""Figure 3 (v2 design): CatFORGE's predictor vs ten tuned baselines on locked test groups.

Data: identical to v1 (figures/compute/fig3_benchmark_compute.py, reading results/predictor_benchmark/final_summary.csv
and final_catforge_vs_baselines.csv). The v1 script runs in a separate process with file writes disabled
(style_v2.v1_dry_run); its tables `summ` and `comp` are plotted here unchanged, and the drawn values are asserted equal to
those in the v1 figure. (Table 1 of the manuscript is checked against the same tables by checks/check_results.py.)
Run from the package root: python figures/fig3_benchmark.py
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, str(Path(__file__).parent))
import style_v2 as S  # noqa: E402

V1 = S.v1_dry_run("fig3_benchmark_compute.py")
g = V1["globals"]
summ, comp, ORDER, LABEL, NAME = g["summ"], g["comp"], g["ORDER"], g["LABEL"], g["NAME"]
TARGET = {"CH4 Conversion": "CH$_4$ conversion", "CO2 Conversion": "CO$_2$ conversion", "Syngas_Ratio": "H$_2$/CO ratio",
          "CO Conversion": "CO conversion", "X(CH4), %": "CH$_4$ conversion", "S(C2), %": "C$_2$ selectivity",
          "Y(C2),%": "C$_2$ yield"}
SHORT = {"lgbm": "LightGBM", "xgb": "XGBoost", "xgb_logit": "XGBoost + logit", "rf": "Random forest", "et": "Extra trees",
         "ridge": "Ridge", "knn": "k-NN", "svr": "SVR", "mlp": "MLP", "mean": "Training mean"}

# row positions: groups of datasets separated by an extra half row
GROUP_GAP = 0.55
ypos, y = [], 0.0
for i, (ds, t) in enumerate(ORDER):
    if i > 0 and ORDER[i - 1][0] != ds:
        y += GROUP_GAP
    ypos.append(y)
    y += 1.0
YMAX = ypos[-1]
YLIM = (YMAX + 0.62, -0.62)          # inverted: first row on top

# ---------------------------------------------------------------- layout (mm)
H = 81.5
AX_TOP, AX_H = 16.0, 55.0
A_LEFT, A_W = 31.0, 55.0
B_LEFT, B_W, B_GAP = 101.0, 24.0, 4.0
ROW_HEAD, ROW_KEY = 1.2, 5.6          # mm from the top: panel letters/headers, keys
fig = S.figure(H)
axa = S.axes_mm(fig, A_LEFT, AX_TOP, A_W, AX_H)
axb = [S.axes_mm(fig, B_LEFT + k * (B_W + B_GAP), AX_TOP, B_W, AX_H) for k in range(3)]


def fig_xy(x_mm, y_mm):
    return x_mm / S.WIDTH_MM, 1 - y_mm / H


def header_mm(x_mm, text):
    fig.text(*fig_xy(x_mm, ROW_HEAD), text, ha="left", va="top", fontsize=S.LABEL_PT, color=S.INK)

# ---------------------------------------------------------------- a: RMSE relative to the best model
ax = axa
full = summ[summ.regime == "full"]
A_VALUES = {}
for yy, (ds, t) in zip(ypos, ORDER):
    cell = full[(full.dataset == ds) & (full.target == t)]
    best = cell.rmse.min()
    rel = {r.model: r.rmse / best for _, r in cell.iterrows()}
    A_VALUES[(ds, t)] = rel
    others = {m: v for m, v in rel.items() if not m.startswith("catforge")}
    best_other = min(others, key=others.get)
    ax.axhline(yy, color=S.GRID, lw=S.GRID_LW, zorder=0).set_gid("ref")
    for m, v in sorted(others.items()):
        if m == best_other:
            continue
        sc = ax.scatter([v], [yy], s=3.0 ** 2, color=S.GREY_LIGHT, edgecolor="white", lw=0.3, zorder=2)
        sc.set_gid(f"a:{ds}:{t}:{m}")
    sc = ax.scatter([others[best_other]], [yy], s=3.6 ** 2, color=S.BEST_OTHER, edgecolor="white", lw=0.3, zorder=3)
    sc.set_gid(f"a:{ds}:{t}:{best_other}")
    ax.text(max(others[best_other], 1.0) + 0.0045, yy - 0.28, SHORT[best_other], color=S.BEST_OTHER,
            fontsize=S.SMALL_PT, ha="left", va="bottom")
    sc = ax.scatter([rel["catforge_v2"]], [yy], s=4.5 ** 2, color=S.CATFORGE, edgecolor="white", lw=0.4, zorder=5)
    sc.set_gid(f"a:{ds}:{t}:catforge_v2")
    sc = ax.scatter([rel["catforge_v1"]], [yy], s=4.5 ** 2, facecolor="none", edgecolor=S.CATFORGE, lw=0.8, zorder=6)
    sc.set_gid(f"a:{ds}:{t}:catforge_v1")
    if (ds, t) == ORDER[0]:
        ax.text(rel["mean"], yy - 0.28, "Training mean", color=S.GREY_MID, fontsize=S.SMALL_PT, ha="center", va="bottom")
ax.set_xlim(0.975, 1.6)
S.fixed_ticks(ax, "x", [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6], labels=["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6"])
ax.set_ylim(*YLIM)
ax.set_yticks(ypos, labels=[TARGET[t] for _, t in ORDER])
ax.tick_params(axis="y", length=0, pad=3, labelsize=S.LABEL_PT)      # 7 pt so that subscripts stay near 5 pt
ax.spines["left"].set_visible(False)
S.ref_line(ax, 1.0, orientation="v", color=S.GREY_MID, lw=0.5, zorder=1)
ax.set_xlabel("RMSE relative to the best model")
header_mm(A_LEFT, "Locked test groups, all development records for training")
key_a = [Line2D([], [], marker="o", ls="", ms=4.5, color=S.CATFORGE, mec="white", mew=0.4),
         Line2D([], [], marker="o", ls="", ms=4.5, mfc="none", mec=S.CATFORGE, mew=0.8),
         Line2D([], [], marker="o", ls="", ms=3.6, color=S.BEST_OTHER, mec="white", mew=0.3),
         Line2D([], [], marker="o", ls="", ms=3.0, color=S.GREY_LIGHT, mec="white", mew=0.3)]
fig.legend(key_a, ["CatFORGE", "CatFORGE before repair", "Best other model (named)", "Other baselines"], ncol=2,
           loc="upper left", bbox_to_anchor=fig_xy(A_LEFT - 1.2, ROW_KEY), bbox_transform=fig.transFigure,
           handletextpad=0.2, columnspacing=1.2, handlelength=1.0, borderaxespad=0.0, borderpad=0.0)
# dataset groups: label and bracket left of the row labels
for ds in ("drm", "wgs", "ocm"):
    ys = [yy for yy, (d, _) in zip(ypos, ORDER) if d == ds]
    ax.annotate(S.DATASET_NAME[ds], xy=(0, np.mean(ys)), xycoords=("figure fraction", "data"),
                xytext=(1.2 / S.WIDTH_MM, 0), textcoords=("figure fraction", "offset points"), ha="left", va="center",
                fontsize=S.LABEL_PT, color=S.INK, annotation_clip=False)
    xb = 8.6 / S.WIDTH_MM
    br, = ax.plot([xb, xb], [ys[0] - 0.32, ys[-1] + 0.32], transform=blended_transform_factory(fig.transFigure, ax.transData),
                  color=S.GREY_LIGHT, lw=0.6, clip_on=False, solid_capstyle="butt")
    br.set_gid("nocheck")

# ---------------------------------------------------------------- b: ratio to best other model, three training sizes
REG = [("50", "50 records"), ("200", "200 records"), ("full", "All records")]
XB = (0.85, 1.36)
CLIP = 1.345
B_VALUES = {}
for k, (rg, title) in enumerate(REG):
    ax = axb[k]
    sh = ax.axvspan(1.0, 10, color=S.WORSE_SHADE, lw=0, zorder=0)
    sh.set_gid("shade")
    S.ref_line(ax, 1.0, orientation="v", color=S.GREY_MID, lw=0.5, zorder=1)
    for yy, (ds, t) in zip(ypos, ORDER):
        sel = (comp.dataset == ds) & (comp.target == t) & (comp.regime == rg) & (comp.baseline == "best_other")
        c2 = comp[sel & (comp.focus == "catforge_v2")].iloc[0]
        c1 = comp[sel & (comp.focus == "catforge_v1")].iloc[0]
        B_VALUES[(ds, t, rg)] = (c2.ratio, c2.ratio_lo, c2.ratio_hi, c2.verdict, c1.ratio)
        loss = c2.verdict == "LOSS"
        assert c2.verdict in ("LOSS", "TIE") and loss == (c2.ratio_lo > 1.0), (ds, t, rg, c2.verdict)
        eb = ax.errorbar([c2.ratio], [yy], xerr=[[c2.ratio - c2.ratio_lo], [c2.ratio_hi - c2.ratio]], fmt="o", ms=4.2,
                         color=S.CATFORGE, mfc=S.CATFORGE if loss else S.tint(S.CATFORGE, 0.28), mec=S.CATFORGE,
                         mew=0.7, elinewidth=S.ERR_LW + 0.1, capsize=0, zorder=4)
        eb[0].set_gid(f"b:{ds}:{t}:{rg}:v2")
        eb[2][0].set_gid(f"b:{ds}:{t}:{rg}:ci")
        xv1 = min(c1.ratio, CLIP)
        clipped = c1.ratio > CLIP
        sc = ax.scatter([xv1], [yy + 0.30], marker=">" if clipped else "D", s=(3.6 if clipped else 3.1) ** 2,
                        facecolor="white", edgecolor=S.CATFORGE, lw=0.7, zorder=3, clip_on=False)
        sc.set_gid(f"b:{ds}:{t}:{rg}:v1")
        if clipped:
            ax.text(CLIP - 0.022, yy + 0.30, f"{c1.ratio:.2f}", fontsize=S.SMALL_PT, color=S.CATFORGE, ha="right",
                    va="center")
    ax.set_xlim(*XB)
    S.fixed_ticks(ax, "x", [0.9, 1.0, 1.1, 1.2, 1.3], labels=["0.9", "1.0", "1.1", "1.2", "1.3"])
    ax.set_ylim(*YLIM)
    ax.set_yticks(ypos, labels=[])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(title, loc="center", fontsize=S.LABEL_PT, pad=3)
axb[1].set_xlabel("RMSE ratio, CatFORGE / best other model")
header_mm(B_LEFT, "CatFORGE vs best other model, by number of training records")
key_b = [Line2D([], [], marker="o", ls="-", lw=S.ERR_LW + 0.1, ms=4.2, color=S.CATFORGE, mec=S.CATFORGE, mew=0.7),
         Line2D([], [], marker="D", ls="", ms=3.1, mfc="white", mec=S.CATFORGE, mew=0.7),
         Line2D([], [], marker="o", ls="-", lw=S.ERR_LW + 0.1, ms=4.2, color=S.CATFORGE, mfc=S.tint(S.CATFORGE, 0.28),
                mec=S.CATFORGE, mew=0.7),
         Patch(facecolor=S.WORSE_SHADE, edgecolor=S.GREY_LIGHT, lw=0.4)]
fig.legend(key_b, ["Loss (95% interval above 1)", "Before repair", "Tie (interval includes 1)", "CatFORGE worse (> 1)"],
           ncol=2, loc="upper left", bbox_to_anchor=fig_xy(B_LEFT - 1.2, ROW_KEY), bbox_transform=fig.transFigure,
           handletextpad=0.3, columnspacing=1.2, handlelength=1.6, borderaxespad=0.0, borderpad=0.0)

letters = S.panel_letters(fig, [[(axa, "a"), (axb, "b")]], top_mm=ROW_HEAD - 0.35)
letters[1].set_x((B_LEFT - 5.5) / S.WIDTH_MM)
FIGNAME = "Fig3_benchmark_heldout"


# ---------------------------------------------------------------- verification against the v1 figure
def verify():
    v1 = V1["artists"]
    rows_v1 = list(range(len(ORDER)))[::-1]          # v1 row y-coordinates (first target on top)
    ymap = {ry: ORDER[i] for i, ry in enumerate(rows_v1)}
    XA = 1.6
    n, clipped_v1 = 0, []
    # a: one scatter per (row, model) in v1, offset (min(rel, 0.97 XA), row)
    for s in v1[0]["scatter"]:
        (x1, y1), = s["xy"]
        ds, t = ymap[int(round(y1))]
        model = next(k for k, v in NAME.items() if v == s["label"])
        rel = A_VALUES[(ds, t)][model]
        if np.isclose(x1, XA * 0.97) and rel > XA * 0.97:
            clipped_v1.append((ds, t, model, round(rel, 3)))
        else:
            S.assert_close(rel, x1, f"a {ds} {t} {model}")
        # the v2 artist carries the same value
        s2 = next(c for c in S.extract_artists(fig)[0]["scatter"] if c["gid"] == f"a:{ds}:{t}:{model}")
        S.assert_close(s2["xy"][0, 0], rel, f"a v2 artist {ds} {t} {model}")
        n += 1
    # b: v1 error-bar markers (ratio), error segments (lo, hi) and before-repair diamonds, keyed by their v1 y position
    v2art = S.extract_artists(fig)
    m1 = {}
    for l in v1[1]["lines"]:
        if l["data"] and len(l["x"]) == 1:
            m1.setdefault(round(float(l["y"][0]), 6), []).append(float(l["x"][0]))
    seg1 = {}
    for sg in v1[1]["segments"]:
        for s in sg["segs"]:
            seg1[round(float(s[0, 1]), 6)] = (float(s[0, 0]), float(s[1, 0]))
    br1 = {round(float(s["xy"][0, 1]), 6): float(s["xy"][0, 0]) for s in v1[1]["scatter"]}
    nb = 0
    for i, (ds, t) in enumerate(ORDER):
        for rg, sh in (("50", -0.24), ("200", 0.0), ("full", 0.24)):
            yk = round(rows_v1[i] + sh, 6)
            r2, lo2, hi2, verdict, r1 = B_VALUES[(ds, t, rg)]
            assert any(np.isclose(x, r2, atol=1e-12) for x in m1[yk]), (ds, t, rg, "ratio")
            S.assert_close(seg1[yk], (lo2, hi2), f"b CI {ds} {t} {rg}")
            S.assert_close(br1[yk], min(r1, 3.6 * 0.97), f"b before repair {ds} {t} {rg}")
            # v2 artists
            k = [r for r, _ in REG].index(rg)
            line2 = next(l for l in v2art[1 + k]["lines"] if l["gid"] == f"b:{ds}:{t}:{rg}:v2")
            S.assert_close(line2["x"], [r2], f"b v2 ratio {ds} {t} {rg}")
            seg2 = next(sg for sg in v2art[1 + k]["segments"] if sg["gid"] == f"b:{ds}:{t}:{rg}:ci")["segs"][0]
            S.assert_close(seg2[:, 0], [lo2, hi2], f"b v2 CI {ds} {t} {rg}")
            sc2 = next(c for c in v2art[1 + k]["scatter"] if c["gid"] == f"b:{ds}:{t}:{rg}:v1")
            S.assert_close(sc2["xy"][0, 0], min(r1, CLIP), f"b v2 before repair {ds} {t} {rg}")
            nb += 1
    print(f"verification Fig 3: {n} model points in a and {nb} x 3 values in b match the v1 figure.")
    if clipped_v1:
        print("   note: v1 drew these points at its clip position x = 1.552 although their value is below the 1.6 axis "
              "limit; v2 draws the exact values:", clipped_v1)
    clipped_v2 = [(k, round(v[4], 3)) for k, v in B_VALUES.items() if v[4] > CLIP]
    print("   v2 before-repair ratios beyond the panel-b axis (drawn as arrows at the edge, value printed):", clipped_v2)


if __name__ == "__main__":
    S.save(fig, FIGNAME)
    verify()
