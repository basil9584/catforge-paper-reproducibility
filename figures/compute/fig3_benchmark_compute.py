"""Figure 3: repaired CatFORGE predictor vs ten tuned baselines on locked test groups (one-shot evaluation).
Source: results/predictor_benchmark/final_summary.csv, final_catforge_vs_baselines.csv (docs/REPAIR_REPORT.md).
Package: set CATFORGE_RESULTS_DIR (e.g. output/benchmarks) to use re-run results instead."""
import os
import sys
from pathlib import Path
import pandas as pd
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).parent))
from figstyle import plt, DOUBLE, MM, GRID, MUTED, panel, save

PKG = Path(__file__).resolve().parents[2]
R = Path(os.environ.get("CATFORGE_RESULTS_DIR", PKG / "results")).resolve() / "predictor_benchmark"
summ = pd.read_csv(R / "final_summary.csv", dtype={"regime": str})
comp = pd.read_csv(R / "final_catforge_vs_baselines.csv", dtype={"regime": str})
LABEL = {"drm": "DRM", "wgs": "WGS", "ocm": "OCM"}
TARGET = {"CH4 Conversion": "CH$_4$ conversion", "CO2 Conversion": "CO$_2$ conversion", "Syngas_Ratio": "H$_2$/CO ratio",
          "CO Conversion": "CO conversion", "X(CH4), %": "CH$_4$ conversion", "S(C2), %": "C$_2$ selectivity", "Y(C2),%": "C$_2$ yield"}
ORDER = [("drm", "CH4 Conversion"), ("drm", "CO2 Conversion"), ("drm", "Syngas_Ratio"), ("wgs", "CO Conversion"),
         ("ocm", "X(CH4), %"), ("ocm", "S(C2), %"), ("ocm", "Y(C2),%")]
NAME = {"catforge_v2": "CatFORGE", "catforge_v1": "CatFORGE before repair", "lgbm": "LightGBM", "xgb": "XGBoost",
        "xgb_logit": "XGBoost + logit", "rf": "Random forest", "et": "Extra trees", "ridge": "Ridge", "knn": "k-NN", "svr": "SVR",
        "mlp": "MLP", "mean": "Training mean"}
COL = {"catforge_v2": "#D55E00", "catforge_v1": "#D55E00", "lgbm": "#009E73", "xgb": "#0072B2", "xgb_logit": "#56B4E9", "rf": "#CC79A7",
       "et": "#E69F00", "ridge": "#8C8C8C", "knn": "#B0B0B0", "svr": "#5F5F5F", "mlp": "#CFCFCF", "mean": "#000000"}
VERDICT = {"WIN": "#009E73", "TIE": "#7F7F7F", "LOSS": "#D55E00"}
XA, XB = 1.6, 3.6

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 92 * MM), sharey=True, gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.08})
rows = list(range(len(ORDER)))[::-1]
full = summ[summ.regime == "full"]
for y, (ds, t) in zip(rows, ORDER):
    cell = full[(full.dataset == ds) & (full.target == t)]
    best = cell.rmse.min()
    for _, r in cell.sort_values("model").iterrows():
        rel = r.rmse / best
        v2, v1 = r.model == "catforge_v2", r.model == "catforge_v1"
        ax1.scatter(min(rel, XA * 0.97), y, s=34 if (v1 or v2) else 13, marker=">" if rel > XA else "o",
                    facecolor="white" if v1 else COL[r.model], edgecolor="black" if v2 else (COL[r.model] if v1 else "none"),
                    linewidth=0.8 if (v1 or v2) else 0, zorder=4 if v2 else (3 if v1 else 2), label=NAME[r.model])
ax1.set_yticks(rows); ax1.set_yticklabels([f"{LABEL[d]}: {TARGET[t]}" for d, t in ORDER])
ax1.set_xscale("log"); ax1.set_xlim(0.97, XA); ax1.set_xticks([1, 1.1, 1.2, 1.4, 1.6]); ax1.set_xticklabels(["1", "1.1", "1.2", "1.4", "1.6"]); ax1.minorticks_off()
ax1.axvline(1, color="#BBBBBB", lw=0.7, zorder=0); ax1.grid(axis="x", color=GRID, lw=0.5); ax1.set_axisbelow(True)
ax1.set_xlabel("RMSE relative to the best model (1 = best)")
panel(ax1, "a", "Locked test groups, all development records for training", x=-0.32)
shift = {"50": -0.24, "200": 0.0, "full": 0.24}
mark = {"50": "v", "200": "s", "full": "o"}
for y, (ds, t) in zip(rows, ORDER):
    for rg in ("50", "200", "full"):
        for focus in ("catforge_v1", "catforge_v2"):
            c = comp[(comp.dataset == ds) & (comp.target == t) & (comp.regime == rg) & (comp.baseline == "best_other") & (comp.focus == focus)].iloc[0]
            if focus == "catforge_v1":
                ax2.scatter(min(c.ratio, XB * 0.97), y + shift[rg], marker="D", s=9, facecolor="white", edgecolor="#9A9A9A", lw=0.6, zorder=2)
            else:
                ax2.errorbar(c.ratio, y + shift[rg], xerr=[[c.ratio - c.ratio_lo], [c.ratio_hi - c.ratio]], fmt=mark[rg],
                             color=VERDICT[c.verdict], ms=3.2, lw=0.8, capsize=1.2, zorder=3)
ax2.axvline(1, color="black", lw=0.7)
ax2.set_xscale("log"); ax2.set_xlim(0.8, XB); ax2.set_xticks([0.8, 1, 1.5, 2, 3]); ax2.set_xticklabels(["0.8", "1", "1.5", "2", "3"]); ax2.minorticks_off()
ax2.grid(axis="x", color=GRID, lw=0.5); ax2.set_axisbelow(True)
ax2.set_xlabel("CatFORGE RMSE / best other model's RMSE  (< 1: CatFORGE better)")
panel(ax2, "b", "CatFORGE vs best other model", x=-0.04)


leg = [Line2D([], [], marker=mark[r], ls="", color="#444", ms=3.2, label=("all" if r == "full" else r) + " training records") for r in ("50", "200", "full")]
leg += [Line2D([], [], marker="s", ls="", color=VERDICT[v], ms=3.2, label={"WIN": "win", "TIE": "tie", "LOSS": "loss"}[v]) for v in ("TIE", "LOSS")]
leg += [Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor="#9A9A9A", ms=3, label="before repair (ratio only)")]
ax2.legend(handles=leg, frameon=False, loc="upper right", fontsize=5.6, ncol=1, handletextpad=0.3, borderaxespad=0.2)
h, l = ax1.get_legend_handles_labels()
seen = {}
for hh, ll in zip(h, l):
    seen.setdefault(ll, hh)
order = ["CatFORGE", "CatFORGE before repair", "LightGBM", "XGBoost", "XGBoost + logit", "Extra trees", "Random forest", "SVR", "Ridge", "k-NN", "MLP", "Training mean"]
fig.legend([seen[k] for k in order], order, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=6, frameon=False, fontsize=6.0,
           handletextpad=0.2, columnspacing=1.0)
save(fig, "Fig3_benchmark_heldout")
