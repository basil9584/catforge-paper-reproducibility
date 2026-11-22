"""Figure 2: synthetic records reproduce the data but do not improve prediction for unseen studies.
Source: results/augmentation_benchmark/benchmark_summary.csv and benchmark_fidelity_vs_utility.csv (augmentation
benchmark, 2 Nov 2026; group-held-out splits, random-forest and XGBoost predictors, MAE relative to real-only training,
bootstrap 95% CIs). Package: set CATFORGE_RESULTS_DIR (e.g. output/benchmarks) to use re-run results instead."""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from figstyle import plt, DOUBLE, MM, DS_COL, DS_NAME, MUTED, GRID, panel, save

PKG = Path(__file__).resolve().parents[2]
R = Path(os.environ.get("CATFORGE_RESULTS_DIR", PKG / "results")).resolve() / "augmentation_benchmark"
s = pd.read_csv(R / "benchmark_summary.csv")
main = s[s.grid == "main"]
abl = s[s.grid == "ablation"]

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 58 * MM), gridspec_kw={"wspace": 0.42})
# a: CatFORGE augmentation (200 synthetic rows) vs real only, random and few-study sampling
ax = axes[0]
for ds in ("drm", "wgs", "ocm"):
    for samp, ls, mk in (("random", "-", "o"), ("clustered", "--", "s")):
        d = main[(main.dataset == ds) & (main.sampling == samp) & (main.arm == "catforge_aug")].sort_values("size")
        ax.plot(d["size"], d.rel_delta_pct, ls=ls, marker=mk, ms=2.6, color=DS_COL[ds], lw=1.0, mfc="white" if samp == "clustered" else DS_COL[ds])
        if samp == "random":
            ax.fill_between(d["size"], d.ci_low, d.ci_high, color=DS_COL[ds], alpha=0.12, lw=0)
ax.axhline(0, color="#444", lw=0.6)
ax.set_xscale("log"); ax.set_xticks([25, 50, 100, 200, 400]); ax.set_xticklabels(["25", "50", "100", "200", "400"]); ax.minorticks_off()
ax.set_xlabel("Real training records"); ax.set_ylabel("Change in held-out MAE vs real only (%)")
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
panel(ax, "a", "Real + 200 synthetic records"); ax.set_ylim(-6, 15.5)
ax.text(0.98, 0.97, "solid: records from many studies\ndashed: records from a few studies", transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=MUTED)

# b: synthetic records only
ax = axes[1]
for ds in ("drm", "wgs", "ocm"):
    d = main[(main.dataset == ds) & (main.sampling == "random") & (main.arm == "catforge_synth_only")].sort_values("size")
    ax.plot(d["size"], d.rel_delta_pct, marker="o", ms=2.6, color=DS_COL[ds], lw=1.0, label=DS_NAME[ds])
    ax.fill_between(d["size"], d.ci_low, d.ci_high, color=DS_COL[ds], alpha=0.12, lw=0)
ax.axhline(0, color="#444", lw=0.6)
ax.set_xscale("log"); ax.set_xticks([25, 50, 100, 200, 400]); ax.set_xticklabels(["25", "50", "100", "200", "400"]); ax.minorticks_off()
ax.set_xlabel("Real records used to train the generator"); ax.set_ylabel("Change in held-out MAE vs real only (%)")
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right", handlelength=1.4)
panel(ax, "b", "Synthetic records only")

# c: fidelity (quality score) vs usefulness, generator ablation at 50 and 200 records
ax = axes[2]
gen_mark = {"gc_aug_r1": ("Gaussian copula", "D"), "ctgan_aug_r1": ("CTGAN", "^"), "tvae_aug_r1": ("TVAE", "v")}
for ds in ("drm", "wgs", "ocm"):
    for arm in abl.arm.unique():
        if arm == "real_only":
            continue
        d = abl[(abl.dataset == ds) & (abl.arm == arm)]
        if arm.startswith("catforge"):
            lab, mk = "CatFORGE (r = 0.5\u20134)", "o"
        else:
            lab, mk = gen_mark[arm]
        ax.scatter(d.quality * 100, d.rel_delta_pct, marker=mk, s=11, color=DS_COL[ds], edgecolor="white", lw=0.3, zorder=3)
ax.axhline(0, color="#444", lw=0.6)
ax.set_xlabel("Synthetic-data quality score (%)"); ax.set_ylabel("Change in held-out MAE vs real only (%)")
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker=m, ls="", color="#666", ms=3.2, label=l) for l, m in
     (("CatFORGE (r = 0.5\u20134)", "o"), ("Gaussian copula", "D"), ("CTGAN", "^"), ("TVAE", "v"))]
ax.legend(handles=h, frameon=False, loc="center left", fontsize=5.8, handletextpad=0.3, ncol=1, bbox_to_anchor=(0.0, 0.55)); ax.set_ylim(-1.9, 12.0)
panel(ax, "c", "Quality score vs usefulness")
save(fig, "Fig2_synthetic_fidelity_vs_usefulness")

# ---- numbers for the text ----
def row(ds, samp, size, arm):
    d = main[(main.dataset == ds) & (main.sampling == samp) & (main["size"] == size) & (main.arm == arm)]
    return d.iloc[0] if len(d) else None
ca = main[main.arm == "catforge_aug"]
print("catforge_aug rel_delta range (all main cells):", round(ca.rel_delta_pct.min(), 1), round(ca.rel_delta_pct.max(), 1))
print("catforge_aug cells within +-2%:", int((ca.rel_delta_pct.abs() <= 2).sum()), "of", len(ca))
print("catforge_aug cells whose CI excludes 0 (improve / worsen):", int((ca.ci_high < 0).sum()), int((ca.ci_low > 0).sum()))
for ds, samp, size in (("drm", "clustered", 25), ("drm", "clustered", 50), ("wgs", "random", 25), ("wgs", "random", 50)):
    r = row(ds, samp, size, "catforge_aug"); print(ds, samp, size, "catforge_aug", round(r.rel_delta_pct, 1), [round(r.ci_low, 1), round(r.ci_high, 1)])
so = main[main.arm == "catforge_synth_only"]
print("synth_only worse in", int((so.rel_delta_pct > 0).sum()), "of", len(so), "; max", round(so.rel_delta_pct.max(), 1), "; min", round(so.rel_delta_pct.min(), 1))
print(so.loc[so.rel_delta_pct.idxmin(), ["dataset", "sampling", "size", "rel_delta_pct", "ci_low", "ci_high"]].to_dict())
gc = main[main.arm == "gc_aug"].set_index(["dataset", "sampling", "size"]).rel_delta_pct
cf = ca.set_index(["dataset", "sampling", "size"]).rel_delta_pct
print("mean |catforge - gaussian copula| (pct points):", round((cf - gc).abs().mean(), 2))
print("ablation worst (max) rel_delta by generator:", abl[abl.arm != "real_only"].groupby("arm").rel_delta_pct.max().round(1).to_dict())
print("ablation quality range by generator:", abl[abl.arm != "real_only"].groupby("arm").quality.agg(["min", "max"]).round(3).to_dict("index"))
fu = pd.read_csv(R / "benchmark_fidelity_vs_utility.csv")
print(fu.round(3).to_string())
