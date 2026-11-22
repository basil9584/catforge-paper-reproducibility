"""Figure 6 of the working scripts = Figure 5 of the manuscript: illustration of CatFORGE's cost-aware ranking on the
DRM literature compilation.

Faithful to the platform's optimization module (catforge/optimization_catcost.py; platform path
backend/models/optimization/optimization_catcost.py):
  surrogate   = step2 StandardScaler on inputs + step6 bootstrap ensemble of 10 GradientBoostingRegressor
                (n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, bootstrap seed 42+i, random_state 42+i);
                prediction = ensemble mean, uncertainty = ensemble standard deviation (as stored in '<target>_uncertainty')
  cost        = calculate_materials_cost (quantity x unit price with low/baseline/high values), per kg catalyst
  ranking     = calculate_topsis_score (criteria: predicted CH4 and CO2 conversion maximized, cost minimized);
                Pareto front = non-dominated candidates (step7 logic) on predicted CH4 conversion vs cost
Candidates: Ni loading 2-25 wt% on a representative Ni/Al2O3 profile (medians of Al2O3-supported records tested at 700 C),
fixed test conditions 700 C, CH4 feed fraction 0.5, GHSV 30,000 mL g-1 h-1, 10 h on stream, no modifier.
Prices: raw-material value of Ni (metal price) and Al2O3 (alumina price) from USGS Mineral Commodity Summaries 2025,
passed as a JSON argument (US$/kg; figures/fig5_cost.py passes the values used for the manuscript).
ILLUSTRATION ONLY: materials cost, not manufacturing cost."""
import json
import sys
import types
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from figstyle import plt, DOUBLE, MM, GRID, MUTED, INK, panel, save

ROOT = Path(__file__).resolve().parents[2]          # package root
# the module imports optuna only for candidate sampling, which this illustration does not use
optuna = types.ModuleType("optuna"); optuna.logging = types.SimpleNamespace(set_verbosity=lambda *a: None, ERROR=40)
sys.modules["optuna"] = optuna
spec = importlib.util.spec_from_file_location("optcat", ROOT / "catforge/optimization_catcost.py")
opt = importlib.util.module_from_spec(spec); spec.loader.exec_module(opt)

PRICES = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
if PRICES is None:
    raise SystemExit("usage: fig6_cost.py '<json prices: ni_low, ni_base, ni_high, al_low, al_base, al_high (US$/kg)>'")

F = ROOT / "data/drm_literature.csv"                 # DRM literature compilation (see data/README.md)
INPUTS = ["Ratio of CH4 in Feed", "Reaction Temperature", "Ni Loading", "Reaction Time", "Pore Size", "Pore Volume", "Surface Area",
          "H2-TPR Peak Temperature", "Ni Particle Size", "Modifier Electronegativity", "GHSV"]
TARGETS = ["CH4 Conversion", "CO2 Conversion"]
raw = pd.read_csv(F)
df = raw.dropna(subset=INPUTS + TARGETS + ["Syngas_Ratio"]).reset_index(drop=True)
is_al = df.Catalyst.astype(str).str.contains("Al2O3", case=False, regex=False)
ref = df[is_al & (df["Reaction Temperature"] == 700)]
profile = {c: float(ref[c].median()) for c in ["Pore Size", "Pore Volume", "Surface Area", "H2-TPR Peak Temperature", "Ni Particle Size"]}
profile.update({"Ratio of CH4 in Feed": 0.5, "Reaction Temperature": 700.0, "Reaction Time": 10.0, "GHSV": 30000.0, "Modifier Electronegativity": 0.0})

scaler = StandardScaler().fit(df[INPUTS])
X = scaler.transform(df[INPUTS])
ens = {}
for t in TARGETS:
    y = df[t].to_numpy(); models = []
    for i in range(10):
        idx = np.random.default_rng(42 + i).choice(len(X), size=len(X), replace=True)
        models.append(GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42 + i).fit(X[idx], y[idx]))
    ens[t] = models

# held-out error of this surrogate recipe for CH4 conversion: 5 folds that each withhold a fifth of the publications
grp = df["Paper no."].astype(str).to_numpy(); ug = np.random.default_rng(7).permutation(np.unique(grp)); parts = np.array_split(ug, 5)
yv = df["CH4 Conversion"].to_numpy(); oof = np.zeros(len(yv))
for p in parts:
    te = np.isin(grp, p); sc = StandardScaler().fit(df.loc[~te, INPUTS]); Xtr, Xte = sc.transform(df.loc[~te, INPUTS]), sc.transform(df.loc[te, INPUTS])
    preds = []
    for i in range(10):
        idx = np.random.default_rng(42 + i).choice(len(Xtr), size=len(Xtr), replace=True)
        preds.append(GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42 + i).fit(Xtr[idx], yv[~te][idx]).predict(Xte))
    oof[te] = np.mean(preds, axis=0)
HELDOUT_RMSE = float(np.sqrt(np.mean((oof - yv) ** 2)))

loadings = np.round(np.arange(2.0, 25.01, 0.5), 2)
cand = pd.DataFrame([{**profile, "Ni Loading": w} for w in loadings])[INPUTS]
Xc = scaler.transform(cand)
for t in TARGETS:
    P = np.array([m.predict(Xc) for m in ens[t]])
    cand[f"{t}_pred"], cand[f"{t}_uncertainty"] = P.mean(axis=0), P.std(axis=0)


def unit_cost(w, key):
    mats = [{"name": "Ni (metal value)", "quantity": w / 100,
             "unit_price": {"baseline": PRICES["ni_base"], "low": PRICES["ni_low"], "high": PRICES["ni_high"]}},
            {"name": "Al2O3 support", "quantity": 1 - w / 100,
             "unit_price": {"baseline": PRICES["al_base"], "low": PRICES["al_low"], "high": PRICES["al_high"]}}]
    return opt.calculate_materials_cost(mats, scale=1)["cost_per_kg"][key]


for key in ("baseline", "low", "high"):
    cand[f"cost_{key}"] = [unit_cost(w, key) for w in cand["Ni Loading"]]


def pareto(df, perf, cost):
    pts = df[[perf, cost]].to_numpy(); keep = np.ones(len(pts), bool)
    for i in range(len(pts)):
        dom = (pts[:, 0] >= pts[i, 0]) & (pts[:, 1] <= pts[i, 1]) & ((pts[:, 0] > pts[i, 0]) | (pts[:, 1] < pts[i, 1]))
        keep[i] = not dom.any()
    return keep


cand["pareto"] = pareto(cand, "CH4 Conversion_pred", "cost_baseline")
wgrid = np.round(np.arange(0.0, 0.81, 0.05), 2)
choice = {}
for key in ("low", "baseline", "high"):
    sel = []
    for wc in wgrid:
        wp = (1 - wc) / 2
        s = opt.calculate_topsis_score(cand, ["CH4 Conversion_pred", "CO2 Conversion_pred", f"cost_{key}"],
                                       {"CH4 Conversion_pred": wp, "CO2 Conversion_pred": wp, f"cost_{key}": wc},
                                       {"CH4 Conversion_pred": "maximize", "CO2 Conversion_pred": "maximize", f"cost_{key}": "minimize"})
        sel.append(float(cand["Ni Loading"].iloc[int(np.argmax(s))]))
    choice[key] = sel

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 60 * MM), gridspec_kw={"wspace": 0.62, "width_ratios": [1.1, 1, 1]})
ax = axes[0]
meas = df[is_al & (df["Reaction Temperature"] == 700) & (df["Ratio of CH4 in Feed"] == 0.5)]
ax.scatter(meas["Ni Loading"], meas["CH4 Conversion"], s=6, color="#BBBBBB", lw=0, label="measured records (700 \u00b0C, 1:1)", zorder=1)
for t, col, lab in (("CH4 Conversion", "#0072B2", "CH$_4$"), ("CO2 Conversion", "#D55E00", "CO$_2$")):
    m, s = cand[f"{t}_pred"], cand[f"{t}_uncertainty"]
    ax.plot(cand["Ni Loading"], m, color=col, lw=1.2, label=f"{lab} predicted", zorder=3)
    ax.fill_between(cand["Ni Loading"], m - s, m + s, color=col, alpha=0.18, lw=0, zorder=2)
ax.set_xlabel("Ni loading (wt%)"); ax.set_ylabel("Conversion (%)"); ax.set_ylim(0, 100)
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax2 = ax.twinx()
ax2.plot(cand["Ni Loading"], cand["cost_baseline"], color="#555", lw=0.9, ls="--")
ax2.fill_between(cand["Ni Loading"], cand["cost_low"], cand["cost_high"], color="#777", alpha=0.12, lw=0)
ax2.set_ylabel("Raw-material cost (US\\$ kg$^{-1}$ catalyst)", color="#444"); ax2.spines["right"].set_visible(True); ax2.tick_params(axis="y", colors="#444")
ax.legend(frameon=False, fontsize=5.6, loc="upper left", handlelength=1.4)
panel(ax, "a", "Predicted conversion and cost", x=-0.28)

ax = axes[1]
ax.errorbar(cand["cost_baseline"], cand["CH4 Conversion_pred"], yerr=cand["CH4 Conversion_uncertainty"], fmt="o", ms=2.4, color="#9AB7CF",
            ecolor="#CCD9E4", elinewidth=0.6, lw=0, zorder=2, label="candidates")
p = cand[cand.pareto].sort_values("cost_baseline")
ax.plot(p["cost_baseline"], p["CH4 Conversion_pred"], "-o", ms=3, color="#0072B2", lw=1.0, zorder=3, label="Pareto front")
for wc, mk, lab in ((0.0, "^", "performance only"), (0.05, "s", "cost weight 0.05"), (0.2, "D", "cost weight 0.2")):
    wsel = choice["baseline"][list(wgrid).index(wc)]
    r = cand[cand["Ni Loading"] == wsel].iloc[0]
    ax.scatter(r["cost_baseline"], r["CH4 Conversion_pred"], marker=mk, s=26, facecolor="#E69F00", edgecolor="black", lw=0.5, zorder=4,
               label=f"TOPSIS: {lab}")
ax.set_xlabel("Raw-material cost (US\\$ kg$^{-1}$ catalyst)"); ax.set_ylabel("Predicted CH$_4$ conversion (%)"); ax.set_ylim(54, 76)
ax.text(0.97, 0.97, f"surrogate RMSE on held-out\npublications: {HELDOUT_RMSE:.0f} percentage points", transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color=MUTED)
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=5.4, loc="lower right", handletextpad=0.3, borderaxespad=0.2)
panel(ax, "b", "Performance\u2013cost trade-off", x=-0.27)

ax = axes[2]
for key, ls, col in (("low", ":", "#777"), ("baseline", "-", "#E69F00"), ("high", "--", "#777")):
    ax.step(wgrid, choice[key], where="mid", ls=ls, color=col, lw=1.2, label={"low": "low prices", "baseline": "baseline prices", "high": "high prices"}[key])
ax.set_xlabel("Weight on cost in TOPSIS"); ax.set_ylabel("Ni loading of top-ranked candidate (wt%)")
ax.set_ylim(0, 26); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=5.8, loc="upper right")
panel(ax, "c", "Top-ranked candidate vs cost weight", x=-0.25)
save(fig, "Fig6_cost_aware_ranking")

out = ROOT / "output/compute_drafts/fig5_cost_numbers.json"      # stored reference: results/reference_numbers/
out.parent.mkdir(parents=True, exist_ok=True)
summary = {"heldout_rmse_ch4": HELDOUT_RMSE, "prices": PRICES, "profile": profile, "n_measured_context": int(len(meas)),
           "cand": cand[["Ni Loading", "CH4 Conversion_pred", "CH4 Conversion_uncertainty", "CO2 Conversion_pred", "CO2 Conversion_uncertainty",
                         "cost_low", "cost_baseline", "cost_high", "pareto"]].round(4).to_dict("list"),
           "topsis_choice": {k: dict(zip([float(w) for w in wgrid], v)) for k, v in choice.items()}}
out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("surrogate held-out RMSE CH4:", round(HELDOUT_RMSE, 2)); print("profile:", {k: round(v, 2) for k, v in profile.items()})
print("CH4 pred at 2/5/10/15/20/25 wt%:", [round(float(cand.loc[cand['Ni Loading'] == w, 'CH4 Conversion_pred'].iloc[0]), 1) for w in (2, 5, 10, 15, 20, 25)])
print("CH4 sd range:", round(cand['CH4 Conversion_uncertainty'].min(), 1), round(cand['CH4 Conversion_uncertainty'].max(), 1))
print("cost at 2/10/25 wt% (baseline):", [round(unit_cost(w, 'baseline'), 2) for w in (2, 10, 25)])
print("Pareto loadings:", cand.loc[cand.pareto, 'Ni Loading'].tolist())
print("TOPSIS choice vs cost weight (baseline):", dict(zip(wgrid.tolist(), choice['baseline'])))
print("measured context records:", len(meas), "papers:", meas['Paper no.'].nunique())
