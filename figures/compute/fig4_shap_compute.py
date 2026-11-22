"""Figure 4: what the platform's explanations recover on the literature DRM compilation.

Method = CatFORGE's SHAP step (catforge/generalized_predictor.py, section 5A): numeric inputs standardized, one XGBRegressor
(n_estimators=100, max_depth=3, random_state=42) per target, TreeSHAP on the training records. TreeSHAP values are computed
with XGBoost's built-in implementation (pred_contribs=True), the same algorithm the shap library's TreeExplainer applies.
Stability: the proxy is refitted on five publication-grouped 80% subsets; bars show the full-data value, whiskers the range."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from figstyle import plt, DOUBLE, MM, GRID, MUTED, INK, panel, save

ROOT = Path(__file__).resolve().parents[2]          # package root
F = ROOT / "data/drm_literature.csv"                 # DRM literature compilation (see data/README.md)
INPUTS = ["Ratio of CH4 in Feed", "Reaction Temperature", "Ni Loading", "Reaction Time", "Pore Size", "Pore Volume", "Surface Area",
          "H2-TPR Peak Temperature", "Ni Particle Size", "Modifier Electronegativity", "GHSV"]
TARGETS = ["CH4 Conversion", "CO2 Conversion", "Syngas_Ratio"]
df = pd.read_csv(F).dropna(subset=INPUTS + TARGETS).reset_index(drop=True)
groups = df["Paper no."].astype(str).to_numpy()
X = StandardScaler().fit_transform(df[INPUTS])
NICE = {"Ratio of CH4 in Feed": "CH$_4$ fraction in feed", "Reaction Temperature": "Reaction temperature", "Ni Loading": "Ni loading",
        "Reaction Time": "Time on stream", "Pore Size": "Pore size", "Pore Volume": "Pore volume", "Surface Area": "Surface area",
        "H2-TPR Peak Temperature": "H$_2$-TPR peak temperature", "Ni Particle Size": "Ni particle size",
        "Modifier Electronegativity": "Modifier electronegativity", "GHSV": "GHSV"}
TNICE = {"CH4 Conversion": "CH$_4$ conversion", "CO2 Conversion": "CO$_2$ conversion", "Syngas_Ratio": "H$_2$/CO ratio"}


def shap_values(Xm, y):
    m = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42).fit(Xm, y)
    contrib = m.get_booster().predict(xgb.DMatrix(Xm), pred_contribs=True)
    return contrib[:, :-1]          # last column is the bias term


full, folds = {}, {t: [] for t in TARGETS}
ug = np.unique(groups); rng = np.random.default_rng(7); perm = rng.permutation(ug); parts = np.array_split(perm, 5)
for t in TARGETS:
    y = df[t].to_numpy()
    sv = shap_values(X, y); full[t] = sv
    for p in parts:
        keep = ~np.isin(groups, p)
        folds[t].append(np.abs(shap_values(X[keep], y[keep])).mean(axis=0))

imp = pd.DataFrame({t: np.abs(full[t]).mean(axis=0) for t in TARGETS}, index=INPUTS)
share = imp / imp.sum() * 100
order = share["CH4 Conversion"].sort_values().index.tolist()

fig = plt.figure(figsize=(DOUBLE, 118 * MM))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.62, wspace=0.12)
gsb = gs[1, :].subgridspec(1, 3, wspace=0.85, width_ratios=[1, 1, 0.9])
cols = {"CH4 Conversion": "#0072B2", "CO2 Conversion": "#009E73", "Syngas_Ratio": "#CC79A7"}
for j, t in enumerate(TARGETS):
    ax = fig.add_subplot(gs[0, j])
    lo = np.min(np.vstack(folds[t]), axis=0) / imp[t].sum() * 100
    hi = np.max(np.vstack(folds[t]), axis=0) / imp[t].sum() * 100
    lo, hi = pd.Series(lo, index=INPUTS), pd.Series(hi, index=INPUTS)
    yy = np.arange(len(order))
    ax.barh(yy, share.loc[order, t], color=cols[t], height=0.66, alpha=0.9)
    ax.errorbar(share.loc[order, t], yy, xerr=[np.clip(share.loc[order, t] - lo[order], 0, None), np.clip(hi[order] - share.loc[order, t], 0, None)],
                fmt="none", ecolor="#333", lw=0.6, capsize=1.2)
    ax.set_yticks(yy); ax.set_yticklabels([NICE[f] for f in order] if j == 0 else [])
    ax.set_xlabel("Share of mean |SHAP| (%)"); ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title(TNICE[t], loc="left", fontsize=7.2, pad=3)
    if j == 0:
        panel(ax, "a", None, x=-0.62, y=1.04)
# b, c: dependence for CH4 conversion
ti = "CH4 Conversion"
for k, (feat, letter, cfeat) in enumerate((("Reaction Temperature", "b", "Ni Loading"), ("GHSV", "c", "Reaction Temperature"))):
    ax = fig.add_subplot(gsb[0, k])
    x = df[feat].to_numpy(); s = full[ti][:, INPUTS.index(feat)]; c = df[cfeat].to_numpy()
    sc = ax.scatter(x, s, c=c, s=4, cmap="viridis", lw=0, alpha=0.8, rasterized=True)
    if feat == "GHSV":
        ax.set_xscale("log")
    ax.axhline(0, color="#444", lw=0.6)
    ax.set_xlabel(("Reaction temperature (\u00b0C)" if feat == "Reaction Temperature" else "GHSV (mL g$^{-1}$ h$^{-1}$, as reported)"))
    ax.set_ylabel("SHAP value for CH$_4$ conversion\n(percentage points)")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label("Ni loading (wt%)" if cfeat == "Ni Loading" else "Reaction temperature (\u00b0C)", fontsize=6); cb.ax.tick_params(labelsize=5.8)
    panel(ax, letter, None, x=-0.3, y=1.04)
    rho = spearmanr(x, s).statistic
    ax.text(0.03, 0.96, f"Spearman \u03c1 = {rho:.2f}", transform=ax.transAxes, va="top", fontsize=6, color=MUTED)
# d: model accuracy context (in-sample vs group-held-out for the proxy)
ax = fig.add_subplot(gsb[0, 2])
r2_in, r2_out = [], []
from sklearn.metrics import r2_score
for t in TARGETS:
    y = df[t].to_numpy()
    m = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X, y); r2_in.append(r2_score(y, m.predict(X)))
    pred = np.zeros_like(y, dtype=float)
    for p in parts:
        te = np.isin(groups, p)
        mm = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X[~te], y[~te]); pred[te] = mm.predict(X[te])
    r2_out.append(r2_score(y, pred))
xx = np.arange(3)
ax.bar(xx - 0.18, r2_in, 0.34, color="#BBBBBB", label="fitted records (in-sample)")
ax.bar(xx + 0.18, r2_out, 0.34, color="#444444", label="held-out publications")
ax.set_xticks(xx); ax.set_xticklabels(["CH$_4$\nconv.", "CO$_2$\nconv.", "H$_2$/CO\nratio"], fontsize=6)
ax.set_ylabel("R$^2$ of the explanation model"); ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=5.8, loc="upper right")
panel(ax, "d", None, x=-0.25, y=1.04)
save(fig, "Fig4_interpretation_shap")

print("records", len(df), "publications", len(ug))
print("share of mean |SHAP| (%):"); print(share.round(1).sort_values("CH4 Conversion", ascending=False).to_string())
for t in TARGETS:
    ranks = [pd.Series(f, index=INPUTS).rank(ascending=False) for f in folds[t]]
    top = imp[t].idxmax(); print(t, "top feature", top, "fold ranks of top feature:", [int(r[top]) for r in ranks])
for feat in ("Reaction Temperature", "GHSV", "Ni Loading"):
    print(feat, "Spearman(feature, SHAP for CH4):", round(spearmanr(df[feat], full[ti][:, INPUTS.index(feat)]).statistic, 2))
print("proxy R2 in-sample:", np.round(r2_in, 3), "group-held-out:", np.round(r2_out, 3))
print("temperature range:", df["Reaction Temperature"].min(), df["Reaction Temperature"].max(), "| GHSV range:", df.GHSV.min(), df.GHSV.max())
