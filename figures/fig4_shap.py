"""Figure 4 (v2 design): what the platform's explanations recover on the DRM literature compilation.

Data: identical to v1 (figures/compute/fig4_shap_compute.py: standardized inputs, one XGBRegressor per response,
TreeSHAP via pred_contribs, five publication-grouped refits, on data/drm_literature.csv). The v1 script runs in a
separate process with file writes disabled (style_v2.v1_dry_run); its arrays (`share`, `imp`, `folds`, `full`, `r2_in`,
`r2_out`, `df`) are plotted here unchanged, the drawn values are asserted equal to those in the v1 figure, and its printed
numbers are asserted equal to results/reference_numbers/fig4_numbers.txt.
Run from the package root: python figures/fig4_shap.py

Design change: v1 coloured panels b and c by Ni loading and by reaction temperature. Neither colour carried visible
structure (Spearman correlation of Ni loading with the SHAP residual within temperature levels 0.02; of temperature with
the SHAP residual within GHSV bins 0.15), so v2 uses a single colour with transparency (see README).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import style_v2 as S  # noqa: E402

V1 = S.v1_dry_run("fig4_shap_compute.py")
g = V1["globals"]
df, share, imp, folds, full, INPUTS, TARGETS, NICE = (g[k] for k in ("df", "share", "imp", "folds", "full", "INPUTS",
                                                                     "TARGETS", "NICE"))
r2_in, r2_out = g["r2_in"], g["r2_out"]
order = share["CH4 Conversion"].sort_values().index.tolist()       # as v1: largest share on top
RCOL = {"CH4 Conversion": S.RESPONSE["ch4"], "CO2 Conversion": S.RESPONSE["co2"], "Syngas_Ratio": S.RESPONSE["h2co"]}
TNICE = {"CH4 Conversion": "CH$_4$ conversion", "CO2 Conversion": "CO$_2$ conversion", "Syngas_Ratio": "H$_2$/CO ratio"}

# ---------------------------------------------------------------- layout (mm)
H = 106.5
A_TOP, A_H = 5.5, 38.0
A_LEFT, A_W, A_GAP = 34.0, 45.0, 5.0
B_TOP, B_H = 62.0, 34.0
fig = S.figure(H)
axa = [S.axes_mm(fig, A_LEFT + k * (A_W + A_GAP), A_TOP, A_W, A_H) for k in range(3)]
axb = S.axes_mm(fig, 15.5, B_TOP, 41.0, B_H)
axc = S.axes_mm(fig, 75.5, B_TOP, 41.0, B_H)
axd = S.axes_mm(fig, 139.5, B_TOP, 40.0, B_H)

# ---------------------------------------------------------------- a: share of mean |SHAP|, three responses
yy = np.arange(len(order))
A_DRAWN = {}
for k, t in enumerate(TARGETS):
    ax = axa[k]
    col = RCOL[t]
    lo = pd.Series(np.min(np.vstack(folds[t]), axis=0) / imp[t].sum() * 100, index=INPUTS)
    hi = pd.Series(np.max(np.vstack(folds[t]), axis=0) / imp[t].sum() * 100, index=INPUTS)
    v = share.loc[order, t].to_numpy()
    lo_e = np.clip(v - lo[order].to_numpy(), 0, None)
    hi_e = np.clip(hi[order].to_numpy() - v, 0, None)
    A_DRAWN[t] = (v, v - lo_e, v + hi_e)
    ax.barh(yy, v, height=0.62, color=S.tint(col, 0.55), lw=0, zorder=2)
    eb = ax.errorbar(v, yy, xerr=[lo_e, hi_e], fmt="none", ecolor=S.shade(col, 0.75), elinewidth=S.ERR_LW, capsize=0,
                     zorder=3)
    eb[2][0].set_gid(f"a:{t}:range")
    ax.set_xlim(0, 50)
    S.fixed_ticks(ax, "x", [0, 10, 20, 30, 40, 50])
    S.grid(ax, "x")
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_yticks(yy, labels=[NICE[f] for f in order] if k == 0 else [])
    ax.tick_params(axis="y", length=0, pad=3, labelsize=S.LABEL_PT)  # 7 pt so that subscripts stay near 5 pt
    ax.spines["left"].set_color(S.INK)
    S.header(ax, TNICE[t], pad=3)
axa[1].set_xlabel("Share of mean |SHAP| (%)")

# ---------------------------------------------------------------- b, c: dependence of the CH4-conversion SHAP value
ti = "CH4 Conversion"
SP = {}
for ax, feat in ((axb, "Reaction Temperature"), (axc, "GHSV")):
    x = df[feat].to_numpy()
    s = full[ti][:, INPUTS.index(feat)]
    sc = ax.scatter(x, s, s=3.0 ** 2, color=S.RESPONSE["ch4"], alpha=0.25, lw=0, zorder=3)
    sc.set_gid(f"dep:{feat}")
    S.ref_line(ax, 0, color=S.REF, lw=0.5)
    SP[feat] = spearmanr(x, s).statistic
    ax.set_ylabel("SHAP value, CH$_4$ conversion\n(percentage points)")
axb.set_xlim(370, 1030)
S.fixed_ticks(axb, "x", [400, 600, 800, 1000])
axb.set_ylim(-72, 42)
S.fixed_ticks(axb, "y", [-60, -40, -20, 0, 20, 40])
axb.set_xlabel("Reaction temperature (°C)")
axb.text(0.04, 0.97, f"Spearman ρ = {S.num(round(SP['Reaction Temperature'], 2), 2)}", transform=axb.transAxes,
         ha="left", va="top", fontsize=S.SMALL_PT, color=S.MUTED)
axc.set_xscale("log")
axc.set_xlim(1.2e3, 1.4e6)
axc.set_xticks([1e3, 1e4, 1e5, 1e6], labels=["1", "10", "100", "1,000"])     # thousands (see axis label)
axc.set_ylim(-17, 17)
S.fixed_ticks(axc, "y", [-15, -10, -5, 0, 5, 10, 15])
axc.set_xlabel("GHSV (10$^3$ mL g$^{-1}$ h$^{-1}$, as reported)")
axc.text(0.96, 0.97, f"Spearman ρ = {S.num(round(SP['GHSV'], 2), 2)}", transform=axc.transAxes, ha="right",
         va="top", fontsize=S.SMALL_PT, color=S.MUTED)

# ---------------------------------------------------------------- d: R2 of the explanation model
ax = axd
rows = np.arange(3)
BH = 0.34
b_in = ax.barh(rows - BH / 2 - 0.01, r2_in, height=BH, color=S.GREY_LIGHT, lw=0, zorder=2)
b_out = ax.barh(rows + BH / 2 + 0.01, r2_out, height=BH, color=S.GREY_DARK, lw=0, zorder=2)
for p in b_in:
    p.set_gid("d:in")
for p in b_out:
    p.set_gid("d:out")
ax.set_ylim(2.6, -0.6)
ax.set_yticks(rows, labels=[TNICE[t] for t in TARGETS])
ax.tick_params(axis="y", length=0, pad=3, labelsize=S.LABEL_PT)
ax.set_xlim(0, 1.0)
S.fixed_ticks(ax, "x", [0, 0.25, 0.5, 0.75, 1.0], labels=["0", "0.25", "0.50", "0.75", "1.00"])
S.grid(ax, "x")
ax.set_xlabel("R$^2$ of the explanation model")
ax.text(0.025, rows[0] - BH / 2 - 0.01, "Fitted records", ha="left", va="center", fontsize=S.SMALL_PT, color=S.INK)
ax.text(r2_out[0] + 0.025, rows[0] + BH / 2 + 0.01, "Held-out publications", ha="left", va="center",
        fontsize=S.SMALL_PT, color=S.GREY_DARK)

S.header(axb, "Reaction temperature", pad=4)
S.header(axc, "Gas hourly space velocity", pad=4)
S.header(axd, "Explanation-model accuracy", pad=4)
S.panel_letters(fig, [[(axa, "a")], [(axb, "b"), (axc, "c"), (axd, "d")]])
FIGNAME = "Fig4_interpretation_shap"


# ---------------------------------------------------------------- verification against the v1 figure
def verify():
    v1, v2 = V1["artists"], S.extract_artists(fig)
    n = 0
    v1a = [a for a in v1 if a["xlabel"] == "Share of mean |SHAP| (%)"]
    assert len(v1a) == 3
    for k, t in enumerate(TARGETS):
        w1 = np.array([b[2] for b in v1a[k]["bars"]])
        w2 = np.array([b[2] for b in v2[k]["bars"]])
        S.assert_close(w2, w1, f"a bars {t}")
        y1 = np.array([b[1] + b[3] / 2 for b in v1a[k]["bars"]])
        y2 = np.array([b[1] + b[3] / 2 for b in v2[k]["bars"]])
        S.assert_close(y2, y1, f"a bar positions {t}")
        s1 = np.array([[s[0, 0], s[1, 0]] for s in v1a[k]["segments"][0]["segs"]])
        s2 = np.array([[s[0, 0], s[1, 0]] for s in next(sg for sg in v2[k]["segments"] if sg["gid"] == f"a:{t}:range")["segs"]])
        S.assert_close(s2, s1, f"a refit ranges {t}")
        n += 3
    for feat, lab in (("Reaction Temperature", "Reaction temperature"), ("GHSV", "GHSV")):
        a1 = next(a for a in v1 if a["xlabel"].startswith(lab))
        a2 = next(a for a in v2 if any(s["gid"] == f"dep:{feat}" for s in a["scatter"]))
        p1 = a1["scatter"][0]["xy"]
        p2 = next(s for s in a2["scatter"] if s["gid"] == f"dep:{feat}")["xy"]
        S.assert_close(p2, p1, f"dependence points {feat}")
        t1 = [t for t in a1["texts"] if t.startswith("Spearman")][0]
        assert t1.replace("-", "−") == f"Spearman ρ = {S.num(round(SP[feat], 2), 2)}", (t1, SP[feat])
        n += 2
    a1 = next(a for a in v1 if a["ylabel"].startswith("R$^2$"))
    h1 = np.array([b[3] for b in a1["bars"]])
    h2 = np.array([b[2] for b in v2[5]["bars"]])
    S.assert_close(h2, h1, "d R2 (in-sample, then held-out)")
    n += 1
    stored = S.v1_numbers_text("fig4_numbers.txt").splitlines()      # v1 printed numbers ("saved <path>" line removed)
    assert [s.rstrip() for s in V1["stdout"].splitlines()] == [s.rstrip() for s in stored], "v1 printed numbers changed"
    n += 1
    print(f"verification Fig 4: {n} array comparisons against the v1 figure passed "
          f"(bars, refit ranges, {len(p1)} dependence points per panel, Spearman labels, R2 bars, printed numbers).")


if __name__ == "__main__":
    S.save(fig, FIGNAME)
    S.write_numbers("fig4_numbers.txt", V1["stdout"])
    verify()
