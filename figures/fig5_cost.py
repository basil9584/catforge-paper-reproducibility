"""Figure 5 of the manuscript (v2 design; "Fig. 6" in the authors' working scripts): illustration of CatFORGE's
cost-aware ranking on the DRM literature compilation.

Data: identical to v1 (figures/compute/fig5_cost_compute.py: surrogate = bootstrap ensemble of 10 gradient-boosting
models from the platform's optimization module, materials cost, TOPSIS, Pareto front), run with the v1 price argument
PRICES below. The v1 script runs in a separate process with all file writes disabled (style_v2.v1_dry_run); its
results (`cand`, `choice`, `wgrid`, `meas`, `HELDOUT_RMSE`) are plotted here unchanged, the drawn values are asserted
equal to those in the v1 figure, and the numbers file the v1 script would write is compared with the stored
results/reference_numbers/fig5_cost_numbers.json (originally fig6_numbers.json).
Run from the package root: python figures/fig5_cost.py

Design change: v1 drew cost on a twin y-axis of panel a; v2 stacks two panels that share the Ni-loading axis
(top: conversion, bottom: cost). Both remain panel a.
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
import style_v2 as S  # noqa: E402

PRICES = '{"ni_low":17.0,"ni_base":17.0,"ni_high":21.495,"al_low":0.481,"al_base":0.570,"al_high":0.570}'
V1 = S.v1_dry_run("fig5_cost_compute.py", (PRICES,))
g = V1["globals"]
cand, choice, wgrid, meas, RMSE = g["cand"], g["choice"], g["wgrid"], g["meas"], g["HELDOUT_RMSE"]
ORANGE = S.OKABE_ITO["orange"]
C_CH4, C_CO2 = S.RESPONSE["ch4"], S.RESPONSE["co2"]
TOPSIS_W = (0.0, 0.05, 0.2)                     # the three cost weights highlighted in b (as v1)

# ---------------------------------------------------------------- layout (mm)
H = 62.0
TOP = 7.0
fig = S.figure(H)
ax_conv = S.axes_mm(fig, 13.5, TOP, 39.0, 27.5)
ax_cost = S.axes_mm(fig, 13.5, TOP + 27.5 + 3.0, 39.0, 14.5)
axb = S.axes_mm(fig, 73.0, TOP, 47.0, 45.0)
axc = S.axes_mm(fig, 135.5, TOP, 44.0, 45.0)
x = cand["Ni Loading"].to_numpy()

# ---------------------------------------------------------------- a (top): predicted conversion vs Ni loading
ax = ax_conv
sc = ax.scatter(meas["Ni Loading"], meas["CH4 Conversion"], s=3.0 ** 2, color=S.GREY_LIGHT, lw=0, zorder=2)
sc.set_gid("a:measured")
for t, col in (("CH4 Conversion", C_CH4), ("CO2 Conversion", C_CO2)):
    m, s = cand[f"{t}_pred"].to_numpy(), cand[f"{t}_uncertainty"].to_numpy()
    band = ax.fill_between(x, m - s, m + s, color=col, alpha=0.22, lw=0, zorder=3)
    band.set_gid(f"a:band:{t}")
    ln, = ax.plot(x, m, color=col, lw=S.DATA_LW, zorder=4)
    ln.set_gid(f"a:pred:{t}")
ax.set_xlim(0, 26)
ax.set_ylim(0, 90)
S.fixed_ticks(ax, "y", [0, 20, 40, 60, 80])
S.fixed_ticks(ax, "x", [0, 5, 10, 15, 20, 25])
ax.tick_params(axis="x", labelbottom=False)
S.grid(ax, "y")
ax.set_ylabel("Conversion (%)")
for t, lab in (("CO2 Conversion", "CO$_2$"), ("CH4 Conversion", "CH$_4$")):
    ax.text(25.6, cand[f"{t}_pred"].iloc[-1], lab, color=C_CH4 if t.startswith("CH4") else S.shade(C_CO2, 0.85),
            fontsize=S.LABEL_PT, ha="left", va="center", clip_on=False)
ka = [Line2D([], [], color=S.GREY_MID, lw=S.DATA_LW), Line2D([], [], marker="o", ls="", ms=3.0, color=S.GREY_LIGHT)]
ax.legend(ka, ["Predicted (±1 SD)", "Measured CH$_4$ conversion"], loc="lower left",
          handlelength=1.3, handletextpad=0.4, borderaxespad=0.25, labelspacing=0.25)
S.header(ax, "Conversion and cost vs Ni loading")

# ---------------------------------------------------------------- a (bottom): raw-material cost vs Ni loading
ax = ax_cost
band = ax.fill_between(x, cand["cost_low"], cand["cost_high"], color=S.GREY_MID, alpha=0.22, lw=0, zorder=2)
band.set_gid("a:cost_band")
ln, = ax.plot(x, cand["cost_baseline"], color=S.GREY_DARK, lw=S.THIN_LW + 0.15, ls="--", dashes=(3.2, 1.6), zorder=3)
ln.set_gid("a:cost")
ax.set_xlim(0, 26)
ax.set_ylim(0, 6.5)
S.fixed_ticks(ax, "y", [0, 2, 4, 6])
S.fixed_ticks(ax, "x", [0, 5, 10, 15, 20, 25])
ax.set_xlabel("Ni loading (wt%)")
ax.set_ylabel("Cost\n(US\\$ kg$^{-1}$)")
ax.text(0.03, 0.96, "Dashed: baseline prices\nBand: low–high prices", transform=ax.transAxes, ha="left", va="top",
        fontsize=S.SMALL_PT, color=S.MUTED, linespacing=1.15)

# ---------------------------------------------------------------- b: predicted CH4 conversion vs cost
ax = axb
cb, cm, cs = cand["cost_baseline"].to_numpy(), cand["CH4 Conversion_pred"].to_numpy(), cand["CH4 Conversion_uncertainty"].to_numpy()
# candidates are ordered by Ni loading and hence by cost: their +-1 SD is drawn as a band (same values as v1's bars)
assert np.all(np.diff(cb) > 0)
band = ax.fill_between(cb, cm - cs, cm + cs, color=C_CH4, alpha=0.16, lw=0, zorder=1)
band.set_gid("b:cand_sd")
cl, = ax.plot(cb, cm, "o", ms=3.0, color=S.tint(C_CH4, 0.45), mec="none", zorder=2)
cl.set_gid("b:cand")
p = cand[cand.pareto].sort_values("cost_baseline")
ln, = ax.plot(p["cost_baseline"], p["CH4 Conversion_pred"], "-o", ms=3.4, color=C_CH4, lw=S.DATA_LW, mec="white",
              mew=0.3, zorder=3)
ln.set_gid("b:pareto")
B_TOPSIS = {}
LAB_Y = {0.0: 70.6, 0.05: 70.6, 0.2: 62.2}
LAB_DX = {0.0: (0.0, "center"), 0.05: (0.0, "center"), 0.2: (-0.03, "left")}
for wc in TOPSIS_W:
    wsel = choice["baseline"][list(wgrid).index(wc)]
    r = cand[cand["Ni Loading"] == wsel].iloc[0]
    B_TOPSIS[wc] = (wsel, r["cost_baseline"], r["CH4 Conversion_pred"])
    s_ = ax.scatter([r["cost_baseline"]], [r["CH4 Conversion_pred"]], marker="D", s=4.4 ** 2, facecolor=ORANGE,
                    edgecolor=S.INK, lw=0.5, zorder=5)
    s_.set_gid(f"b:topsis:{wc}")
    above = LAB_Y[wc] > r["CH4 Conversion_pred"]
    ax.annotate(f"w = {S.num(wc)}\n{S.num(wsel)} wt% Ni", xy=(r["cost_baseline"], r["CH4 Conversion_pred"]),
                xytext=(r["cost_baseline"] + LAB_DX[wc][0], LAB_Y[wc]), ha=LAB_DX[wc][1], va="bottom" if above else "top",
                fontsize=S.SMALL_PT, color=S.INK, linespacing=1.1,
                arrowprops=dict(arrowstyle="-", color=S.GREY_MID, lw=0.4, shrinkA=1.0, shrinkB=3.2))
ax.set_xlim(0.6, 4.9)
S.fixed_ticks(ax, "x", [1, 2, 3, 4])
ax.set_ylim(56, 74)
S.fixed_ticks(ax, "y", [56, 60, 64, 68, 72])
ax.set_xlabel("Raw-material cost (US\\$ kg$^{-1}$ catalyst)")
ax.set_ylabel("Predicted CH$_4$ conversion (%)")
S.header(ax, "Performance–cost trade-off")
kb = [Line2D([], [], marker="o", ls="-", ms=3.4, color=C_CH4, lw=S.DATA_LW, mec="white", mew=0.3),
      (Patch(facecolor=C_CH4, alpha=0.16, lw=0), Line2D([], [], marker="o", ls="", ms=3.0, color=S.tint(C_CH4, 0.45))),
      Line2D([], [], marker="D", ls="", ms=4.4, mfc=ORANGE, mec=S.INK, mew=0.5)]
ax.legend(kb, ["Pareto front", "Candidates (±1 SD)", "TOPSIS top choice"], loc="lower right",
          bbox_to_anchor=(1.0, 0.15), handlelength=1.4, handletextpad=0.4, borderaxespad=0.3,
          labelspacing=0.3)
ax.text(0.98, 0.03, f"Surrogate RMSE on held-out publications: {RMSE:.1f} percentage points".replace(": ", ":\n"),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=S.SMALL_PT, color=S.MUTED, linespacing=1.15)

# ---------------------------------------------------------------- c: Ni loading of the top-ranked candidate vs cost weight
ax = axc
for key in ("low", "baseline", "high"):
    st, = ax.step(wgrid, choice[key], where="mid", color=S.GREY_DARK, lw=S.DATA_LW, zorder=3)
    st.set_gid(f"c:{key}")
for wc in TOPSIS_W:
    s_ = ax.scatter([wc], [choice["baseline"][list(wgrid).index(wc)]], marker="D", s=4.4 ** 2, facecolor=ORANGE,
                    edgecolor=S.INK, lw=0.5, zorder=5, clip_on=False)
    s_.set_gid(f"c:topsis:{wc}")
ax.set_xlim(-0.02, 0.82)
S.fixed_ticks(ax, "x", [0, 0.2, 0.4, 0.6, 0.8])
ax.set_ylim(0, 16.5)
S.fixed_ticks(ax, "y", [0, 5, 10, 15])
S.grid(ax, "y")
ax.set_xlabel("Weight on cost in TOPSIS, w")
ax.set_ylabel("Ni loading of top-ranked\ncandidate (wt%)")
ax.text(0.80, 8.0, "Low, baseline and high prices\n(the three lines overlap)", ha="right", va="center",
        fontsize=S.SMALL_PT, color=S.INK, linespacing=1.15)
S.header(ax, "Top-ranked candidate")

S.panel_letters(fig, [[([ax_conv, ax_cost], "a"), (axb, "b"), (axc, "c")]])
FIGNAME = "Fig5_cost_aware_ranking"      # manuscript numbering (the working folders called it Fig6_cost_aware_ranking)


# ---------------------------------------------------------------- verification against the v1 figure and numbers
def verify():
    v1, v2 = V1["artists"], S.extract_artists(fig)
    n = 0
    # the numbers file the v1 run would have written equals the stored v1 numbers file
    written = next(iter(V1["writes"].values()))
    stored = (S.V1_NUMBERS / "fig5_cost_numbers.json").read_text(encoding="utf-8")
    assert json.loads(written) == json.loads(stored), "fig6 numbers differ from the stored v1 numbers file"
    assert abs(json.loads(stored)["heldout_rmse_ch4"] - RMSE) < 1e-12
    assert f"{RMSE:.1f}" == "17.6"
    n += 2
    a1, b1, c1, tw1 = v1[0], v1[1], v1[2], v1[3]
    a2, cost2, b2, c2 = v2[0], v2[1], v2[2], v2[3]
    S.assert_close(a2["scatter"][0]["xy"], a1["scatter"][0]["xy"], "a measured records"); n += 1
    l1 = [l for l in a1["lines"] if l["data"]]
    for k, t in enumerate(("CH4 Conversion", "CO2 Conversion")):
        l2 = next(l for l in a2["lines"] if l["gid"] == f"a:pred:{t}")
        S.assert_close(l2["x"], l1[k]["x"], f"a {t} x"); S.assert_close(l2["y"], l1[k]["y"], f"a {t} y")
        f2 = next(f for f in a2["fills"] if f["gid"] == f"a:band:{t}")
        S.assert_close(f2["verts"][0], a1["fills"][k]["verts"][0], f"a {t} band"); n += 3
    lc1 = [l for l in tw1["lines"] if l["data"]][0]
    lc2 = next(l for l in cost2["lines"] if l["gid"] == "a:cost")
    S.assert_close(lc2["x"], lc1["x"], "a cost x"); S.assert_close(lc2["y"], lc1["y"], "a cost y")
    S.assert_close(cost2["fills"][0]["verts"][0], tw1["fills"][0]["verts"][0], "a cost band"); n += 3
    # b: candidates with SD, Pareto front, TOPSIS choices
    cand1 = next(l for l in b1["lines"] if l["data"] and len(l["x"]) == len(cand))     # the error-bar data line
    cand2 = next(l for l in b2["lines"] if l["gid"] == "b:cand")
    S.assert_close(cand2["x"], cand1["x"], "b candidates x"); S.assert_close(cand2["y"], cand1["y"], "b candidates y")
    # v1 drew +-1 SD as vertical bars, v2 as a band through the same end points
    seg1 = np.array([[s[0, 0], s[0, 1], s[1, 1]] for s in b1["segments"][0]["segs"]])
    vb = next(f for f in b2["fills"] if f["gid"] == "b:cand_sd")["verts"][0]
    nC = len(cand)
    lower = vb[1:nC + 1]                      # fill_between path: start, lower edge, end, upper edge (reversed), close
    upper = vb[nC + 2:2 * nC + 2][::-1]
    S.assert_close(lower[:, 0], seg1[:, 0], "b SD band x"); S.assert_close(upper[:, 0], seg1[:, 0], "b SD band x (upper)")
    S.assert_close(np.minimum(lower[:, 1], upper[:, 1]), np.minimum(seg1[:, 1], seg1[:, 2]), "b SD lower ends")
    S.assert_close(np.maximum(lower[:, 1], upper[:, 1]), np.maximum(seg1[:, 1], seg1[:, 2]), "b SD upper ends")
    par1 = next(l for l in b1["lines"] if l["label"] == "Pareto front")
    par2 = next(l for l in b2["lines"] if l["gid"] == "b:pareto")
    S.assert_close(par2["x"], par1["x"], "b Pareto x"); S.assert_close(par2["y"], par1["y"], "b Pareto y"); n += 5
    t1 = np.vstack([s["xy"] for s in b1["scatter"]])
    t2 = np.vstack([s["xy"] for s in b2["scatter"]])
    S.assert_close(t2, t1, "b TOPSIS choices (w = 0, 0.05, 0.2)"); n += 1
    # c: three step lines
    st1 = [l for l in c1["lines"] if l["data"]]
    for k, key in enumerate(("low", "baseline", "high")):
        l2 = next(l for l in c2["lines"] if l["gid"] == f"c:{key}")
        S.assert_close(l2["x"], st1[k]["x"], f"c {key} x"); S.assert_close(l2["y"], st1[k]["y"], f"c {key} y"); n += 2
    same = all(np.array_equal(choice["low"], choice[k]) for k in ("baseline", "high"))
    print(f"verification Fig 5 (cost): {n} comparisons against the v1 figure and numbers file passed; held-out RMSE "
          f"{RMSE:.4f} (printed as {RMSE:.1f}); low/baseline/high TOPSIS curves identical: {same}.")


if __name__ == "__main__":
    S.save(fig, FIGNAME)
    S.write_numbers("fig5_cost_numbers.json", next(iter(V1["writes"].values())))
    verify()
