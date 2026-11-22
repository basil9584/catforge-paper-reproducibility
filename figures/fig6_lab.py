"""Figure 6 of the manuscript (v2 design; "Fig. 5" in the authors' working scripts): predictions compared with QEERI
laboratory measurements.

Data: identical to v1 (figures/compute/fig6_lab_compute.py, reading data/qeeri_laboratory_comparisons.csv, originally
Laboratory_Comparisons_Recalculated.csv, without the NiHT-600 separate case). The v1 script runs in a separate process
with file writes disabled (style_v2.v1_dry_run); its table `d` is plotted here unchanged, the drawn values are asserted
equal to those in the v1 figure, and its printed numbers are asserted equal to results/reference_numbers/fig6_lab_numbers.txt.
Run from the package root: python figures/fig6_lab.py

Encoding (whole figure): colour = catalyst; circle = CH4, square = CO2 (as in the caption); filled = measured,
open (white fill, coloured edge) = predicted; solid line = measured, dashed = predicted.
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
import style_v2 as S  # noqa: E402

V1 = S.v1_dry_run("fig6_lab_compute.py")
d = V1["globals"]["d"]
CASE = {"Ni-alumina Flowrence": ("Ni–alumina (Flowrence reactor, 800 °C)", "Ni–alumina", S.OKABE_ITO["blue"]),
        "NiHT higher severity": ("Ni–HT (750 °C)", "Ni–HT", S.OKABE_ITO["purple"]),
        "Ni-AC": ("Ni–AC (date-palm carbon)", "Ni–AC", S.OKABE_ITO["green"]),
        "calc": ("Ni–HT calcination series", None, S.OKABE_ITO["orange"])}
MK = {"CH4": "o", "CO2": "s"}
MS = {"o": 4.3, "s": 3.9}

# ---------------------------------------------------------------- layout (mm)
H = 66.5
TOP, PH = 7.0, 44.0
fig = S.figure(H)
axa = S.axes_mm(fig, 13.0, TOP, 44.0, PH)
axb = S.axes_mm(fig, 72.0, TOP, 33.0, PH)
axc = S.axes_mm(fig, 120.5, TOP, 38.0, PH)

# ---------------------------------------------------------------- a: parity plot of conversion
ax = axa
band = ax.fill_between([20, 105], [15, 100], [25, 110], color="#EBEBEB", lw=0, zorder=0)
band.set_gid("band")
idl, = ax.plot([20, 105], [20, 105], color=S.REF, lw=0.6, zorder=1)
idl.set_gid("ref")
A_POINTS = []
for _, r in d[d.response.str.contains("conversion")].iterrows():
    key = "calc" if "calcination" in r.case else r.case
    col = CASE[key][2]
    mk = MK["CH4"] if r.response.startswith("CH4") else MK["CO2"]
    sc = ax.scatter([float(r.measured)], [float(r.predicted)], marker=mk, s=MS[mk] ** 2, color=col, edgecolor="white",
                    lw=0.4, zorder=3)
    sc.set_gid(f"a:{r.case}:{r.response}")
    A_POINTS.append((float(r.measured), float(r.predicted)))
ax.set_xlim(25, 100)
ax.set_ylim(25, 100)
S.fixed_ticks(ax, "x", [30, 40, 50, 60, 70, 80, 90, 100])
S.fixed_ticks(ax, "y", [30, 40, 50, 60, 70, 80, 90, 100])
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Measured conversion (%)")
ax.set_ylabel("Predicted conversion (%)")
S.header(ax, "Conversion")
key = [Line2D([], [], marker="o", ls="", ms=MS["o"], color=S.GREY_MID, mec="white", mew=0.4),
       Line2D([], [], marker="s", ls="", ms=MS["s"], color=S.GREY_MID, mec="white", mew=0.4),
       Patch(facecolor="#EBEBEB", lw=0)]
ax.legend(key, ["CH$_4$", "CO$_2$", "±5 points"], loc="upper left", handlelength=1.0, handletextpad=0.4,
          borderaxespad=0.5)

# ---------------------------------------------------------------- b: H2/CO ratio, measured vs predicted
ax = axb
hc = d[d.response.str.startswith("H2/CO")]
cases = ["Ni-alumina Flowrence", "NiHT higher severity", "Ni-AC"]
DX = 0.16
B_POINTS = {}
for i, c in enumerate(cases):
    sub = hc[hc.case == c]
    meas = sub.measured.astype(float).to_numpy()
    pred = float(sub.predicted.iloc[0])
    col = CASE[c][2]
    m = meas.mean()
    ax.plot([i - DX, i + DX], [m, pred], color=S.tint(col, 0.55), lw=0.8, zorder=2)[0].set_gid("connector")
    if len(meas) == 2:                       # measured range 0.79-0.80
        rng, = ax.plot([i - DX, i - DX], [meas.min(), meas.max()], color=col, lw=S.ERR_LW, zorder=3)
        rng.set_gid(f"b:{c}:range")
        ax.text(i - DX, m - 0.055, f"{S.num(meas.min(), 2)}–{S.num(meas.max(), 2)}", ha="center", va="top",
                fontsize=S.SMALL_PT, color=S.INK)
    lm, = ax.plot([i - DX], [m], "o", ms=4.5, color=col, mec="white", mew=0.4, zorder=4)
    lm.set_gid(f"b:{c}:measured")
    lp, = ax.plot([i + DX], [pred], "o", ms=4.5, mfc="white", mec=col, mew=0.9, zorder=4)
    lp.set_gid(f"b:{c}:predicted")
    B_POINTS[c] = (m, pred, meas)
ax.set_xlim(-0.55, 2.55)
ax.set_xticks(range(3), labels=[CASE[c][1] for c in cases])
ax.tick_params(axis="x", length=0, pad=3)
ax.set_ylim(0, 1.1)
S.fixed_ticks(ax, "y", [0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
S.grid(ax, "y")
ax.set_ylabel("H$_2$/CO ratio")
S.header(ax, "Syngas ratio")
kb = [Line2D([], [], marker="o", ls="", ms=4.5, color=S.GREY_MID, mec="white", mew=0.4),
      Line2D([], [], marker="o", ls="", ms=4.5, mfc="white", mec=S.GREY_MID, mew=0.9)]
ax.legend(kb, ["Measured", "Predicted"], loc="lower left", handlelength=1.0, handletextpad=0.4, borderaxespad=0.4)

# ---------------------------------------------------------------- c: calcination series
ax = axc
cal = d[d.case.str.contains("calcination")].copy()
cal["T"] = cal.case.str.extract(r"NiHT-(\d+)").astype(int)
col = CASE["calc"][2]
ends = []
C_LINES = {}
for resp, mk, name in (("CO2 conversion (%)", "s", "CO$_2$"), ("CH4 conversion (%)", "o", "CH$_4$")):
    s = cal[cal.response == resp].sort_values("T")
    lm, = ax.plot(s["T"], s.measured.astype(float), "-" + mk, color=col, lw=S.DATA_LW, ms=MS[mk], mec="white", mew=0.4,
                  zorder=4)
    lm.set_gid(f"c:{resp}:measured")
    lp, = ax.plot(s["T"], s.predicted.astype(float), "--" + mk, color=col, lw=S.THIN_LW + 0.2, dashes=(3.2, 1.6),
                  ms=MS[mk], mfc="white", mec=col, mew=0.8, zorder=3)
    lp.set_gid(f"c:{resp}:predicted")
    C_LINES[resp] = (s["T"].to_numpy(), s.measured.astype(float).to_numpy(), s.predicted.astype(float).to_numpy())
    ends.append((float(s.measured.astype(float).iloc[-1]), f"{name}, measured"))
    ends.append((float(s.predicted.astype(float).iloc[-1]), f"{name}, predicted"))
for yv, lab in ends:
    ax.text(935, yv, lab, ha="left", va="center", fontsize=S.LABEL_PT, color=S.INK, clip_on=False)
ax.set_xlim(260, 930)
S.fixed_ticks(ax, "x", [300, 600, 700, 900])
ax.set_ylim(50, 95)
S.fixed_ticks(ax, "y", [50, 60, 70, 80, 90])
S.grid(ax, "y")
ax.set_xlabel("Calcination temperature (°C)")
ax.set_ylabel("Conversion (%)")
S.header(ax, "Ni–HT calcination series")

# ---------------------------------------------------------------- shared catalyst key (bottom)
hk = [Line2D([], [], marker="o", ls="", ms=4.3, color=CASE[k][2], mec="white", mew=0.4) for k in CASE]
fig.legend(hk, [CASE[k][0] for k in CASE], loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0),
           bbox_transform=fig.transFigure, handletextpad=0.2, columnspacing=1.6, borderaxespad=0.6)

S.panel_letters(fig, [[(axa, "a"), (axb, "b"), (axc, "c")]])
FIGNAME = "Fig6_laboratory_comparison"      # manuscript numbering (the working folders called it Fig5_laboratory_comparison)


# ---------------------------------------------------------------- verification against the v1 figure
def verify():
    v1, v2 = V1["artists"], S.extract_artists(fig)
    n = 0
    p1 = np.vstack([s["xy"] for s in v1[0]["scatter"]])
    p2 = np.vstack([s["xy"] for s in v2[0]["scatter"]])
    S.assert_close(p2[np.lexsort(p2.T[::-1])], p1[np.lexsort(p1.T[::-1])], "a parity points"); n += 1
    # b: v1 bars = measured (mean of range) and predicted, in case order
    h1 = [b[3] for b in v1[1]["bars"]]
    h2 = []
    for c in cases:
        h2 += [B_POINTS[c][0], B_POINTS[c][1]]
    S.assert_close(h2, h1, "b H2/CO measured and predicted"); n += 1
    l2 = {l["gid"]: l for l in v2[1]["lines"]}
    for c in cases:
        S.assert_close(l2[f"b:{c}:measured"]["y"], [B_POINTS[c][0]], f"b v2 measured {c}")
        S.assert_close(l2[f"b:{c}:predicted"]["y"], [B_POINTS[c][1]], f"b v2 predicted {c}")
        n += 2
    seg = v1[1]["segments"][0]["segs"][0]
    S.assert_close(l2["b:NiHT higher severity:range"]["y"], seg[:, 1], "b Ni-HT measured range"); n += 1
    # c: four lines
    v1c = [l for l in v1[2]["lines"] if l["data"]]
    v2c = {l["gid"]: l for l in v2[2]["lines"]}
    for l1 in v1c:
        resp = "CH4 conversion (%)" if l1["label"].startswith("CH") else "CO2 conversion (%)"
        kind = "measured" if "measured" in l1["label"] else "predicted"
        l = v2c[f"c:{resp}:{kind}"]
        S.assert_close(l["x"], l1["x"], f"c {resp} {kind} x"); S.assert_close(l["y"], l1["y"], f"c {resp} {kind} y")
        n += 2
    stored = S.v1_numbers_text("fig6_lab_numbers.txt").splitlines()      # v1 printed numbers ("saved <path>" line removed)
    assert [s.rstrip() for s in V1["stdout"].splitlines()] == [s.rstrip() for s in stored], "v1 printed numbers changed"
    n += 1
    print(f"verification Fig 6 (laboratory): {n} array comparisons against the v1 figure passed ({len(p1)} parity points, "
          f"{len(h1)} H2/CO values, {len(v1c)} calcination lines, printed numbers).")


if __name__ == "__main__":
    S.save(fig, FIGNAME)
    S.write_numbers("fig6_lab_numbers.txt", V1["stdout"])
    verify()
