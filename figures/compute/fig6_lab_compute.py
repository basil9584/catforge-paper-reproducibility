"""Figure 5 of the working scripts = Figure 6 of the manuscript: predictions compared with QEERI laboratory measurements
(values as reported in the merged CatFORGE manuscript; historical model runs, not re-executed; differences recalculated).
Source: data/qeeri_laboratory_comparisons.csv (originally Laboratory_Comparisons_Recalculated.csv). The stand-alone
NiHT-600 case is excluded because its reaction time is reported inconsistently (20 h in text, 12 h in the figure capture)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).parent))
from figstyle import plt, DOUBLE, MM, GRID, MUTED, INK, panel, save

d = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "qeeri_laboratory_comparisons.csv")
d = d[d.case != "NiHT-600 separate case"]
CASE = {"Ni-alumina Flowrence": ("Ni\u2013alumina (Flowrence, 800 \u00b0C)", "#0072B2"),
        "NiHT higher severity": ("Ni\u2013HT (750 \u00b0C)", "#CC79A7"),
        "Ni-AC": ("Ni\u2013AC (date-palm carbon)", "#009E73"),
        "calc": ("Ni\u2013HT calcination series", "#E69F00")}
fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 60 * MM), gridspec_kw={"width_ratios": [1.05, 0.85, 1.05], "wspace": 0.42})

ax = axes[0]
ax.fill_between([20, 100], [15, 95], [25, 105], color="#EEEEEE", lw=0, zorder=0)
ax.plot([20, 100], [20, 100], color="#555", lw=0.7, zorder=1)
for _, r in d[d.response.str.contains("conversion")].iterrows():
    key = "calc" if "calcination" in r.case else r.case
    lab, col = CASE[key]
    mk = "o" if r.response.startswith("CH4") else "s"
    ax.scatter(float(r.measured), float(r.predicted), marker=mk, s=18, color=col, edgecolor="white", lw=0.4, zorder=3)
ax.set_xlim(25, 100); ax.set_ylim(25, 100)
ax.set_xlabel("Measured conversion (%)"); ax.set_ylabel("Predicted conversion (%)")
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.text(0.04, 0.95, "grey band: \u00b15 percentage points", transform=ax.transAxes, fontsize=5.8, color=MUTED, va="top")
h = [Line2D([], [], marker="o", ls="", color="#666", ms=3.4, label="CH$_4$"), Line2D([], [], marker="s", ls="", color="#666", ms=3.2, label="CO$_2$")]
ax.legend(handles=h, frameon=False, loc="lower right", fontsize=6, handletextpad=0.2)
panel(ax, "a", "Conversion: predicted vs measured", x=-0.27)

ax = axes[1]
hc = d[d.response.str.startswith("H2/CO")]
cases = ["Ni-alumina Flowrence", "NiHT higher severity", "Ni-AC"]
xx = np.arange(len(cases))
for i, c in enumerate(cases):
    sub = hc[hc.case == c]
    meas = sub.measured.astype(float).to_numpy(); pred = float(sub.predicted.iloc[0])
    col = CASE[c][1]
    if len(meas) == 2:                       # measured range 0.79-0.80
        ax.bar(i - 0.18, meas.mean(), 0.34, color=col, alpha=0.95)
        ax.errorbar(i - 0.18, meas.mean(), yerr=[[meas.mean() - meas.min()], [meas.max() - meas.mean()]], color="#222", lw=0.6, capsize=1.5)
    else:
        ax.bar(i - 0.18, meas[0], 0.34, color=col, alpha=0.95)
    ax.bar(i + 0.18, pred, 0.34, facecolor="white", edgecolor=col, hatch="////", lw=0.8)
ax.set_xticks(xx); ax.set_xticklabels(["Ni\u2013alumina", "Ni\u2013HT", "Ni\u2013AC"])
ax.set_ylabel("H$_2$/CO ratio"); ax.set_ylim(0, 1.15); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#777", label="measured"), Patch(facecolor="white", edgecolor="#777", hatch="////", label="predicted")],
          frameon=False, fontsize=6, loc="upper right")
panel(ax, "b", "Syngas ratio", x=-0.3)

ax = axes[2]
cal = d[d.case.str.contains("calcination")].copy()
cal["T"] = cal.case.str.extract(r"NiHT-(\d+)").astype(int)
for resp, col, mk in (("CH4 conversion (%)", "#0072B2", "o"), ("CO2 conversion (%)", "#D55E00", "s")):
    s = cal[cal.response == resp].sort_values("T")
    ax.plot(s["T"], s.measured.astype(float), color=col, marker=mk, ms=3.4, lw=1.1, label=f"{resp.split()[0].replace('4', '$_4$').replace('2', '$_2$')} measured")
    ax.plot(s["T"], s.predicted.astype(float), color=col, marker=mk, ms=3.4, lw=1.0, ls="--", mfc="white", label=f"{resp.split()[0].replace('4', '$_4$').replace('2', '$_2$')} predicted")
ax.set_xticks([300, 600, 700, 900]); ax.set_xlabel("Calcination temperature of Ni\u2013HT (\u00b0C)"); ax.set_ylabel("Conversion (%)")
ax.set_ylim(50, 101); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=5.8, ncol=2, loc="upper center", columnspacing=0.8, handlelength=1.6)
panel(ax, "c", "Calcination series", x=-0.24)
# shared case legend
hh = [Line2D([], [], marker="o", ls="", color=CASE[k][1], ms=3.6, label=CASE[k][0]) for k in ("Ni-alumina Flowrence", "NiHT higher severity", "Ni-AC", "calc")]
fig.legend(handles=hh, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=6.2, handletextpad=0.2, columnspacing=1.4)
save(fig, "Fig5_laboratory_comparison")

conv = d[d.response.str.contains("conversion")].copy()
conv["absd"] = conv.absolute_difference.astype(float)
print("conversion comparisons:", len(conv), "| within 5 pp:", int((conv.absd <= 5).sum()), "| median |diff|:", conv.absd.median(), "| max:", conv.absd.max())
print("non-calcination conversion |diff|:", conv[~conv.case.str.contains("calcination")].absd.tolist())
