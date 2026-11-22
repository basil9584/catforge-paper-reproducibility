"""Compare regenerated results with the stored references (run by `python reproduce.py --check`).

Checks, grouped by manuscript item (Fig. 1-6, Table 1, Table 2, data and code):
  1. figure scripts   each figure script ran and its own verification passed (every plotted array equals the value
                      computed by its computation script; the computation script's printed numbers equal the stored
                      reference numbers in results/reference_numbers/)
  2. key numbers      every number in results/reference_numbers/manuscript_key_numbers.json (transcribed from the
                      manuscript) is recomputed and must round to the printed value
  3. re-derived tables the stored summary tables that Figs 2-3 and Table 1 read are re-derived from the stored
                      per-split / per-record benchmark outputs and must equal the stored tables
  4. integrity        data files and the platform predictor match the SHA-256 recorded when the benchmarks ran;
                      the locked test groups recomputed from the code equal the stored split files
  5. Fig. 1           key labels present as text in the SVG; width 183 mm; no embedded raster images
  6. images           regenerated PNGs compared with results/reference_figures/ (informational: identical on the
                      authors' machine; may differ elsewhere if Arial or the library versions differ)
Usage: python checks/check_results.py [--figures-dir output/figures] [--out output/check]
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import shutil
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PKG / "figures"), str(PKG / "benchmark")]
import style_v2 as S  # noqa: E402

RES = PKG / "results"
REF_NUM = RES / "reference_numbers"
TARGETS = ["Fig. 1", "Fig. 2", "Fig. 3", "Table 1", "Fig. 4", "Fig. 5", "Fig. 6", "Table 2", "Data and code"]
FIGURES = {  # manuscript figure -> (figure script, output name)
    "Fig. 1": ("fig1_pipeline.py", "Fig1_CatFORGE_pipeline"),
    "Fig. 2": ("fig2_synthetic.py", "Fig2_synthetic_fidelity_vs_usefulness"),
    "Fig. 3": ("fig3_benchmark.py", "Fig3_benchmark_heldout"),
    "Fig. 4": ("fig4_shap.py", "Fig4_interpretation_shap"),
    "Fig. 5": ("fig5_cost.py", "Fig5_cost_aware_ranking"),
    "Fig. 6": ("fig6_lab.py", "Fig6_laboratory_comparison"),
}


class Report:
    def __init__(self):
        self.rows = []

    def add(self, target, check, status, detail=""):
        assert status in ("PASS", "FAIL", "INFO")
        self.rows.append({"target": target, "check": check, "status": status, "detail": str(detail)})
        print(f"  [{status}] {target}: {check}" + (f"  ({detail})" if detail else ""), flush=True)


def quiet(fn, *a, **k):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def fig5_prices():
    """The price argument used for the manuscript, read from figures/fig5_cost.py (single source of truth)."""
    tree = ast.parse((PKG / "figures" / "fig5_cost.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "PRICES":
            return node.value.value
    raise RuntimeError("PRICES not found in figures/fig5_cost.py")


def g(script, args=()):
    return S.v1_dry_run(script, args)["globals"]


# ---------------------------------------------------------------------------------------------------- key numbers
def numbers_fig2():
    g2 = g("fig2_synthetic_compute.py")
    main, abl, fu = g2["main"], g2["abl"], g2["fu"]
    ca, so = main[main.arm == "catforge_aug"], main[main.arm == "catforge_synth_only"]
    out = {}
    import run_benchmark as rb
    for name in ("drm", "wgs", "ocm"):
        ds = quiet(rb.LOADERS[name])
        vals = [len(ds.frame)] + ([] if name == "ocm" else [ds.frame["group"].nunique()])
        out[f"fig2.records_{name}"] = (vals, [])
    out["fig2.sizes"] = ([main["size"].min(), main["size"].max()], [])
    out["fig2.within_2pct"] = ([int((ca.rel_delta_pct.abs() <= 2).sum()), len(ca)], [])
    out["fig2.significant"] = ([int((ca.ci_high < 0).sum()), int((ca.ci_low > 0).sum())], [])
    r = ca.loc[ca.rel_delta_pct.idxmax()]
    out["fig2.max_increase"] = ([r.rel_delta_pct, r.ci_low, r.ci_high, r["size"]], [r.dataset, r.sampling])
    out["fig2.synth_only_worse"] = ([int((so.rel_delta_pct > 0).sum()), len(so)], [])
    out["fig2.synth_only_max"] = ([so.rel_delta_pct.max()], [])
    worst = abl[abl.arm != "real_only"].groupby("arm").rel_delta_pct.max()
    out["fig2.ctgan_tvae_max"] = ([worst["ctgan_aug_r1"], worst["tvae_aug_r1"]], [])
    fa = fu[fu.grid == "ablation"]
    d = fa[fa.dataset == "drm"].mean_within_size_spearman
    out["fig2.spearman_drm"] = ([d.max(), d.min()], [])
    w = fa[fa.dataset.isin(["wgs", "ocm"])].mean_within_size_spearman
    out["fig2.spearman_wgs_ocm"] = ([w.min(), w.max()], [])
    top = [fa[fa.dataset == ds].set_index("arm").quality_mean.idxmax() for ds in ("wgs", "ocm")]
    tv = abl[(abl.arm == "tvae_aug_r1") & abl.dataset.isin(["wgs", "ocm"])]
    out["fig2.tvae_top_quality"] = ([], top + [f"increased error: {bool((tv.rel_delta_pct > 0).all())}"])
    n_sets = sorted(set(main.n_sets) | set(abl.n_sets))
    out["fig2.splits"] = ([n_sets[0] if len(n_sets) == 1 else np.nan], [])
    return out


def numbers_fig3():
    g3 = g("fig3_benchmark_compute.py")
    summ, comp = g3["summ"], g3["comp"]
    full = summ[summ.regime == "full"].copy()
    out = {}
    ng = full.groupby("dataset").n_groups.first()
    out["fig3.locked_groups"] = ([ng["drm"], ng["wgs"], ng["ocm"]], [])
    out["fig3.models"] = ([full.model.nunique()], [])
    c2 = comp[(comp.baseline == "best_other") & (comp.focus == "catforge_v2")]
    c1 = comp[(comp.baseline == "best_other") & (comp.focus == "catforge_v1")]
    f2 = c2[c2.regime == "full"].set_index(["dataset", "target"])
    f1 = c1[c1.regime == "full"].set_index(["dataset", "target"])
    order = [("drm", "CH4 Conversion"), ("drm", "CO2 Conversion"), ("drm", "Syngas_Ratio"), ("wgs", "CO Conversion"),
             ("ocm", "X(CH4), %"), ("ocm", "S(C2), %"), ("ocm", "Y(C2),%")]
    out["fig3.verdicts"] = ([], [f"{ds} {t}: {f2.loc[(ds, t), 'verdict']}" for ds, t in order])
    tie = f2.loc[[k for k in order if k[0] in ("drm", "wgs")]]
    out["fig3.tie_ratios"] = ([tie.ratio.min(), tie.ratio.max()], [])
    ocm = f2.loc[[k for k in order if k[0] == "ocm"]]
    out["fig3.ocm_worse"] = ([(ocm.ratio.min() - 1) * 100, (ocm.ratio.max() - 1) * 100], [])
    out["fig3.ocm_ci"] = ([ocm.ratio_lo.min(), ocm.ratio_hi.max()], [])
    full["rank"] = full.groupby(["dataset", "target"]).rmse.rank()
    mr = full.groupby("model")["rank"].mean()
    out["fig3.mean_rank"] = ([mr["catforge_v2"], mr["lgbm"], mr["xgb"], mr["catforge_v1"]], [])
    out["fig3.small_data"] = ([(c2[c2.regime.isin(["50", "200"])].ratio.max() - 1) * 100], [])
    rm = full.set_index(["dataset", "target", "model"]).rmse
    ids = ["table1.drm_ch4", "table1.drm_co2", "table1.drm_h2co", "table1.wgs_co", "table1.ocm_ch4", "table1.ocm_sel",
           "table1.ocm_yield"]
    for i, (ds, t) in zip(ids, order):
        r = f2.loc[(ds, t)]
        out[i] = ([rm[(ds, t, "catforge_v2")], rm[(ds, t, r.baseline_model)], r.ratio, r.ratio_lo, r.ratio_hi,
                   f1.loc[(ds, t), "ratio"]], [r.baseline_model])
    return out


def numbers_fig4():
    from scipy.stats import spearmanr
    g4 = g("fig4_shap_compute.py")
    df, share, folds, imp, INPUTS, TARGETS_, full = (g4[k] for k in ("df", "share", "folds", "imp", "INPUTS", "TARGETS",
                                                                     "full"))
    out = {"fig4.records": ([len(df), df["Paper no."].astype(str).nunique()], [])}
    out["fig4.temperature_share"] = ([share.loc["Reaction Temperature", t] for t in TARGETS_],
                                     [share[t].idxmax() for t in TARGETS_])
    firsts = []
    for t in TARGETS_:
        topf = imp[t].idxmax()
        firsts.append(sum(int(pd.Series(f, index=INPUTS).rank(ascending=False)[topf]) == 1 for f in folds[t]))
    out["fig4.refits"] = (firsts, [])
    out["fig4.temperature_range"] = ([df["Reaction Temperature"].min(), df["Reaction Temperature"].max()], [])
    s = full["CH4 Conversion"]
    out["fig4.spearman"] = ([spearmanr(df[f], s[:, INPUTS.index(f)]).statistic for f in ("Reaction Temperature", "GHSV")],
                            [])
    out["fig4.r2_fitted"] = ([min(g4["r2_in"]), max(g4["r2_in"])], [])
    out["fig4.r2_heldout"] = ([min(g4["r2_out"]), max(g4["r2_out"])], [])
    return out


def numbers_fig5():
    g5 = g("fig5_cost_compute.py", (fig5_prices(),))
    cand, choice, wgrid, meas, P = g5["cand"], g5["choice"], list(g5["wgrid"]), g5["meas"], g5["PRICES"]
    p = cand.set_index("Ni Loading")["CH4 Conversion_pred"]
    plateau = p[p.index >= 5]
    cb = choice["baseline"]
    w0 = cb[wgrid.index(0.0)]
    tail = [c for w, c in zip(wgrid, cb) if w >= 0.15]
    tail_ok = set(tail) <= {2.5, 2.0} and tail[0] == 2.5 and all(a >= b for a, b in zip(tail[:-1], tail[1:]))
    same = all(list(choice[k]) == list(cb) for k in ("low", "high"))
    return {
        "fig5.records": ([len(g5["df"])], []),
        "fig5.prices": ([P["ni_base"], P["al_base"]], []),
        "fig5.surrogate_size": ([len(g5["ens"]["CH4 Conversion"])], []),
        "fig5.ch4_2_5": ([p[2.0], p[5.0]], []),
        # "stayed within 0.8 percentage points of that value": the tightest one-decimal bound on |p - p(5 wt%)| above 5 wt%
        "fig5.plateau": ([np.ceil(float((plateau - p[5.0]).abs().max()) * 10) / 10], []),
        "fig5.cost_range": ([cand.cost_baseline.min(), cand.cost_baseline.max()], []),
        "fig5.measured_context": ([meas["Paper no."].nunique(), meas["CH4 Conversion"].min(), meas["CH4 Conversion"].max(),
                                   meas["GHSV"].min(), meas["GHSV"].max()], []),
        "fig5.performance_only": ([w0, p[w0] - p[5.0]], []),
        "fig5.rmse": ([g5["HELDOUT_RMSE"]], []),
        "fig5.cost_weight": ([cb[wgrid.index(0.05)], cb[wgrid.index(0.15)], cb[-1]],
                             [f"w >= 0.15 gives only 2.5 then 2 wt%: {tail_ok}",
                              f"same choices for low, baseline and high prices: {same}"]),
    }


def numbers_fig6():
    d = g("fig6_lab_compute.py")["d"]
    conv = d[d.response.str.contains("conversion")]
    absd = conv.absolute_difference.astype(float)
    cal = d[d.case.str.contains("calcination")].copy()
    cal["T"] = cal.case.str.extract(r"NiHT-(\d+)").astype(int)
    ch4 = cal[cal.response == "CH4 conversion (%)"].sort_values("T")
    pred = ch4.predicted.astype(float).to_numpy()

    def case(name):
        c = d[d.case == name].set_index("response")
        h2 = c.loc[[r for r in c.index if r.startswith("H2/CO")]]
        vals = [c.loc["CH4 conversion (%)", "predicted"], c.loc["CO2 conversion (%)", "predicted"],
                c.loc["CH4 conversion (%)", "measured"], c.loc["CO2 conversion (%)", "measured"], h2.predicted.iloc[0]]
        return [float(v) for v in vals] + [float(v) for v in h2.measured]

    tab2 = []
    for name in ("NiHT higher severity", "Ni-AC", "Ni-alumina Flowrence"):
        c = d[d.case == name].set_index("response").signed_difference_prediction_minus_measured.astype(float)
        tab2 += [c["CH4 conversion (%)"], c["CO2 conversion (%)"]]
    return {
        "fig6.within5": ([int((absd <= 5).sum()), len(conv)], []),
        "fig6.max": ([absd.max()], []),
        "fig6.calcination": ([int(ch4["T"].iloc[int(np.argmax(ch4.measured.astype(float)))]),
                              int(ch4["T"].iloc[int(np.argmax(pred))])],
                             [f"predictions decline monotonically: {bool(np.all(np.diff(pred) < 0))}"]),
        "fig6.ni_ht": (case("NiHT higher severity"), []),
        "fig6.ni_ac": (case("Ni-AC"), []),
        "fig6.ni_alumina": (case("Ni-alumina Flowrence"), []),
        "table2.lab_column": (tab2, []),
    }


def target_of(item):
    f = item["figure"]
    for t in ("Table 1", "Table 2"):
        if f.startswith(t):
            return t
    for t in ("Fig. 2", "Fig. 3", "Fig. 4", "Fig. 5", "Fig. 6"):
        if f.startswith(t):
            return t
    return f


def check_key_numbers(rep):
    ref = json.loads((REF_NUM / "manuscript_key_numbers.json").read_text(encoding="utf-8"))["items"]
    computed = {}
    for fn in (numbers_fig2, numbers_fig3, numbers_fig4, numbers_fig5, numbers_fig6):
        try:
            computed.update(quiet(fn))
        except Exception as e:  # noqa: BLE001
            rep.add("Data and code", f"key numbers: {fn.__name__} raised an error", "FAIL", repr(e)[:300])
    rows = []
    for it in ref:
        tgt = target_of(it)
        if it["id"] not in computed:
            rep.add(tgt, f"manuscript number {it['id']}", "FAIL", "not computed")
            continue
        vals, labels = computed[it["id"]]
        exp = it["values"]
        dec = it.get("decimals", 0)
        dec = dec if isinstance(dec, list) else [dec] * len(exp)
        ok = len(vals) == len(exp)
        shown = []
        for v, e, dd in zip(vals, exp, dec):
            v = float(v)
            good = np.isfinite(v) and abs(v - e) <= 0.5 * 10 ** (-dd) + 1e-9
            ok &= bool(good)
            shown.append(f"{v:.{dd + 2}f}")
        if it.get("labels") is not None:
            ok &= [str(x) for x in labels] == it["labels"]
        detail = f"printed {exp}{' ' + str(it['labels']) if it.get('labels') else ''}; recomputed [{', '.join(shown)}]" \
                 f"{' ' + str([str(x) for x in labels]) if labels else ''}"
        rep.add(tgt, f"manuscript number {it['id']}: \"{it['manuscript'][:70]}\"", "PASS" if ok else "FAIL", detail)
        rows.append({**it, "recomputed": [float(v) for v in vals], "recomputed_labels": [str(x) for x in labels],
                     "pass": bool(ok)})
    return rows


# ---------------------------------------------------------------------------------------------------- re-derivation
def same_table(a, b):
    """'identical' (bytes), 'numerically equal' (1e-9) or a description of the difference."""
    a, b = Path(a), Path(b)
    if a.read_bytes() == b.read_bytes():
        return True, "byte-identical"
    if a.suffix == ".json":
        eq = json.loads(a.read_text(encoding="utf-8")) == json.loads(b.read_text(encoding="utf-8"))
        return eq, "JSON-equal" if eq else "JSON content differs"
    x, y = pd.read_csv(a), pd.read_csv(b)
    if list(x.columns) != list(y.columns) or x.shape != y.shape:
        return False, f"shape/columns differ: {x.shape} vs {y.shape}"
    worst, where = 0.0, ""
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
            u, v = x[c].to_numpy(float), y[c].to_numpy(float)
            if not np.array_equal(np.isnan(u), np.isnan(v)):
                return False, f"missing values differ in {c}"
            m = np.nanmax(np.abs(u - v)) if np.isfinite(u).any() else 0.0
            if np.isfinite(m) and m > worst:
                worst, where = float(m), c
        elif not (x[c].astype(str) == y[c].astype(str)).all():
            return False, f"text column {c} differs"
    return worst <= 1e-9, f"max |difference| {worst:.3g}" + (f" (column {where})" if where else "")


def check_rederived(rep, work):
    import run_benchmark as rb
    aug = work / "augmentation_benchmark"
    shutil.rmtree(aug, ignore_errors=True)
    aug.mkdir(parents=True)
    for p in sorted((RES / "augmentation_benchmark").glob("*_raw.csv")):
        shutil.copy2(p, aug / p.name)
    rb.OUT = aug
    quiet(rb.summarise)
    for f in ("benchmark_summary.csv", "benchmark_fidelity_vs_utility.csv"):
        ok, how = same_table(aug / f, RES / "augmentation_benchmark" / f)
        rep.add("Fig. 2", f"{f} re-derived from the stored per-split results (run_benchmark.summarise)",
                "PASS" if ok else "FAIL", how)
    import analyze_final
    pred = work / "predictor_benchmark"
    quiet(analyze_final.main, RES / "predictor_benchmark" / "final", pred)
    for f in ("final_summary.csv", "final_catforge_vs_baselines.csv", "final_headline.json"):
        ok, how = same_table(pred / f, RES / "predictor_benchmark" / f)
        for tgt in ("Fig. 3", "Table 1"):
            rep.add(tgt, f"{f} re-derived from the stored locked-test predictions (analyze_final.py)",
                    "PASS" if ok else "FAIL", how)


# ---------------------------------------------------------------------------------------------------- integrity
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_integrity(rep):
    info = json.loads((RES / "run_info.json").read_text(encoding="utf-8"))["runs"]
    for ds in ("drm", "wgs", "ocm"):
        rec = info[f"augmentation_main_{ds}"]
        ok = sha(PKG / rec["data_file"]) == rec["data_sha256"]
        rep.add("Data and code", f"{rec['data_file']} has the SHA-256 recorded by the benchmark run", "PASS" if ok else "FAIL",
                rec["data_sha256"][:16] + "...")
    rec = info["locked_test_drm"]
    ok = sha(PKG / "catforge" / "generalized_predictor.py") == rec["predictor_file_sha256"]
    rep.add("Data and code", "catforge/generalized_predictor.py is the version evaluated on the locked test groups",
            "PASS" if ok else "FAIL", rec["predictor_file_sha256"][:16] + "...")
    import run_ablation as ra
    import run_benchmark as rb
    for ds in ("drm", "wgs", "ocm"):
        d = quiet(rb.LOADERS[ds])
        test, dev = ra.locked_split(d)
        st = json.loads((RES / "predictor_benchmark" / "splits" / f"split_{ds}.json").read_text(encoding="utf-8"))
        rows = int(d.frame["group"].isin(test).sum())
        ok = list(test) == st["locked_test_groups"] and rows == st["locked_test_rows"] and len(d.frame) - rows == st["dev_rows"]
        for tgt in ("Fig. 3", "Table 1"):
            rep.add(tgt, f"locked test groups recomputed by run_ablation.locked_split equal splits/split_{ds}.json",
                    "PASS" if ok else "FAIL", f"{len(test)} groups, {rows} rows")


# ---------------------------------------------------------------------------------------------------- Fig. 1, images
FIG1_LABELS = ["Biomass pre-treatment", "Carbonization", "Chemical activation", "Ni incorporation", "Post-treatment",
               "Characterization", "Fixed-bed DRM testing", "Experimental records", "QEERI laboratory measurements",
               "Literature compilations", "Generate", "Evaluate", "Predict & explain", "Optimize", "Cost & trade-offs",
               "Ranked candidates", "new measurements appended", "no programming required", "TOPSIS", "CatCost-style",
               "repeated for every change"]


def check_fig1(rep, fig_dir):
    svg = fig_dir / "Fig1_CatFORGE_pipeline.svg"
    if not svg.exists():
        rep.add("Fig. 1", "SVG output present", "FAIL", "run the figures first")
        return
    root = ET.parse(svg).getroot()
    ns = "{http://www.w3.org/2000/svg}"
    text = " ".join("".join(t.itertext()) for t in root.iter(f"{ns}text"))
    missing = [m for m in FIG1_LABELS if m not in text]
    rep.add("Fig. 1", f"{len(FIG1_LABELS)} key labels present as text in the SVG", "PASS" if not missing else "FAIL",
            f"missing: {missing}" if missing else "all present")
    w_mm = float(root.get("width").replace("pt", "")) / 72 * 25.4
    images = len(list(root.iter(f"{ns}image")))
    rep.add("Fig. 1", "vector figure, 183 mm wide, no embedded raster image",
            "PASS" if abs(w_mm - 183) < 0.2 and images == 0 else "FAIL", f"{w_mm:.1f} mm, {images} images")


def check_images(rep, fig_dir):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    for tgt, (_, name) in FIGURES.items():
        new, ref = fig_dir / f"{name}.png", RES / "reference_figures" / f"{name}.png"
        if not new.exists():
            rep.add(tgt, "regenerated PNG present", "FAIL", f"{new.name} missing")
            continue
        if new.read_bytes() == ref.read_bytes():
            rep.add(tgt, "PNG identical to the manuscript figure (results/reference_figures)", "INFO", "byte-identical")
            continue
        a = np.asarray(Image.open(new).convert("RGB"), dtype=np.int16)
        b = np.asarray(Image.open(ref).convert("RGB"), dtype=np.int16)
        if a.shape != b.shape:
            how = f"different size {a.shape[1]}x{a.shape[0]} vs {b.shape[1]}x{b.shape[0]} px"
        else:
            dmax = np.abs(a - b).max(axis=2)
            how = "pixel-identical" if dmax.max() == 0 else f"{100 * (dmax > 0).mean():.3f}% of pixels differ"
        rep.add(tgt, "PNG compared with the manuscript figure (results/reference_figures)", "INFO",
                how + " (differences are expected if Arial or library versions differ; the numeric checks decide)")


def check_figure_scripts(rep, status_file):
    if not status_file.exists():
        for tgt in FIGURES:
            rep.add(tgt, "figure script ran and verified its plotted values", "FAIL", "no figure run found; run --figures")
        return
    st = json.loads(status_file.read_text(encoding="utf-8"))
    for tgt in FIGURES:
        s = st.get(tgt)
        if s is None:
            rep.add(tgt, "figure script ran", "FAIL", "not run")
            continue
        ok = s["returncode"] == 0
        rep.add(tgt, f"figures/{s['script']} ran" + ("" if tgt == "Fig. 1" else " and its verification passed"),
                "PASS" if ok else "FAIL", s.get("verification") or s.get("error", ""))


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-dir", default=str(PKG / "output" / "figures"))
    ap.add_argument("--status", default=str(PKG / "output" / "logs" / "figures_status.json"))
    ap.add_argument("--out", default=str(PKG / "output" / "check"))
    a = ap.parse_args()
    fig_dir, out = Path(a.figures_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = Report()
    print("1/6 figure scripts", flush=True)
    check_figure_scripts(rep, Path(a.status))
    print("2/6 manuscript key numbers", flush=True)
    numbers = check_key_numbers(rep)
    print("3/6 stored summary tables re-derived from stored benchmark outputs", flush=True)
    check_rederived(rep, out / "rederived")
    print("4/6 data and code integrity, locked test groups", flush=True)
    check_integrity(rep)
    print("5/6 Fig. 1 content", flush=True)
    check_fig1(rep, fig_dir)
    print("6/6 images", flush=True)
    check_images(rep, fig_dir)

    df = pd.DataFrame(rep.rows)
    lines = ["", "=" * 100, "CHECK SUMMARY (PASS = every hard check of that item passed; images are informational)",
             "=" * 100]
    overall = True
    for tgt in TARGETS:
        sub = df[df.target == tgt]
        hard = sub[sub.status != "INFO"]
        if tgt == "Table 2":
            n_ok = int((hard.status == "PASS").sum())
            state = "PASS" if (hard.status == "PASS").all() and len(hard) else "FAIL"
            lines.append(f"{tgt:<14} {state:<5} laboratory-test column only ({n_ok}/{len(hard)} checks); all other Table 2 "
                         "values are historical and NOT reproducible from code (see README)")
            overall &= state == "PASS"
            continue
        state = "PASS" if len(hard) and (hard.status == "PASS").all() else "FAIL"
        overall &= state == "PASS"
        img = sub[(sub.status == "INFO")]
        img_txt = f"; image: {img.detail.iloc[0].split(' (')[0]}" if len(img) else ""
        lines.append(f"{tgt:<14} {state:<5} {int((hard.status == 'PASS').sum())}/{len(hard)} checks passed{img_txt}")
    lines.append("-" * 100)
    lines.append(f"OVERALL: {'PASS' if overall else 'FAIL'}   details: output/check/check_report.json and check_report.txt")
    text = "\n".join(lines)
    print(text)
    full = "\n".join(f"[{r['status']}] {r['target']}: {r['check']}  ({r['detail']})" for r in rep.rows) + "\n" + text + "\n"
    (out / "check_report.txt").write_text(full, encoding="utf-8")
    (out / "check_report.json").write_text(json.dumps({"overall_pass": bool(overall), "checks": rep.rows,
                                                       "manuscript_numbers": numbers}, indent=1, default=str),
                                           encoding="utf-8")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
