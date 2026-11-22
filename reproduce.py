"""Single entry point of the CatFORGE reproducibility package. Run from the package root.

  python reproduce.py --figures            regenerate Figs 1-6 (PDF, SVG, PNG) in output/figures from the stored results
                                           (a few minutes); every figure script also verifies the values it plots
  python reproduce.py --check              compare regenerated numbers with the stored references and the numbers
                                           printed in the manuscript; PASS/FAIL per figure and table (runs --figures
                                           first; about 5 minutes in total)
  python reproduce.py --tests              cost-module tests and the predictor-replica equivalence test (a few minutes)
  python reproduce.py --quick              smoke run of the benchmark code on a small subset (minutes). It does NOT
                                           reproduce manuscript numbers; it only shows that the code runs end to end.
  python reproduce.py --benchmarks         full re-run of the augmentation and predictor benchmarks (hours; prints an
                                           estimate and asks for confirmation; --yes skips the question, --jobs N runs
                                           N benchmark jobs in parallel). Results go to output/benchmarks and are
                                           compared with the stored results.
  python reproduce.py --figures --results-dir output/benchmarks
                                           redraw Figs 2-3 from re-run benchmark results (into output/figures_rerun)

Everything is written below output/ (logs in output/logs); results/ and data/ are only read.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent
OUT = PKG / "output"
LOGS = OUT / "logs"
PY = sys.executable
FIGURES = [  # manuscript figure, script, output file name
    ("Fig. 1", "fig1_pipeline.py", "Fig1_CatFORGE_pipeline"),
    ("Fig. 2", "fig2_synthetic.py", "Fig2_synthetic_fidelity_vs_usefulness"),
    ("Fig. 3", "fig3_benchmark.py", "Fig3_benchmark_heldout"),
    ("Fig. 4", "fig4_shap.py", "Fig4_interpretation_shap"),
    ("Fig. 5", "fig5_cost.py", "Fig5_cost_aware_ranking"),
    ("Fig. 6", "fig6_lab.py", "Fig6_laboratory_comparison"),
]
DATASETS = ("drm", "wgs", "ocm")

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # pragma: no cover
    pass


def banner(text):
    print("\n" + "=" * 100 + f"\n{text}\n" + "=" * 100, flush=True)


def child_env(**extra):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run(args, log, env=None, echo=False, prefix="    "):
    """Run a command from the package root, write its output to `log`, return (returncode, output lines)."""
    LOGS.mkdir(parents=True, exist_ok=True)
    lines = []
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.Popen(args, cwd=PKG, env=env or child_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            fh.write(line)
            fh.flush()
            lines.append(line.rstrip("\n"))
            if echo:
                print(prefix + line.rstrip("\n"), flush=True)
        p.wait()
    return p.returncode, lines


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(PKG))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------------------------------- figures
def step_figures(results_dir=None):
    only = None
    env_extra = {}
    if results_dir:
        rd = Path(results_dir).resolve()
        if not (rd / "augmentation_benchmark" / "benchmark_summary.csv").exists() and \
                not (rd / "predictor_benchmark" / "final_summary.csv").exists():
            sys.exit(f"--results-dir {results_dir}: no benchmark_summary.csv or final_summary.csv found there")
        only = {"Fig. 2", "Fig. 3"}
        env_extra = {"CATFORGE_RESULTS_DIR": rd, "CATFORGE_FIG_OUT": OUT / "figures_rerun"}
        banner(f"Figures 2-3 from re-run results in {rel(rd)} -> output/figures_rerun "
               "(the verification compares with the STORED reference numbers, so differences are reported)")
    else:
        banner("Regenerating Figs 1-6 from the stored results -> output/figures")
    shutil.rmtree(OUT / "cache", ignore_errors=True)          # fresh computation in every run
    status = {}
    for tgt, script, name in FIGURES:
        if only and tgt not in only:
            continue
        t0 = time.time()
        args = [PY, f"figures/{script}"]
        code, lines = run(args, LOGS / f"{Path(script).stem}.log", env=child_env(**env_extra))
        ver = next((l for l in lines if l.startswith("verification")), "")
        if tgt == "Fig. 1":
            ver = "; ".join(l.strip() for l in lines if l.startswith(("text overlaps", "clipped", "card/column")))
        err = "" if code == 0 else next((l for l in reversed(lines) if l.strip()), "")
        status[tgt] = {"script": script, "returncode": code, "seconds": round(time.time() - t0, 1),
                       "verification": ver, "error": err}
        print(f"  {tgt}  {'OK  ' if code == 0 else 'FAIL'}  {time.time() - t0:5.0f} s  figures/{script} -> "
              f"{name}.{{pdf,svg,png}}", flush=True)
        print(f"          {ver or err}", flush=True)
    (LOGS / ("figures_status.json" if not results_dir else "figures_rerun_status.json")).write_text(
        json.dumps(status, indent=1), encoding="utf-8")
    ok = all(s["returncode"] == 0 for s in status.values())
    print(f"\n  {'All figures regenerated' if ok else 'SOME FIGURES FAILED'}; logs in output/logs/", flush=True)
    return ok


# ---------------------------------------------------------------------------------------------------- check
def step_check():
    banner("Checking regenerated numbers against the stored references and the manuscript")
    code, _ = run([PY, "checks/check_results.py"], LOGS / "check.log", echo=True, prefix="")
    return code == 0


# ---------------------------------------------------------------------------------------------------- tests
def step_tests():
    banner("Cost-module tests and predictor-replica equivalence test")
    results = []
    t0 = time.time()
    code, lines = run([PY, "benchmark/test_cost_ranking.py"], LOGS / "test_cost_ranking.log")
    for l in lines:
        if l.startswith(("PASS", "FAIL")) or "checks passed" in l:
            print("    " + l.split("  [")[0], flush=True)
    new, ref = OUT / "tests" / "cost_ranking_test.json", PKG / "results" / "cost_module" / "cost_ranking_test.json"
    same = new.exists() and json.loads(new.read_text()) == json.loads(ref.read_text())
    results.append(("benchmark/test_cost_ranking.py (16 checks)", code == 0 and new.exists() and
                    json.loads(new.read_text())["n_pass"] == json.loads(new.read_text())["n_checks"]))
    results.append(("cost_ranking_test.json equals the stored results/cost_module/cost_ranking_test.json", same))
    print(f"    ({time.time() - t0:.0f} s)", flush=True)
    t0 = time.time()
    code, lines = run([PY, "benchmark/check_fig6_equivalence.py"], LOGS / "check_fig6_equivalence.log")
    for l in lines:
        print("    " + l, flush=True)
    results.append(("benchmark/check_fig6_equivalence.py (interface cost ranks Fig. 5 candidates identically)", code == 0))
    t0 = time.time()
    code, lines = run([PY, "benchmark/check_catforge_v2_equivalence.py"], LOGS / "check_catforge_v2_equivalence.log")
    shown = [l for l in lines if l.startswith(("drm:", "ocm:", "before repair", "after repair"))]
    for l in shown:
        print("    " + l, flush=True)
    diffs = [float(l.split("replica| = ")[1].split(";")[0]) for l in shown if "replica| =" in l]
    results.append(("benchmark/check_catforge_v2_equivalence.py: benchmark replica = platform predictor "
                    "(max difference 0 on DRM and OCM)", code == 0 and len(diffs) == 2 and max(diffs) == 0.0))
    # regression test behind the manuscript's "H2/CO predictions of -140 to -300" (docs/REPAIR_REPORT.md: -301.7 before
    # the repair, 0.62-0.69 after; RMSE 302 -> 0.19). The -140 value comes from the first-round benchmark (not re-run).
    reg = {}
    for l in shown:
        if l.startswith(("before repair", "after repair")):
            lo, hi = l.split("range ")[1].split(";")[0].split("..")
            reg[l.split(" (")[0]] = (float(lo), float(hi), float(l.split("RMSE ")[1]))
    ok_reg = (len(reg) == 2 and round(reg["before repair"][0], 1) == -301.7 and round(reg["before repair"][2]) == 302
              and (round(reg["after repair"][0], 2), round(reg["after repair"][1], 2)) == (0.62, 0.69)
              and round(reg["after repair"][2], 2) == 0.19)
    results.append(("regression test, DRM paper 2020-53 (pore volume 0): H2/CO -301.7 before repair, 0.62-0.69 after; "
                    "RMSE 302 -> 0.19 (as in docs/REPAIR_REPORT.md)", ok_reg))
    print(f"    ({time.time() - t0:.0f} s)", flush=True)
    print()
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
    return all(ok for _, ok in results)


# ---------------------------------------------------------------------------------------------------- quick
def step_quick():
    banner("QUICK SMOKE RUN of the benchmark code on a small subset -- this does NOT reproduce the manuscript numbers")
    q = OUT / "quick"
    shutil.rmtree(q, ignore_errors=True)
    env = child_env(CATFORGE_BENCH_OUT=q)
    t0 = time.time()
    print("  1/4 augmentation benchmark, DRM only: first of the 3 repeats (5 group folds), 50 real records, main grid "
          "(manuscript: 3 repeats, 25-400 records, three datasets, two grids)", flush=True)
    c1, _ = run([PY, "benchmark/run_benchmark.py", "--dataset", "drm", "--grid", "main", "--repeats", "1", "--folds",
                 "5", "--sizes", "50"], LOGS / "quick_augmentation.log", env=env)
    print(f"      {'done' if c1 == 0 else 'FAILED'} in {time.time() - t0:.0f} s", flush=True)
    t1 = time.time()
    print("  2/4 locked-test predictor evaluation, DRM only: first of the five 50-record training draws, all 12 models "
          "(manuscript: 50 / 200 / all records, 5 + 5 + 1 training sets, three datasets)", flush=True)
    c2, _ = run([PY, "benchmark/run_final.py", "--dataset", "drm", "--regimes", "50", "--draws", "1"],
                LOGS / "quick_predictor.log", env=env)
    print(f"      {'done' if c2 == 0 else 'FAILED'} in {time.time() - t1:.0f} s", flush=True)
    print("  3/4 analysis of the smoke predictions (bootstrap over locked test groups)", flush=True)
    c3, _ = run([PY, "benchmark/analyze_final.py"], LOGS / "quick_analysis.log", env=env)
    ok = c1 == 0 and c2 == 0 and c3 == 0
    same = False
    if ok:
        import numpy as np
        import pandas as pd
        s = pd.read_csv(q / "augmentation_benchmark" / "benchmark_summary.csv")
        print("\n  Smoke result, augmentation (change in held-out MAE vs real only, %; 5 training sets per cell):")
        print("    " + s[["dataset", "sampling", "size", "arm", "rel_delta_pct", "n_sets"]].round(2)
              .to_string(index=False).replace("\n", "\n    "))
        f = pd.read_csv(q / "predictor_benchmark" / "final_summary.csv", dtype={"regime": str})
        piv = f.pivot_table(index="model", columns="target", values="rmse").round(2)
        print("\n  Smoke result, locked test groups (RMSE, DRM, one training set of 50 records):")
        print("    " + piv.to_string().replace("\n", "\n    "))
        # 4/4: the smoke subset uses the same seeds as the matching part of the manuscript run
        print("\n  4/4 comparison with the matching subset of the stored manuscript run (same seeds):", flush=True)
        R = PKG / "results"
        a = pd.read_csv(q / "augmentation_benchmark" / "main_drm_raw.csv", dtype={"size": str})
        b = pd.read_csv(R / "augmentation_benchmark" / "main_drm_raw.csv", dtype={"size": str})
        keys = ["dataset", "grid", "sampling", "repeat", "fold", "size", "model", "target", "arm"]
        m = a.merge(b, on=keys, suffixes=("_q", "_s"))
        cols = ["mae", "rmse", "r2", "quality", "top1_regret"]      # what Fig. 2 and its numbers are built from

        def maxdiff(x, y):
            x, y = x.to_numpy(float), y.to_numpy(float)
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                return np.inf
            return float(np.nanmax(np.abs(x - y))) if np.isfinite(x).any() else 0.0
        da = max(maxdiff(m[c + "_q"], m[c + "_s"]) for c in cols) if len(m) == len(a) else np.inf
        ds_ = maxdiff(m["within_group_spearman_q"], m["within_group_spearman_s"]) if len(m) == len(a) else np.inf
        n_sp = int(((m["within_group_spearman_q"] - m["within_group_spearman_s"]).abs() > 1e-9).sum())
        a2 = pd.read_csv(q / "predictor_benchmark" / "final" / "drm.csv.gz", dtype={"regime": str})
        b2 = pd.read_csv(R / "predictor_benchmark" / "final" / "drm.csv.gz", dtype={"regime": str})
        b2 = b2[(b2.regime == "50") & (b2.repeat == 0)]
        m2 = a2.merge(b2, on=["regime", "repeat", "model", "target", "row"], suffixes=("_q", "_s"))
        dp = float((m2.pred_q - m2.pred_s).abs().max()) if len(m2) == len(a2) == len(b2) else np.inf
        tol = 1e-9
        print(f"      augmentation: {len(m)} of {len(a)} per-split results found in results/augmentation_benchmark/"
              f"main_drm_raw.csv; max |difference| in MAE, RMSE, R2, quality score and top-1 regret = {da:.2g} -> "
              f"{'MATCH' if da <= tol else 'DIFFERENT'}")
        print(f"                    (within-publication Spearman, not used in the manuscript: max |difference| {ds_:.2g} "
              f"in {n_sp} of {len(m)} results; random-forest predictions that differ in the 15th digit can swap "
              "near-tied ranks)")
        print(f"      locked test:  {len(m2)} of {len(a2)} predictions found in results/predictor_benchmark/final/drm.csv.gz "
              f"(50 records, draw 0); max |difference| = {dp:.2g} -> {'MATCH' if dp <= tol else 'DIFFERENT'}")
        same = da <= tol and dp <= tol
    print(f"\n  {'Smoke run completed' if ok else 'SMOKE RUN FAILED'} in {time.time() - t0:.0f} s. Outputs in output/quick/, "
          "logs in output/logs/quick_*.log.\n  The summary numbers above are NOT the manuscript's (a small subset of the "
          "design); the subset itself " + ("matches the stored run exactly." if same else "was compared with the stored "
                                           "run above."), flush=True)
    return ok


# ---------------------------------------------------------------------------------------------------- benchmarks
ESTIMATE = """\
  What runs (the exact protocol behind Figs 2-3 and Table 1; same seeds and settings as the stored results):
    a. augmentation benchmark, main grid     3 datasets x 15 group-held-out splits x 2 sampling schemes x 5 sizes,
                                              5 arms, random forest and XGBoost   (benchmark/run_benchmark.py)
    b. augmentation benchmark, ablation grid 3 datasets x 15 splits x 2 sizes; Gaussian copula, CTGAN, TVAE, CatFORGE
    c. locked-test predictor evaluation      3 datasets x (5 + 5 + 1) training sets x 12 models (benchmark/run_final.py)
    d. analysis                              run_benchmark.py --summarise-only and analyze_final.py, then comparison
                                              with the stored tables in results/
  Run times measured on the authors' laptop (Intel Core i7-1185G7, 4 cores / 8 threads, 2026):
    a + b: 1.0-2.0 h per dataset and grid when the six jobs ran in parallel (about 2 h wall-clock in total);
           one after another, roughly 4-9 h (estimate, not measured)
    c:     about 15-20 min per dataset (the three ran in parallel)
    d:     about 1 min
  Total: about 2.5 h with --jobs 6; roughly 5-10 h with --jobs 1 (the default; estimate).
  Not re-run here: the development steps that chose the repair (benchmark/run_sota.py, benchmark/run_ablation.py;
  several more hours, see README).
"""


def compare_tables(pairs):
    sys.path.insert(0, str(PKG / "checks"))
    import check_results as cr
    ok_all = True
    for new, ref in pairs:
        if not Path(new).exists():
            print(f"  [FAIL] {rel(new)} missing")
            ok_all = False
            continue
        ok, how = cr.same_table(new, ref)
        ok_all &= ok
        print(f"  [{'SAME' if ok else 'DIFF'}] {rel(new)} vs {rel(ref)}: {how}")
    return ok_all


def run_pool(jobs, n):
    """jobs: list of (label, args, log, env). Runs up to n at a time; returns {label: returncode}."""
    pending, running, codes = list(jobs), [], {}
    while pending or running:
        while pending and len(running) < n:
            label, args, log, env = pending.pop(0)
            LOGS.mkdir(parents=True, exist_ok=True)
            fh = open(log, "w", encoding="utf-8")
            p = subprocess.Popen(args, cwd=PKG, env=env, stdout=fh, stderr=subprocess.STDOUT)
            running.append((label, p, fh, time.time()))
            print(f"  started  {label}  (log: {rel(log)})", flush=True)
        time.sleep(5)
        for item in list(running):
            label, p, fh, t0 = item
            if p.poll() is not None:
                fh.close()
                running.remove(item)
                codes[label] = p.returncode
                print(f"  finished {label}: {'OK' if p.returncode == 0 else 'FAILED (exit ' + str(p.returncode) + ')'} "
                      f"after {(time.time() - t0) / 60:.0f} min", flush=True)
    return codes


def step_benchmarks(yes, n_jobs):
    banner("FULL RE-RUN of the augmentation and predictor benchmarks")
    print(ESTIMATE, flush=True)
    if not yes:
        if not sys.stdin.isatty():
            print("  Not started: no terminal to confirm. Re-run with --yes to start.")
            return None
        if input("  Start now? Type 'yes' to continue: ").strip().lower() != "yes":
            print("  Not started.")
            return None
    b = OUT / "benchmarks"
    env = child_env(CATFORGE_BENCH_OUT=b, CATFORGE_SKIP_SUMMARY=1)
    jobs = []
    for grid in ("main", "ablation"):
        for ds in DATASETS:
            jobs.append((f"augmentation {grid} {ds}", [PY, "benchmark/run_benchmark.py", "--dataset", ds, "--grid", grid,
                                                        "--resume"], LOGS / f"bench_augmentation_{grid}_{ds}.log", env))
    final = b / "predictor_benchmark" / "final"
    for ds in DATASETS:
        if (final / f"{ds}.csv.gz").exists():
            print(f"  skip locked-test {ds}: {rel(final / (ds + '.csv.gz'))} exists (the locked test is evaluated once; "
                  "delete the file to repeat)")
            continue
        jobs.append((f"locked-test predictor {ds}", [PY, "benchmark/run_final.py", "--dataset", ds],
                     LOGS / f"bench_predictor_{ds}.log", env))
    t0 = time.time()
    codes = run_pool(jobs, max(1, n_jobs))
    print(f"\n  benchmark jobs finished in {(time.time() - t0) / 3600:.1f} h", flush=True)
    c1, _ = run([PY, "benchmark/run_benchmark.py", "--summarise-only"], LOGS / "bench_summarise.log",
                env=child_env(CATFORGE_BENCH_OUT=b))
    c2, _ = run([PY, "benchmark/analyze_final.py"], LOGS / "bench_analyze_final.log", env=child_env(CATFORGE_BENCH_OUT=b))
    print("\n  Comparison of the re-run tables with the stored results:")
    R = PKG / "results"
    same = compare_tables([
        (b / "augmentation_benchmark" / "benchmark_summary.csv", R / "augmentation_benchmark" / "benchmark_summary.csv"),
        (b / "augmentation_benchmark" / "benchmark_fidelity_vs_utility.csv",
         R / "augmentation_benchmark" / "benchmark_fidelity_vs_utility.csv"),
        (b / "predictor_benchmark" / "final_summary.csv", R / "predictor_benchmark" / "final_summary.csv"),
        (b / "predictor_benchmark" / "final_catforge_vs_baselines.csv",
         R / "predictor_benchmark" / "final_catforge_vs_baselines.csv"),
        (b / "predictor_benchmark" / "final_headline.json", R / "predictor_benchmark" / "final_headline.json")])
    print("\n  To draw Figs 2-3 from these results: python reproduce.py --figures --results-dir output/benchmarks")
    ok = all(c == 0 for c in codes.values()) and c1 == 0 and c2 == 0
    print(f"  Re-run {'completed' if ok else 'had FAILURES (see output/logs)'}; tables "
          f"{'equal the stored results' if same else 'differ from the stored results (see above)'}.")
    return ok


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", action="store_true", help="regenerate Figs 1-6 into output/figures")
    ap.add_argument("--check", action="store_true", help="compare numbers with the stored references (runs --figures)")
    ap.add_argument("--tests", action="store_true", help="cost-module and predictor-replica tests")
    ap.add_argument("--quick", action="store_true", help="minutes-long smoke run of the benchmark code (not paper numbers)")
    ap.add_argument("--benchmarks", action="store_true", help="full benchmark re-run (hours)")
    ap.add_argument("--yes", action="store_true", help="with --benchmarks: start without asking")
    ap.add_argument("--jobs", type=int, default=1, help="with --benchmarks: number of parallel benchmark jobs")
    ap.add_argument("--results-dir", default=None, help="with --figures: draw Figs 2-3 from re-run results in this folder")
    a = ap.parse_args()
    if not any((a.figures, a.check, a.tests, a.quick, a.benchmarks)):
        ap.print_help()
        return 0
    if a.results_dir and (a.check or not a.figures):
        ap.error("--results-dir works with --figures only (the check compares the stored results)")
    print(f"CatFORGE reproducibility package | Python {sys.version.split()[0]} | package root: {PKG.name}")
    t0 = time.time()
    outcome = {}
    if a.figures or a.check:
        outcome["figures"] = step_figures(a.results_dir)
    if a.check:
        outcome["check"] = step_check()
    if a.tests:
        outcome["tests"] = step_tests()
    if a.quick:
        outcome["quick"] = step_quick()
    if a.benchmarks:
        outcome["benchmarks"] = step_benchmarks(a.yes, a.jobs)
    banner("Done in {:.1f} min: ".format((time.time() - t0) / 60) +
           ", ".join(f"{k} {'not started' if v is None else ('OK' if v else 'FAILED')}" for k, v in outcome.items()))
    if any(v is False for v in outcome.values()):
        return 1
    return 2 if any(v is None for v in outcome.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
