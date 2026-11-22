"""Benchmark: does synthetic augmentation help predict catalyst performance for publications
or chemistries a model has never seen?

Generalises pilot_drm.py to three literature datasets:
  drm  dry reforming of methane; groups = publications
  wgs  low-temperature water-gas shift (CO conversion); groups = references
  ocm  oxidative coupling of methane; groups = element systems. The table records no
       publication, so a held-out group is a combination of elements the model has not seen.

and two ways of drawing the small real training set from the training groups:
  random     rows drawn at random across all training groups (as in the pilot)
  clustered  rows from the fewest randomly chosen whole groups that hold the requested
             number of rows, so the set comes from a few studies, as one laboratory's would

Leakage rules are those of the pilot: groups are split before anything is fitted; generators,
noise and predictors see only the sampled training rows; no outcome is used as an input.

Grids:
  main      arms real_only, noise_aug, gc_aug, catforge_aug, catforge_synth_only with 200
            synthetic rows per training set; sizes 25-400; both sampling schemes
  ablation  random sampling, sizes 50 and 200; synthetic rows as a multiple r of the real rows:
            GaussianCopula, CTGAN and TVAE at r = 1, CatFORGE at r = 0.5, 1, 2 and 4

Usage:
  python run_benchmark.py --dataset wgs --grid main
  python run_benchmark.py --dataset ocm --grid ablation --resume
  python run_benchmark.py --dataset drm --grid main --repeats 1 --folds 2 --sizes 50   # smoke test
  python run_benchmark.py --summarise-only
"""
import argparse
import hashlib
import importlib.util
import json
import logging
import platform
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, spearmanr
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# SDV/SDMetrics deprecation notices about metadata classes; they do not affect results.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sdv")

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
from sdmetrics.reports.single_table import QualityReport

# The CTGAN and TVAE networks are small; one thread is as fast as four and lets runs share the CPU.
torch.set_num_threads(1)

# Package paths (reproducibility package): data under data/, platform module under catforge/, re-runs under output/.
import repro_paths as rp  # noqa: E402

ROOT = rp.PKG
DRM_FILE = rp.DRM_FILE
WGS_FILE = rp.WGS_FILE
OCM_FILE = rp.OCM_FILE
SYNTH_MODULE = rp.CATFORGE / "data_synthesizer.py"
OUT = rp.BENCH_OUT / "augmentation_benchmark"

N_SYNTH = 200           # main grid: synthetic rows per training set, as in the historical CatFORGE runs
NOISE_SD = 0.05         # noise_aug: input noise as a fraction of each column's training SD
MIN_ROWS_PER_GROUP = 5  # within-group ranking metrics need at least this many test rows
QUALITY_PAIR_THRESHOLD = 0.5  # SDMetrics QualityReport default real-correlation threshold
CATFORGE_RATIOS = [0.5, 1, 2, 4]
MAIN_ARMS = ["real_only", "noise_aug", "gc_aug", "catforge_aug", "catforge_synth_only"]
ABLATION_ARMS = ["real_only", "gc_aug_r1", "ctgan_aug_r1", "tvae_aug_r1"] + [f"catforge_aug_r{r:g}" for r in CATFORGE_RATIOS]
GRIDS = {
    "main": {"sizes": [25, 50, 100, 200, 400], "sampling": ["random", "clustered"]},
    "ablation": {"sizes": [50, 200], "sampling": ["random"]},
}


@dataclass
class Dataset:
    name: str
    frame: pd.DataFrame  # inputs, targets and a "group" column
    inputs: list
    targets: list
    rank_targets: list   # targets where the best row is the one with the highest value
    source: Path
    group_unit: str
    notes: dict = field(default_factory=dict)

    @property
    def cols(self):
        return self.inputs + self.targets


def load_drm():
    # Ni Dispersion is dropped: 1,056 of 1,714 values are blank.
    inputs = [
        "Ratio of CH4 in Feed", "Reaction Temperature", "Ni Loading", "Reaction Time",
        "Pore Size", "Pore Volume", "Surface Area", "H2-TPR Peak Temperature",
        "Ni Particle Size", "Modifier Electronegativity", "GHSV",
    ]
    targets = ["CH4 Conversion", "CO2 Conversion", "Syngas_Ratio"]
    df = pd.read_csv(DRM_FILE).dropna(subset=inputs + targets).reset_index(drop=True)
    df["group"] = df["Paper no."].astype(str)
    # H2/CO is excluded from ranking metrics: higher is not simply better.
    return Dataset("drm", df[inputs + targets + ["group"]], inputs, targets,
                   ["CH4 Conversion", "CO2 Conversion"], DRM_FILE, "publication")


def load_wgs():
    raw = pd.read_csv(WGS_FILE)
    # Each reference name appears only on its first row; "Data" restarts at 1 on the same rows.
    raw["group"] = raw["Reference"].ffill().str.strip()
    target = "CO Conversion"
    inputs = [c for c in raw.columns if c not in ("Total # of Data", "Reference", "Data", "group", target)]
    df = raw.dropna(subset=inputs + [target]).reset_index(drop=True)
    return Dataset("wgs", df[inputs + [target, "group"]], inputs, [target], [target], WGS_FILE, "reference")


OCM_SLOTS = ["Cation 1", "Cation 2", "Cation 3", "Cation 4", "Anion 1", "Anion 2", "Support 1", "Support 2"]
# p(CH4)/p(O2) is left out: it is the ratio of two inputs already present.
OCM_CONDITIONS = ["Temperature, C", "p(CH4), bar", "p(O2), bar", "P total, bar", "Contact time, s"]
OCM_TARGETS = ["X(CH4), %", "S(C2), %", "Y(C2),%"]
OCM_MIN_ELEMENT_ROWS = 18  # about 1% of catalysts; rarer elements are pooled into one column


def load_ocm():
    raw = pd.read_csv(OCM_FILE)
    raw = raw.dropna(subset=["Cation 1 value"] + OCM_CONDITIONS + OCM_TARGETS).reset_index(drop=True)
    long = pd.concat([
        pd.DataFrame({"row": raw.index, "element": raw[f"{s} value"].str.strip(),
                      "mol": pd.to_numeric(raw[f"{s} mol%"], errors="coerce")})
        for s in OCM_SLOTS
    ]).dropna(subset=["element"])
    # Element-level composition: an element listed as both cation and support is summed.
    comp = long.pivot_table(index="row", columns="element", values="mol", aggfunc="sum", fill_value=0.0)
    comp = comp.reindex(raw.index, fill_value=0.0)
    common = sorted(e for e in comp.columns if (comp[e] > 0).sum() >= OCM_MIN_ELEMENT_ROWS)
    feats = comp[common].add_suffix(" mol%")
    feats["Other elements mol%"] = comp.drop(columns=common).sum(axis=1)
    feats["Cl promoter"] = raw["Pro- motor"].fillna("").str.contains("Cl").astype(int)
    df = pd.concat([feats, raw[OCM_CONDITIONS + OCM_TARGETS].apply(pd.to_numeric, errors="coerce")], axis=1)
    df["group"] = long.groupby("row")["element"].agg(lambda s: "-".join(sorted(set(s))))
    df = df.dropna().reset_index(drop=True)
    inputs = list(feats.columns) + OCM_CONDITIONS
    notes = {"elements_kept": common, "elements_pooled": sorted(set(comp.columns) - set(common))}
    return Dataset("ocm", df, inputs, OCM_TARGETS, OCM_TARGETS, OCM_FILE, "element system", notes)


LOADERS = {"drm": load_drm, "wgs": load_wgs, "ocm": load_ocm}


def load_catforge_run_synthesis():
    # run_synthesis logs every sampling batch at INFO; route it to a null handler.
    logging.basicConfig(level=logging.ERROR, handlers=[logging.NullHandler()])
    spec = importlib.util.spec_from_file_location("catforge_data_synthesizer", SYNTH_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_synthesis


def catforge_column_types(ds):
    # 0/1 indicator columns are declared discrete so CatFORGE rounds them back to 0/1;
    # all other columns are continuous, as in the pilot (the DRM table has no indicators).
    return {c: "discrete" if set(ds.frame[c].unique()) <= {0, 1} else "continuous" for c in ds.cols}


def catforge_augment(run_synthesis, ds, types, train, n_synth, seed, workdir):
    real_path = workdir / "real.csv"
    train[ds.cols].to_csv(real_path, index=False)
    config = {
        "original_file_path": str(real_path),
        "model_selected": "GaussianCopula",
        "synthesis_parameters": {"num_samples": n_synth},
        # Bounds default to the training rows' own range, as a user would set from the data.
        "column_configs": [
            {
                "name": c,
                "column_role": "Output" if c in ds.targets else "Input",
                "column_data_type": types[c],
                "min": float(train[c].min()),
                "max": float(train[c].max()),
            }
            for c in ds.cols
        ],
    }
    details = workdir / "details.json"
    details.write_text(json.dumps(config), encoding="utf-8")
    result = run_synthesis("benchmark", str(workdir), str(details), str(workdir / "synthesis.log"), seed=seed)
    if result.get("error"):
        raise RuntimeError(f"CatFORGE synthesis failed: {result['error']}")
    return pd.read_csv(result["output_path"])[ds.cols]


SDV_GENERATORS = {"gc": GaussianCopulaSynthesizer, "ctgan": CTGANSynthesizer, "tvae": TVAESynthesizer}


def sdv_augment(kind, ds, train, n_synth, seed):
    """Plain SDV generator with default settings (CTGAN and TVAE: 300 epochs)."""
    # Declared explicitly: auto-detection can mistake an all-unique column for a primary key.
    metadata = SingleTableMetadata.load_from_dict({"columns": {c: {"sdtype": "numerical"} for c in ds.cols}})
    np.random.seed(seed)
    torch.manual_seed(seed)
    synthesizer = SDV_GENERATORS[kind](metadata=metadata)
    synthesizer.fit(train[ds.cols])
    return synthesizer.sample(num_rows=n_synth)[ds.cols]


def noise_augment(ds, train, n_synth, rng):
    rows = train.iloc[rng.integers(0, len(train), n_synth)][ds.cols].reset_index(drop=True)
    sd = train[ds.inputs].std().fillna(0.0).to_numpy()
    rows[ds.inputs] = rows[ds.inputs] + rng.normal(0.0, 1.0, (n_synth, len(ds.inputs))) * sd * NOISE_SD
    rows[ds.inputs] = rows[ds.inputs].clip(train[ds.inputs].min(), train[ds.inputs].max(), axis=1)
    return rows


def sdmetrics_quality_score(ds, real, synthetic):
    metadata = {"columns": {c: {"sdtype": "numerical"} for c in ds.cols}}
    report = QualityReport()
    report.generate(real[ds.cols], synthetic[ds.cols], metadata, verbose=False)
    return report.get_score()


def pearson_matrix(x):
    """Pearson r between columns, undefined (NaN) for any pair involving a constant column.

    SDMetrics rejects constant columns. np.corrcoef instead returns rounding noise for a column
    that repeats a value with no exact binary form (e.g. 0.3), because its mean is off by an ulp.
    """
    c = np.corrcoef(x, rowvar=False)
    constant = np.ptp(x, axis=0) == 0
    c[constant, :] = np.nan
    c[:, constant] = np.nan
    return c


def quality_score(ds, real, synthetic):
    """SDMetrics QualityReport score for an all-numerical table, computed directly.

    The report averages Column Shapes (mean KS complement per column) and Column Pair Trends
    (mean of 1 - |r_real - r_synthetic| / 2, Pearson r, over the pairs whose real |r| exceeds
    the report's default threshold of 0.5), skipping a property that has no defined score.
    Looping over every pair makes the report take seconds per call on the 81-column WGS table;
    check_quality_score.py confirms this gives the same value.
    """
    r, s = real[ds.cols].to_numpy(float), synthetic[ds.cols].to_numpy(float)
    shapes = np.mean([1 - ks_2samp(r[:, j], s[:, j]).statistic for j in range(r.shape[1])])
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        upper = np.triu_indices(r.shape[1], 1)
        r_real, r_synth = pearson_matrix(r)[upper], pearson_matrix(s)[upper]
        strong = np.abs(np.nan_to_num(r_real)) > QUALITY_PAIR_THRESHOLD
        pairs = 1 - np.abs(r_real[strong] - r_synth[strong]) / 2
        trends = np.nanmean(pairs) if np.isfinite(pairs).any() else np.nan
    return float(np.nanmean([shapes, trends]))


def make_models(seed):
    return {
        "rf": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "xgb": XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=0,
        ),
    }


def within_group_metrics(ds, test, target, pred):
    """Mean Spearman rho and top-1 regret over held-out groups with enough rows."""
    rhos, regrets = [], []
    frame = pd.DataFrame({"group": test["group"].to_numpy(), "y": test[target].to_numpy(), "p": pred})
    for _, g in frame.groupby("group"):
        if len(g) < MIN_ROWS_PER_GROUP or g["y"].nunique() < 2 or g["p"].nunique() < 2:
            continue
        rho = spearmanr(g["y"], g["p"]).statistic
        if np.isfinite(rho):
            rhos.append(rho)
        # Regret: best measured value minus the measured value of the top-predicted row.
        if target in ds.rank_targets:
            regrets.append(g["y"].max() - g["y"].iloc[int(np.argmax(g["p"].to_numpy()))])
    return (
        float(np.mean(rhos)) if rhos else np.nan,
        float(np.mean(regrets)) if regrets else np.nan,
        len(rhos),
    )


def evaluate(ds, train_xy, test, seed):
    records = []
    for model_name, model in make_models(seed).items():
        for target in ds.targets:
            model.fit(train_xy[ds.inputs], train_xy[target])
            pred = model.predict(test[ds.inputs])
            y = test[target].to_numpy()
            err = pred - y
            rho, regret, n_rank_groups = within_group_metrics(ds, test, target, pred)
            records.append({
                "model": model_name, "target": target,
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
                "within_group_spearman": rho, "top1_regret": regret,
                "n_rank_groups": n_rank_groups,
            })
    return records


def split_groups(groups, n_folds, rng):
    shuffled = rng.permutation(groups)
    return [set(fold) for fold in np.array_split(shuffled, n_folds)]


def sample_training_set(pool, size, scheme, seed):
    if scheme == "random":
        return pool.sample(n=size, random_state=seed)
    # Clustered: whole groups in random order until they hold at least `size` rows,
    # then `size` rows from those groups.
    order = np.random.default_rng(seed).permutation(np.array(sorted(pool["group"].unique())))
    counts = pool["group"].value_counts()
    n_groups = int(np.searchsorted(np.cumsum(counts.loc[order].to_numpy()), size) + 1)
    chosen = pool[pool["group"].isin(order[:n_groups])]
    return chosen.sample(n=size, random_state=seed)


def augmented_sets(grid, ds, types, run_synthesis, train, seed):
    """Training data and synthetic-data quality for every arm of the grid."""
    real = train[ds.cols]
    if grid == "main":
        with tempfile.TemporaryDirectory() as tmp:
            cf_rows = catforge_augment(run_synthesis, ds, types, train, N_SYNTH, seed, Path(tmp))
        gc_rows = sdv_augment("gc", ds, train, N_SYNTH, seed)
        noise_rows = noise_augment(ds, train, N_SYNTH, np.random.default_rng(seed))
        q_cf, q_gc = quality_score(ds, train, cf_rows), quality_score(ds, train, gc_rows)
        return {
            "real_only": (real, 0, np.nan),
            "noise_aug": (pd.concat([real, noise_rows], ignore_index=True), N_SYNTH, np.nan),
            "gc_aug": (pd.concat([real, gc_rows], ignore_index=True), N_SYNTH, q_gc),
            "catforge_aug": (pd.concat([real, cf_rows], ignore_index=True), N_SYNTH, q_cf),
            "catforge_synth_only": (cf_rows, N_SYNTH, q_cf),
        }
    n = len(train)
    arms = {"real_only": (real, 0, np.nan)}
    for kind in SDV_GENERATORS:
        rows = sdv_augment(kind, ds, train, n, seed)
        arms[f"{kind}_aug_r1"] = (pd.concat([real, rows], ignore_index=True), n, quality_score(ds, train, rows))
    # One CatFORGE run at the largest ratio; each smaller ratio uses its leading rows.
    with tempfile.TemporaryDirectory() as tmp:
        cf_rows = catforge_augment(run_synthesis, ds, types, train, int(max(CATFORGE_RATIOS) * n), seed, Path(tmp))
    for r in CATFORGE_RATIOS:
        rows = cf_rows.iloc[: int(r * n)]
        arms[f"catforge_aug_r{r:g}"] = (pd.concat([real, rows], ignore_index=True), len(rows), quality_score(ds, train, rows))
    return arms


def raw_path(grid, name):
    return OUT / f"{grid}_{name}_raw.csv"


def run(args, name):
    OUT.mkdir(parents=True, exist_ok=True)
    ds = LOADERS[name]()
    types = catforge_column_types(ds)
    run_synthesis = load_catforge_run_synthesis()
    groups = np.array(sorted(ds.frame["group"].unique()))
    grid = GRIDS[args.grid]
    sizes = args.sizes or grid["sizes"]
    path = raw_path(args.grid, name)
    done_units = set()
    if path.exists() and args.resume:
        prev = pd.read_csv(path, dtype={"size": str})
        done_units = set(zip(prev["sampling"], prev["repeat"], prev["fold"], prev["size"]))
    elif path.exists():
        path.unlink()

    info = {
        "dataset": name, "grid": args.grid, "group_unit": ds.group_unit,
        "data_file": str(ds.source.relative_to(ROOT)),
        "data_sha256": hashlib.sha256(ds.source.read_bytes()).hexdigest(),
        "rows_used": len(ds.frame), "groups": len(groups),
        "exact_duplicate_rows": int(ds.frame.duplicated(subset=ds.cols).sum()),
        "inputs": ds.inputs, "targets": ds.targets,
        "catforge_discrete_columns": [c for c, t in types.items() if t == "discrete"],
        "arms": MAIN_ARMS if args.grid == "main" else ABLATION_ARMS,
        "sampling": grid["sampling"], "sizes": sizes, "folds": args.folds, "repeats": args.repeats,
        "n_synth": N_SYNTH if args.grid == "main" else "ratio x real rows",
        "noise_sd": NOISE_SD, "seed": args.seed, **ds.notes,
        "python": platform.python_version(),
        "packages": {m: sys.modules[m].__version__ for m in ["numpy", "pandas", "sklearn", "xgboost", "sdv", "sdmetrics", "torch"] if m in sys.modules},
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"[{name}/{args.grid}] {len(ds.frame)} rows, {len(groups)} {ds.group_unit}s, {len(ds.inputs)} inputs, "
          f"{len(info['catforge_discrete_columns'])} discrete; resuming past {len(done_units)} units", flush=True)

    def write(rows):
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)

    units = [(s, rep, fold, size) for rep in range(args.repeats) for fold in range(args.folds)
             for s in grid["sampling"] for size in sizes]
    total, done, failures, t0 = len(units), 0, [], time.time()
    for rep in range(args.repeats):
        rng = np.random.default_rng(args.seed + rep)
        for fold, test_groups in enumerate(split_groups(groups, args.folds, rng)):
            test =ds.frame[ds.frame["group"].isin(test_groups)]
            pool = ds.frame[~ds.frame["group"].isin(test_groups)]
            split_meta = {"dataset": name, "grid": args.grid, "repeat": rep, "fold": fold,
                          "n_test_rows": len(test), "n_test_groups": len(test_groups)}

            # Reference: all real training rows, no augmentation.
            if args.grid == "main" and ("all", rep, fold, "all") not in done_units:
                rows = evaluate(ds, pool, test, args.seed)
                for rec in rows:
                    rec.update(split_meta, sampling="all", size="all", n_train=len(pool),
                               n_train_groups=pool["group"].nunique(), arm="real_only", n_synth=0, quality=np.nan)
                write(rows)

            for scheme in grid["sampling"]:
                for size in sizes:
                    done += 1
                    if (scheme, rep, fold, str(size)) in done_units:
                        continue
                    seed = args.seed + 1000 * rep + 100 * fold + size
                    train = sample_training_set(pool, size, scheme, seed)
                    try:
                        arms = augmented_sets(args.grid, ds, types, run_synthesis, train, seed)
                    except Exception as e:  # noqa: BLE001 - recorded, and the whole set is skipped
                        # Skip the whole training set so every arm stays paired on the same splits.
                        failures.append({"sampling": scheme, "repeat": rep, "fold": fold, "size": size, "error": repr(e)[:300]})
                        print(f"  synthesis failed ({scheme} rep {rep} fold {fold} n={size}): {e!r}"[:300], flush=True)
                        continue
                    rows = []
                    for arm, (train_xy, n_synth, q) in arms.items():
                        for rec in evaluate(ds, train_xy, test, seed):
                            rec.update(split_meta, sampling=scheme, size=str(size), n_train=size,
                                       n_train_groups=train["group"].nunique(), arm=arm, n_synth=n_synth, quality=q)
                            rows.append(rec)
                    write(rows)
                    elapsed = time.time() - t0
                    print(f"  [{done}/{total}] {scheme} rep {rep} fold {fold} n={size} "
                          f"({train['group'].nunique()} groups)  {elapsed / 60:.1f} min", flush=True)

    info["synthesis_failures"] = failures
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT / f"{args.grid}_{name}_run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def bootstrap_ci(values, rng, n_boot=2000):
    boots = rng.choice(values, (n_boot, len(values))).mean(axis=1)
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def summarise():
    """Paired comparison of every arm against real_only on identical training sets."""
    keys = ["dataset", "grid", "sampling", "repeat", "fold", "size", "model", "target"]
    raws = [pd.read_csv(p, dtype={"size": str}) for p in sorted(OUT.glob("*_raw.csv")) if not p.name.startswith("pilot")]
    if not raws:
        print("No benchmark results yet.")
        return
    raw = pd.concat(raws, ignore_index=True)
    ref = raw[raw["size"] == "all"]
    raw = raw[raw["size"] != "all"].copy()
    raw["size"] = raw["size"].astype(int)
    base = raw[raw["arm"] == "real_only"].set_index(keys)
    rng = np.random.default_rng(0)

    per_target, per_set = [], []
    for arm, g in raw[raw["arm"] != "real_only"].groupby("arm"):
        g = g.set_index(keys)
        b = base.loc[g.index]
        g = g.assign(delta=g["mae"] - b["mae"], rel=100 * (g["mae"] - b["mae"]) / b["mae"],
                     d_rho=g["within_group_spearman"] - b["within_group_spearman"],
                     d_regret=g["top1_regret"] - b["top1_regret"]).reset_index()
        per_target.append(g)
    deltas = pd.concat(per_target, ignore_index=True)

    rows = []
    for (dataset, grid, sampling, size, arm, model, target), g in deltas.groupby(
            ["dataset", "grid", "sampling", "size", "arm", "model", "target"]):
        lo, hi = bootstrap_ci(g["delta"].to_numpy(), rng)
        rows.append({
            "dataset": dataset, "grid": grid, "sampling": sampling, "size": size, "arm": arm,
            "model": model, "target": target, "mae": g["mae"].mean(), "r2": g["r2"].mean(),
            "delta_mae_vs_real": g["delta"].mean(), "delta_ci_low": lo, "delta_ci_high": hi,
            "rel_delta_pct": g["rel"].mean(), "win_rate_vs_real": float(np.mean(g["delta"] < 0)),
            "within_group_spearman": g["within_group_spearman"].mean(), "delta_spearman": g["d_rho"].mean(),
            "top1_regret": g["top1_regret"].mean(), "delta_regret": g["d_regret"].mean(),
            "quality": g["quality"].mean(), "n_splits": len(g),
        })
    pd.DataFrame(rows).to_csv(OUT / "benchmark_summary_by_target.csv", index=False)

    # Headline: one relative change per training set (mean over predictors and targets),
    # so each set counts once and every target weighs the same.
    split_keys = ["dataset", "grid", "sampling", "repeat", "fold", "size", "arm"]
    sets = deltas.groupby(split_keys).agg(rel=("rel", "mean"), quality=("quality", "first"),
                                          d_rho=("d_rho", "mean"), n_train_groups=("n_train_groups", "first")).reset_index()
    for (dataset, grid, sampling, size, arm), g in sets.groupby(["dataset", "grid", "sampling", "size", "arm"]):
        lo, hi = bootstrap_ci(g["rel"].to_numpy(), rng)
        per_set.append({"dataset": dataset, "grid": grid, "sampling": sampling, "size": size, "arm": arm,
                        "rel_delta_pct": g["rel"].mean(), "ci_low": lo, "ci_high": hi,
                        "win_rate_vs_real": float(np.mean(g["rel"] < 0)), "delta_spearman": g["d_rho"].mean(),
                        "quality": g["quality"].mean(), "train_groups": g["n_train_groups"].mean(), "n_sets": len(g)})
    headline = pd.DataFrame(per_set)
    headline.to_csv(OUT / "benchmark_summary.csv", index=False)

    # Does the synthetic-data quality score predict whether augmentation helps?
    # Correlated within each training-set size so the shared trend with size cannot drive it.
    fidelity = []
    for (dataset, grid, sampling, arm), g in sets.dropna(subset=["quality"]).groupby(["dataset", "grid", "sampling", "arm"]):
        if arm == "catforge_synth_only":
            continue
        rhos = [spearmanr(s["quality"], s["rel"]).statistic for _, s in g.groupby("size") if len(s) > 2]
        fidelity.append({"dataset": dataset, "grid": grid, "sampling": sampling, "arm": arm,
                         "mean_within_size_spearman": float(np.nanmean(rhos)) if rhos else np.nan,
                         "quality_mean": g["quality"].mean(), "n_training_sets": len(g)})
    pd.DataFrame(fidelity).to_csv(OUT / "benchmark_fidelity_vs_utility.csv", index=False)

    ceiling = ref.groupby(["dataset", "target", "model"])[["mae", "r2", "n_train"]].mean().reset_index()
    ceiling.to_csv(OUT / "benchmark_ceiling_all_real.csv", index=False)

    # Learning curves: each arm's error relative to the model trained on every real training row
    # of the same split, so augmentation's effect can be set against that of more real data.
    split = ["dataset", "grid", "repeat", "fold", "model", "target"]
    curves = raw[raw["grid"] == "main"].join(ref.set_index(split)["mae"].rename("mae_all_real"), on=split)
    curves["rel_all"] = 100 * (curves["mae"] / curves["mae_all_real"] - 1)
    learning = []
    for (dataset, sampling, size, arm), g in curves.groupby(split_keys)["rel_all"].mean().reset_index().groupby(
            ["dataset", "sampling", "size", "arm"]):
        lo, hi = bootstrap_ci(g["rel_all"].to_numpy(), rng)
        learning.append({"dataset": dataset, "sampling": sampling, "size": size, "arm": arm,
                         "rel_to_all_real_pct": g["rel_all"].mean(), "ci_low": lo, "ci_high": hi, "n_sets": len(g)})
    pd.DataFrame(learning).to_csv(OUT / "benchmark_learning_curves.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_rows", 500, "display.precision", 2):
        print(headline.to_string(index=False))
        print(pd.DataFrame(fidelity).to_string(index=False))
        print(ceiling.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(LOADERS) + ["all"], default="all")
    parser.add_argument("--grid", choices=list(GRIDS), default="main")
    parser.add_argument("--sizes", type=int, nargs="+", help="override the grid's training-set sizes")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="keep existing results and skip finished training sets")
    parser.add_argument("--summarise-only", action="store_true")
    args = parser.parse_args()
    if not args.summarise_only:
        for name in (list(LOADERS) if args.dataset == "all" else [args.dataset]):
            run(args, name)
        # Package addition: when reproduce.py runs several datasets in parallel it summarises once at the end, so a
        # run does not read another run's half-written raw file.
        if __import__("os").environ.get("CATFORGE_SKIP_SUMMARY") == "1":
            sys.exit(0)
    summarise()
