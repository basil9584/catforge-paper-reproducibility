"""Recompute the synthetic-data quality column of a finished benchmark run.

The first runs scored quality with np.corrcoef directly, which returns rounding noise instead
of an undefined correlation for a column that repeats a value with no exact binary form, so
a few scores differed from SDMetrics (see run_benchmark.pearson_matrix). Synthesis is
deterministic for a given seed, so this regenerates every training set's synthetic rows as
run_benchmark.run() does, checks that the earlier formula reproduces each stored score, and
rewrites the quality column with the corrected score. No other column changes.

Usage: python recompute_quality.py --dataset drm --grid main
Writes results/quality_recompute_{grid}_{dataset}.csv (stored, reproduced and corrected scores).

Reproducibility package: included for provenance only. It was applied once to the first runs (2 Nov 2026), so the
stored raw files in results/augmentation_benchmark/ already carry the corrected scores, and a fresh run of
run_benchmark.py computes the corrected score directly. It works on the files of a re-run in
output/benchmarks/augmentation_benchmark/ (run_benchmark.OUT), never on the stored results.
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import run_benchmark as rb

KEYS = ["sampling", "repeat", "fold", "size", "arm"]


def first_run_quality_score(ds, real, synthetic):
    """The quality formula the first runs used (np.corrcoef without the constant-column check)."""
    r, s = real[ds.cols].to_numpy(float), synthetic[ds.cols].to_numpy(float)
    shapes = np.mean([1 - ks_2samp(r[:, j], s[:, j]).statistic for j in range(r.shape[1])])
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        upper = np.triu_indices(r.shape[1], 1)
        r_real, r_synth = np.corrcoef(r, rowvar=False)[upper], np.corrcoef(s, rowvar=False)[upper]
        strong = np.abs(np.nan_to_num(r_real)) > rb.QUALITY_PAIR_THRESHOLD
        pairs = 1 - np.abs(r_real[strong] - r_synth[strong]) / 2
        trends = np.nanmean(pairs) if np.isfinite(pairs).any() else np.nan
    return float(np.nanmean([shapes, trends]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(rb.LOADERS), required=True)
    parser.add_argument("--grid", choices=list(rb.GRIDS), required=True)
    args = parser.parse_args()
    name, grid = args.dataset, args.grid

    info_path = rb.OUT / f"{grid}_{name}_run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    path = rb.raw_path(grid, name)
    raw = pd.read_csv(path, dtype={"size": str})

    ds = rb.LOADERS[name]()
    types = rb.catforge_column_types(ds)
    run_synthesis = rb.load_catforge_run_synthesis()
    groups = np.array(sorted(ds.frame["group"].unique()))

    # augmented_sets() looks quality_score up at call time, so this records both formulas
    # on the same synthetic rows without changing how they are generated.
    corrected = rb.quality_score
    rb.quality_score = lambda d, real, synth: (first_run_quality_score(d, real, synth), corrected(d, real, synth))

    # Same loop, seeds and order as run_benchmark.run().
    records, t0 = [], time.time()
    total = info["repeats"] * info["folds"] * len(info["sampling"]) * len(info["sizes"])
    done = 0
    for rep in range(info["repeats"]):
        rng = np.random.default_rng(info["seed"] + rep)
        for fold, test_groups in enumerate(rb.split_groups(groups, info["folds"], rng)):
            pool = ds.frame[~ds.frame["group"].isin(test_groups)]
            for scheme in info["sampling"]:
                for size in info["sizes"]:
                    done += 1
                    seed = info["seed"] + 1000 * rep + 100 * fold + size
                    train = rb.sample_training_set(pool, size, scheme, seed)
                    try:
                        arms = rb.augmented_sets(grid, ds, types, run_synthesis, train, seed)
                    except Exception as e:  # noqa: BLE001 - the run skipped this set too
                        print(f"  synthesis failed ({scheme} rep {rep} fold {fold} n={size}): {e!r}"[:300], flush=True)
                        continue
                    for arm, (_, _, q) in arms.items():
                        if isinstance(q, tuple):
                            records.append({"sampling": scheme, "repeat": rep, "fold": fold, "size": str(size),
                                            "arm": arm, "reproduced": q[0], "corrected": q[1]})
                    print(f"  [{done}/{total}] {scheme} rep {rep} fold {fold} n={size}  "
                          f"{(time.time() - t0) / 60:.1f} min", flush=True)

    scores = pd.DataFrame(records)
    stored = raw.dropna(subset=["quality"]).groupby(KEYS, as_index=False)["quality"].agg(["first", "nunique"])
    if (stored["nunique"] > 1).any():
        raise SystemExit("a training set has more than one stored quality per arm")
    scores = scores.merge(stored.rename(columns={"first": "stored"})[KEYS + ["stored"]], on=KEYS, how="outer")
    scores["reproduction_error"] = (scores["stored"] - scores["reproduced"]).abs()
    scores["correction"] = scores["corrected"] - scores["stored"]
    scores.to_csv(rb.OUT / f"quality_recompute_{grid}_{name}.csv", index=False)

    unmatched = scores["stored"].isna() | scores["reproduced"].isna()
    max_err = scores["reproduction_error"].max()
    print(f"{len(scores)} scored synthetic sets; {int(unmatched.sum())} without a stored or reproduced score; "
          f"max |stored - reproduced| = {max_err:.2e}; "
          f"{int((scores['correction'].abs() > 1e-12).sum())} corrected, "
          f"max |correction| = {scores['correction'].abs().max():.4f}")
    if unmatched.any() or max_err > 1e-12:
        raise SystemExit("synthesis did not reproduce the stored scores; raw file left unchanged")

    new = raw.merge(scores[KEYS + ["corrected"]], on=KEYS, how="left")
    has_score = raw["quality"].notna().to_numpy()
    raw.loc[has_score, "quality"] = new.loc[has_score, "corrected"].to_numpy()
    tmp = path.with_suffix(".tmp")
    raw.to_csv(tmp, index=False)
    tmp.replace(path)
    info["quality_recomputed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
