"""Analyse the one-shot locked-test evaluation (results/final/*.csv.gz).

For each dataset / regime / target: RMSE of every model with a 95% bootstrap over whole locked test groups, and the
paired ratio of catforge_v2 (and catforge_v1) to the best other model, with the same WIN / TIE / LOSS rule as before.
The ratio compares each CatFORGE version with the best of the *other* models, excluding the other CatFORGE version.

Package change: the analysis is unchanged, but the folders are arguments so that the stored locked-test predictions
(results/predictor_benchmark/final) can be re-analysed without overwriting the stored tables.
Usage: python analyze_final.py [--predictions DIR] [--out DIR]
  defaults: --predictions output/benchmarks/predictor_benchmark/final  --out output/benchmarks/predictor_benchmark"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_sota as az
import repro_paths as rp

DEFAULT_RES = rp.BENCH_OUT / "predictor_benchmark"


def main(final_dir, res_dir):
    final_dir, res_dir = Path(final_dir), Path(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2027)
    summaries, comps = [], []
    for ds in az.DATASETS:
        fn = final_dir / f"{ds}.csv.gz"
        if not fn.exists():
            continue
        df = pd.read_csv(fn, dtype={"regime": str})  # regime holds '50', '200' and 'full': keep every value a string
        df["row"] = df["row"].astype(int)
        for (regime, target), sub in df.groupby(["regime", "target"]):
            s, c2 = az.cell(ds, regime, target, sub, rng, focus="catforge_v2", exclude=("catforge_v1",))
            _, c1 = az.cell(ds, regime, target, sub, rng, focus="catforge_v1", exclude=("catforge_v2",))
            c2["focus"], c1["focus"] = "catforge_v2", "catforge_v1"
            summaries.append(s)
            comps.extend([c2, c1])
            print(ds, regime, target, flush=True)
    summary = pd.concat(summaries, ignore_index=True)
    comp = pd.concat(comps, ignore_index=True)
    summary.to_csv(res_dir / "final_summary.csv", index=False)
    comp.to_csv(res_dir / "final_catforge_vs_baselines.csv", index=False)
    head = {}
    for focus in ("catforge_v1", "catforge_v2"):
        for regime in ("full", "200", "50"):
            for base in ("best_other", "lgbm"):
                sel = comp[(comp["focus"] == focus) & (comp["regime"] == regime) & (comp["baseline"] == base)]
                head[f"{focus}|{regime}|vs_{base}"] = sel["verdict"].value_counts().to_dict()
    (res_dir / "final_headline.json").write_text(json.dumps(head, indent=2), encoding="utf-8")
    print(json.dumps(head, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(DEFAULT_RES / "final"))
    ap.add_argument("--out", default=str(DEFAULT_RES))
    a = ap.parse_args()
    main(a.predictions, a.out)
