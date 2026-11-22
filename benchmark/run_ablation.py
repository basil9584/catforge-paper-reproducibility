"""Development run: which parts of the CatFORGE predictor help, which hurt?

DESIGN (fixed before running)
  Locked test set  30% of the groups of each dataset are set aside by a seeded shuffle and are NOT used here.
                   They are touched once, by run_final.py, after the repaired predictor is chosen.
                   DRM paper 2020-53 (the failure found in the first benchmark) is kept in the development
                   groups, because it motivated the repair and so cannot count as an independent test.
  Development CV   the other 70% of groups, grouped 5-fold, one repeat; regimes 'full' and 200 rows.
  Variants         see catforge_variants.VARIANTS (feature pipeline) x combinations of fitted members.
  References       tuned LightGBM, XGBoost, random forest, extra trees on the raw inputs.

Usage: python run_ablation.py --dataset drm   (resumable)"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import catforge_variants as cv
import run_sota as rs

HERE = Path(__file__).resolve().parent
OUT = rs.rp.BENCH_OUT / "predictor_development" / "ablation"     # package: re-runs write below output/
TEST_FRAC, SPLIT_SEED, FORCE_DEV = 0.30, 4242, {"drm": ["2020-53"]}


def locked_split(ds):
    groups = sorted(ds.frame["group"].unique())
    order = list(np.random.default_rng(SPLIT_SEED).permutation(groups))
    n_test = int(round(TEST_FRAC * len(groups)))
    test = order[:n_test]
    for g in FORCE_DEV.get(ds.name, []):
        if g in test:
            test.remove(g)
            test.append(next(x for x in order[n_test:] if x not in FORCE_DEV[ds.name]))
    return sorted(test), sorted(set(groups) - set(test))


def run(args):
    ds = rs.rb.LOADERS[args.dataset]()
    df = ds.frame.reset_index(drop=True)
    test_groups, dev_groups = locked_split(ds)
    variants = cv.VARIANTS if args.round == 1 else cv.VARIANTS2
    out = OUT / ("dev" if args.round == 1 else "dev2") / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    (OUT / "split_" f"{args.dataset}.json").write_text(json.dumps({
        "dataset": args.dataset, "locked_test_groups": test_groups, "dev_groups": len(dev_groups),
        "split_seed": SPLIT_SEED, "test_frac": TEST_FRAC, "locked_test_rows": int(df["group"].isin(test_groups).sum()),
        "dev_rows": int((~df["group"].isin(test_groups)).sum())}, indent=1), encoding="utf-8")
    dev = df[~df["group"].isin(test_groups)]
    parts = rs.partition(dev["group"].unique(), 5, 5000)
    for k in [int(x) for x in args.folds.split(",")]:
        fn = out / f"fold{k}.csv.gz"
        if fn.exists():
            continue
        t0 = time.time()
        test = dev[dev["group"].isin(parts[k])]
        pool = dev[~dev["group"].isin(parts[k])]
        recs = []
        for regime in ("full", 200):
            seed = 100 * k + (0 if regime == "full" else regime)
            train = pool if regime == "full" else pool.sample(n=min(regime, len(pool)), random_state=seed)
            Xtr, Xte, gtr = train[ds.inputs], test[ds.inputs], train["group"]
            preds = {}
            for vname, variant in variants.items():
                for combo, p in variant.fit_predict(Xtr, train[ds.targets], gtr, Xte, seed).items():
                    preds[f"{vname}|{combo}"] = p
            # Round 2 reuses the (seed-identical) reference fits stored by round 1.
            for name in (("lgbm", "xgb", "rf", "et") if args.round == 1 else ()):
                preds[f"ref|{name}"] = pd.DataFrame({t: rs.bound(rs.fit_predict_baseline(name, seed, Xtr, train[t], gtr, Xte, t, 2), t, train[t])
                                                     for t in ds.targets}, index=test.index)
            for model, p in preds.items():
                for t in ds.targets:
                    recs.append(pd.DataFrame({"regime": str(regime), "fold": k, "model": model, "target": t,
                                              "row": test.index.to_numpy(), "group": test["group"].to_numpy(),
                                              "y": test[t].to_numpy(), "pred": p[t].to_numpy()}))
        pd.concat(recs).to_csv(fn, index=False, compression="gzip")
        print(f"[{args.dataset}] dev fold {k} done in {time.time() - t0:.0f}s (train rows {len(pool)}, test rows {len(test)})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(rs.rb.LOADERS))
    ap.add_argument("--round", type=int, default=1, choices=[1, 2])
    ap.add_argument("--folds", default="0,1,2,3,4")
    run(ap.parse_args())
