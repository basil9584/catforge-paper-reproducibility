"""ONE-SHOT evaluation on the locked test groups (30% of the groups of each dataset; see run_ablation.locked_split).

Nothing in the ablation or in the repair saw these groups. Every model is trained on the development groups only
(all rows, or 200 / 50 rows drawn at random from them; 5 independent draws for the two small regimes) and
evaluated on every record of the locked groups.

Models: catforge_v1 (predictor as shipped before the repair), catforge_v2 (repaired predictor), and the ten baselines of
run_sota.py. Usage: python run_final.py --dataset drm

Package additions (defaults reproduce the manuscript run): --out (where predictions are written; default
output/benchmarks/predictor_benchmark/final) and --regimes / --draws (only for the short smoke run of
reproduce.py --quick, which does NOT reproduce the manuscript numbers)."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_ablation as ra
import run_sota as rs
from catforge_v2 import CatForgeV2

HERE = Path(__file__).resolve().parent
OUT = rs.rp.BENCH_OUT / "predictor_benchmark" / "final"     # package: re-runs write below output/
BASELINES = [m for m in rs.MODELS if m != "catforge"]
DRAWS = {"50": 5, "200": 5, "full": 1}


def run(args):
    global OUT
    if args.out:
        OUT = Path(args.out).resolve()
    regimes = [r for r in ("50", "200", "full") if r in args.regimes.split(",")]
    draws = {r: (min(DRAWS[r], args.draws) if args.draws else DRAWS[r]) for r in regimes}
    ds = rs.rb.LOADERS[args.dataset]()
    df = ds.frame.reset_index(drop=True)
    test_groups, _ = ra.locked_split(ds)
    pool, test = df[~df["group"].isin(test_groups)], df[df["group"].isin(test_groups)]
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / f"{args.dataset}.csv.gz"
    assert not fn.exists(), "the locked test set is evaluated once; delete the file deliberately to repeat"
    patched = (rs.PRED_DIR / "generalized_predictor.py").read_bytes()
    info = {"dataset": args.dataset, "dev_rows": len(pool), "locked_test_rows": len(test), "locked_test_groups": len(test_groups),
            "predictor_file_sha256": hashlib.sha256(patched).hexdigest(), "draws": draws, "models": ["catforge_v1", "catforge_v2"] + BASELINES}
    (OUT / f"{args.dataset}_run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    recs = []
    for regime in regimes:
        for draw in range(draws[regime]):
            t0 = time.time()
            seed = 7000 + 10 * draw + (0 if regime == "full" else int(regime))
            train = pool if regime == "full" else pool.sample(n=int(regime), random_state=seed)
            Xtr, Xte, gtr = train[ds.inputs], test[ds.inputs], train["group"]
            preds = {"catforge_v1": rs.CatForgeModel().fit(Xtr, train[ds.targets]).predict(Xte),
                     "catforge_v2": CatForgeV2().fit(Xtr, train[ds.targets]).predict(Xte)}
            for name in BASELINES:
                preds[name] = pd.DataFrame({t: rs.bound(rs.fit_predict_baseline(name, seed, Xtr, train[t], gtr, Xte, t, 2), t, train[t])
                                            for t in ds.targets}, index=test.index)
            for model, p in preds.items():
                for t in ds.targets:
                    recs.append(pd.DataFrame({"regime": regime, "repeat": draw, "fold": 0, "model": model, "target": t,
                                              "row": test.index.to_numpy(), "group": test["group"].to_numpy(),
                                              "y": test[t].to_numpy(), "pred": p[t].to_numpy()}))
            print(f"[{args.dataset}] regime {regime} draw {draw} done in {time.time() - t0:.0f}s", flush=True)
    pd.concat(recs).to_csv(fn, index=False, compression="gzip")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(rs.rb.LOADERS))
    ap.add_argument("--out", default=None, help="folder for the predictions (package addition)")
    ap.add_argument("--regimes", default="50,200,full", help="smoke runs only; the manuscript used all three")
    ap.add_argument("--draws", type=int, default=0, help="smoke runs only: at most this many draws per small regime")
    run(ap.parse_args())
