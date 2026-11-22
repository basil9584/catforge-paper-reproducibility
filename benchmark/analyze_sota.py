"""Analyse results/sota: held-out-group error of the CatFORGE predictor versus the baselines.

Verdict rule (fixed in run_sota.py before any result was seen): for each dataset / regime / target, the
paired group-bootstrap 95% CI of RMSE_catforge / RMSE_baseline. WIN = CI entirely below 1, LOSS = CI
entirely above 1, otherwise TIE. 'best_other' is the baseline with the lowest RMSE in that cell.

Usage: python analyze_sota.py [--allow-partial]
Writes results/sota_summary.csv, results/sota_catforge_vs_baselines.csv and results/sota_headline.json."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import repro_paths as rp  # noqa: E402  (package paths)

HERE = Path(__file__).resolve().parent
RES = rp.BENCH_OUT / "predictor_first_round"     # package: first-round re-runs live below output/
SOTA = RES / "sota"
DATASETS = ["drm", "wgs", "ocm"]
RANK_TARGETS = {"drm": ["CH4 Conversion", "CO2 Conversion"], "wgs": ["CO Conversion"],
                "ocm": ["X(CH4), %", "S(C2), %", "Y(C2),%"]}
B = 2000
MIN_ROWS = 5


def load(ds, allow_partial):
    info = json.loads((SOTA / ds / "run_info.json").read_text(encoding="utf-8"))
    frames = []
    for f in sorted((SOTA / ds).glob("rep*_fold*.csv.gz")):
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    folds = df.groupby("repeat")["fold"].nunique()
    complete = [r for r, n in folds.items() if n == info["folds"]]
    if not allow_partial:
        df = df[df["repeat"].isin(complete)]
    return df, info, complete


def within_group(y, p, groups, rank):
    rhos, regrets = [], []
    for g in groups:
        yy, pp = y[g], p[g]
        if yy.std() == 0 or pp.std() == 0:
            continue
        rho = spearmanr(yy, pp).statistic
        if np.isfinite(rho):
            rhos.append(rho)
        if rank:
            regrets.append(yy.max() - yy[int(np.argmax(pp))])
    return (np.mean(rhos) if rhos else np.nan), (np.mean(regrets) if regrets else np.nan)


def cell(ds, regime, target, sub, rng, focus="catforge", exclude=()):
    models = sorted(sub["model"].unique())
    reps = sorted(sub["repeat"].unique())
    rows = np.sort(sub["row"].unique())
    pos = {r: i for i, r in enumerate(rows)}
    y = np.full(len(rows), np.nan)
    grp = np.empty(len(rows), dtype=object)
    P = np.full((len(models), len(reps), len(rows)), np.nan)
    for (m, r), g in sub.groupby(["model", "repeat"]):
        idx = g["row"].map(pos).to_numpy()
        P[models.index(m), reps.index(r), idx] = g["pred"].to_numpy()
        y[idx] = g["y"].to_numpy()
        grp[idx] = g["group"].to_numpy()
    keep = ~np.isnan(P).any(axis=(0, 1)) & ~np.isnan(y)
    P, y, grp = P[:, :, keep], y[keep], grp[keep]
    SE = (P - y) ** 2
    ug, gi = np.unique(grp, return_inverse=True)
    G = len(ug)
    S = np.zeros((len(models), len(reps), G))
    for m in range(len(models)):
        for r in range(len(reps)):
            S[m, r] = np.bincount(gi, weights=SE[m, r], minlength=G)
    N = np.bincount(gi, minlength=G).astype(float)
    rmse = np.sqrt(SE.mean(axis=2)).mean(axis=1)
    mae = np.abs(P - y).mean(axis=2).mean(axis=1)
    sst = ((y - y.mean()) ** 2).mean()
    r2 = (1 - SE.mean(axis=2) / sst).mean(axis=1)
    # Group bootstrap: resample whole publications / references / element systems with replacement; the same
    # resample is applied to every model and repeat, so differences between models are paired.
    boots = np.empty((B, len(models)))
    for b in range(B):
        c = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
        boots[b] = np.sqrt((S @ c) / (N @ c)).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    members = [np.where(gi == k)[0] for k in range(G)]
    members = [m for m in members if len(m) >= MIN_ROWS]
    rank = target in RANK_TARGETS[ds]
    wg = []
    for m in range(len(models)):
        vals = [within_group(y, P[m, r], members, rank) for r in range(len(reps))]
        wg.append((np.nanmean([v[0] for v in vals]), np.nanmean([v[1] for v in vals])))
    summ = pd.DataFrame({"dataset": ds, "regime": regime, "target": target, "model": models, "rmse": rmse, "rmse_lo": lo,
                         "rmse_hi": hi, "mae": mae, "r2": r2, "within_group_spearman": [w[0] for w in wg],
                         "top1_regret": [w[1] for w in wg], "n_rows": len(y), "n_groups": G, "n_repeats": len(reps)})
    # paired bootstrap of CatFORGE against each other model
    comp = []
    ci = models.index(focus)
    others = [m for m in models if m != focus and m not in exclude]
    best = others[int(np.argmin([rmse[models.index(m)] for m in others]))]
    for m in others + ["best_other"]:
        j = models.index(best if m == "best_other" else m)
        ratio = boots[:, ci] / boots[:, j]
        l, h = np.percentile(ratio, [2.5, 97.5])
        point = rmse[ci] / rmse[j]
        verdict = "WIN" if h < 1 else ("LOSS" if l > 1 else "TIE")
        comp.append({"dataset": ds, "regime": regime, "target": target, "baseline": m,
                     "baseline_model": best if m == "best_other" else m, "ratio": point, "ratio_lo": l, "ratio_hi": h,
                     "verdict": verdict})
    return summ, pd.DataFrame(comp)


def main(allow_partial):
    rng = np.random.default_rng(2026)
    summaries, comps, meta = [], [], {}
    for ds in DATASETS:
        if not (SOTA / ds / "run_info.json").exists():
            continue
        df, info, complete = load(ds, allow_partial)
        meta[ds] = {"complete_repeats": [int(c) for c in complete], "rows": info["rows"], "groups": info["groups"]}
        if df.empty:
            continue
        df["row"] = df["row"].astype(int)
        for (regime, target), sub in df.groupby(["regime", "target"]):
            s, c = cell(ds, regime, target, sub, rng)
            summaries.append(s)
            comps.append(c)
            print(f"{ds} {regime} {target}: done", flush=True)
    summary = pd.concat(summaries, ignore_index=True)
    comp = pd.concat(comps, ignore_index=True)
    summary.to_csv(RES / "sota_summary.csv", index=False)
    comp.to_csv(RES / "sota_catforge_vs_baselines.csv", index=False)
    head = {"meta": meta}
    for regime in sorted(comp["regime"].unique()):
        for base in ("best_other", "lgbm"):
            sel = comp[(comp["regime"] == regime) & (comp["baseline"] == base)]
            head[f"{regime}_vs_{base}"] = sel["verdict"].value_counts().to_dict()
    (RES / "sota_headline.json").write_text(json.dumps(head, indent=2), encoding="utf-8")
    print(json.dumps(head, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    main(ap.parse_args().allow_partial)
