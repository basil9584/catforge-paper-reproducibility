"""Head-to-head benchmark: the CatFORGE predictor versus tuned state-of-the-art tabular baselines on
publications (or element systems) the models have never seen.

PRE-SPECIFIED PROTOCOL (written before any result was seen)
  Datasets   drm, wgs, ocm, loaded exactly as in run_benchmark.py (same rows, same group keys).
  Splits     grouped K-fold: whole publications / references / element systems are held out. 5 folds x 3
             repeats, each repeat a different random group partition. Every model sees the same splits.
  Regimes    'full' = all rows of the training groups; 50 and 200 = that many rows drawn at random from the
             training groups (the small-data setting CatFORGE is aimed at).
  Models     catforge     the shipped predictor, rebuilt from generalized_predictor.train_and_predict_on_dataframe
                          (PLS-augmented XGB + RF + PLS voting ensemble, logit target for % columns); fitted on
                          all targets of a dataset jointly, as the platform does.
             lgbm, xgb    tuned by randomized search with inner grouped CV on the training groups only
             xgb_logit    xgb with the same logit target transform CatFORGE applies (isolates that component)
             rf, et       random forest, extra trees (300 trees, defaults)
             ridge, knn, svr, mlp, mean   simple references; mean = training-set mean
  Primary    RMSE on held-out groups, regime 'full', per target. CatFORGE counts as a WIN against a baseline
  outcome    only if the 95% paired group-bootstrap CI of RMSE_catforge / RMSE_baseline - 1 is entirely
             below 0; a LOSS if entirely above 0; otherwise a TIE (see analyze_sota.py).
  Leakage    groups are split before anything is fitted; tuning uses only training groups; no target or
             test row reaches any preprocessing step.

Usage
  python run_sota.py --dataset drm            # full run for one dataset (resumable)
  python run_sota.py --dataset drm --smoke    # 1 fold, regime 50, fast
Outputs: results/sota/<dataset>/rep{r}_fold{k}.csv.gz (out-of-fold predictions) and run_info.json
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import types
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repro_paths as rp  # noqa: E402  (package paths)

ROOT = rp.PKG
PRED_DIR = rp.CATFORGE          # holds the verbatim generalized_predictor.py
STUBS = HERE / "_stubs"
# generalized_predictor imports shap at module level (only used for explanation plots, which are not part of
# prediction). A stub package keeps the benchmark venv light; PYTHONPATH makes it visible to joblib workers.
# (Package: the stub is shipped in benchmark/_stubs/shap.py and only written if it is missing.)
if not (STUBS / "shap.py").exists():
    STUBS.mkdir(exist_ok=True)
    (STUBS / "shap.py").write_text("# stub: SHAP is not needed to fit or predict\n", encoding="utf-8")
os.environ["PYTHONPATH"] = os.pathsep.join([str(STUBS), str(PRED_DIR), os.environ.get("PYTHONPATH", "")])
sys.path[:0] = [str(STUBS), str(PRED_DIR), str(HERE)]
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVR

import generalized_predictor as gp  # the shipped CatFORGE predictor module
import run_benchmark as rb          # dataset loaders shared with the augmentation benchmark

MODELS = ["catforge", "lgbm", "xgb", "xgb_logit", "rf", "et", "ridge", "knn", "svr", "mlp", "mean"]
OUT = rp.BENCH_OUT / "predictor_first_round" / "sota"


# ---------------------------------------------------------------- CatFORGE predictor (shipped pipeline)
class CatForgeModel:
    """Rebuilds the model pipeline of generalized_predictor.train_and_predict_on_dataframe (lines 307-362)
    without file input/output or explanation plots. Only the classes and heuristics of that module are used."""

    def fit(self, X, Y):
        self.inputs, self.outputs = list(X.columns), list(Y.columns)
        X = X.copy()
        Y = Y.copy()
        self.logit_cols = []
        for c in self.outputs:
            if "%" in c or "conversion" in c.lower():
                v = Y[c]
                if v.min() >= 0 and v.max() <= 100:
                    self.logit_cols.append(c)
        for c in self.logit_cols:
            Y[c] = gp.logit_transform(Y[c])
        num_feats, cat_feats = self.inputs, []
        numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("physics", gp.BlindPhysicsTransformer()),
                            ("scaler", RobustScaler())])
        categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        pre = ColumnTransformer([("num", numeric, num_feats), ("cat", categorical, cat_feats)], remainder="drop")
        pls_comp = max(1, min(3, len(X) - 1, len(self.inputs)))
        try:
            probe = pre.fit_transform(X)
            n_total = gp.PLSFeatureAugmenter(n_components=pls_comp).fit(probe, Y).transform(probe).shape[1]
        except Exception:
            n_total = len(num_feats) * 5 + len(cat_feats) * 3 + pls_comp
        n_sel = max(1, min(15, max(3, (len(num_feats) * 5 + len(cat_feats) * 3) // 2), n_total - 1))
        if n_total >= 2:
            selector = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42),
                           n_features_to_select=n_sel, step=0.1)
        else:
            selector = FunctionTransformer()
        voting = VotingRegressor(estimators=[("xgb", gp.AutoTunedXGB()),
                                             ("rf", RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)),
                                             ("pls", PLSRegression(n_components=pls_comp))], weights=[3, 2, 1])
        self.pipe = Pipeline([("preprocessor", pre), ("pls_augment", gp.PLSFeatureAugmenter(n_components=pls_comp)),
                              ("selector", selector), ("regressor", MultiOutputRegressor(voting))])
        self.pipe.fit(X, Y)
        return self

    def predict(self, X):
        out = pd.DataFrame(self.pipe.predict(X), columns=self.outputs, index=X.index)
        for c in self.logit_cols:
            out[c] = gp.sigmoid_transform(out[c])
        return out


# ---------------------------------------------------------------- baselines
class Tuned:
    """Randomized search with inner grouped CV on the training groups only."""

    def __init__(self, est, space, n_iter, seed):
        self.est, self.space, self.n_iter, self.seed = est, space, n_iter, seed

    def fit(self, X, y, groups):
        k = min(3, int(pd.Series(groups).nunique()))
        splits = list(GroupKFold(n_splits=k).split(X, y, groups)) if k >= 2 else 3
        self.s = RandomizedSearchCV(self.est, self.space, n_iter=self.n_iter, cv=splits, random_state=self.seed,
                                    scoring="neg_root_mean_squared_error", n_jobs=1, refit=True)
        self.s.fit(X, y)
        return self

    def predict(self, X):
        return self.s.predict(X)


def logit_ok(target, y):
    return ("%" in target or "conversion" in target.lower()) and y.min() >= 0 and y.max() <= 100


def fit_predict_baseline(name, seed, Xtr, ytr, gtr, Xte, target, nj):
    """Returns predictions for one target. Every model sees only the training rows."""
    # Plain arrays: LightGBM rejects column names such as "p(CH4), bar"; no baseline uses column names.
    Xtr, Xte = Xtr.to_numpy(float), Xte.to_numpy(float)
    if name == "mean":
        return DummyRegressor().fit(Xtr, ytr).predict(Xte)
    if name == "ridge":
        m = Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
        return m.fit(Xtr, ytr).predict(Xte)
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=nj).fit(Xtr, ytr).predict(Xte)
    if name == "et":
        return ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=nj).fit(Xtr, ytr).predict(Xte)
    if name == "knn":
        est = Pipeline([("s", StandardScaler()), ("k", KNeighborsRegressor(weights="distance"))])
        return Tuned(est, {"k__n_neighbors": [3, 5, 10, 20]}, 4, seed).fit(Xtr, ytr, gtr).predict(Xte)
    if name == "svr":
        est = TransformedTargetRegressor(Pipeline([("s", StandardScaler()), ("v", SVR())]), transformer=StandardScaler())
        space = {"regressor__v__C": [1, 10, 100], "regressor__v__gamma": ["scale", 0.01, 0.1]}
        return Tuned(est, space, 6, seed).fit(Xtr, ytr, gtr).predict(Xte)
    if name == "mlp":
        est = TransformedTargetRegressor(
            Pipeline([("s", StandardScaler()), ("m", MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=600, early_stopping=True,
                                                                  random_state=seed))]), transformer=StandardScaler())
        return est.fit(Xtr, ytr).predict(Xte)
    if name in ("xgb", "xgb_logit"):
        est = XGBRegressor(random_state=seed, n_jobs=nj, verbosity=0)
        space = {"n_estimators": [100, 300, 600], "max_depth": [3, 4, 6], "learning_rate": [0.03, 0.05, 0.1],
                 "subsample": [0.7, 0.9], "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5]}
        use_logit = name == "xgb_logit" and logit_ok(target, ytr)
        y = pd.Series(gp.logit_transform(ytr.to_numpy()), index=ytr.index) if use_logit else ytr
        pred = Tuned(est, space, 12, seed).fit(Xtr, y, gtr).predict(Xte)
        return gp.sigmoid_transform(pred) if use_logit else pred
    if name == "lgbm":
        est = lgb.LGBMRegressor(random_state=seed, n_jobs=nj, verbosity=-1, subsample_freq=1)
        space = {"n_estimators": [100, 300, 600], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [7, 15, 31],
                 "min_child_samples": [5, 10, 20], "subsample": [0.7, 0.9], "colsample_bytree": [0.6, 0.8, 1.0]}
        return Tuned(est, space, 12, seed).fit(Xtr, ytr, gtr).predict(Xte)
    raise ValueError(name)


def bound(pred, target, ytr):
    pred = np.asarray(pred, float)
    return np.clip(pred, 0, 100) if logit_ok(target, ytr) else pred


# ---------------------------------------------------------------- protocol
def partition(groups, k, seed):
    order = np.random.default_rng(seed).permutation(np.array(sorted(groups)))
    return [set(f) for f in np.array_split(order, k)]


def run(args):
    ds = rb.LOADERS[args.dataset]()
    df = ds.frame.reset_index(drop=True)
    out = OUT / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    regimes = [r if r == "full" else int(r) for r in args.regimes.split(",")]
    models = MODELS if args.models == "all" else args.models.split(",")
    folds_to_run = 1 if args.smoke else args.folds
    reps = 1 if args.smoke else args.repeats
    info = {"dataset": ds.name, "rows": len(df), "groups": int(df["group"].nunique()), "group_unit": ds.group_unit,
            "inputs": ds.inputs, "targets": ds.targets, "regimes": regimes, "models": models, "folds": args.folds,
            "repeats": args.repeats, "data_sha256": hashlib.sha256(Path(ds.source).read_bytes()).hexdigest(),
            "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                         "lightgbm": lgb.__version__},
            "protocol": "see module docstring"}
    if not args.smoke:
        (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    nj = args.n_jobs
    for rep in range(reps):
        parts = partition(df["group"].unique(), args.folds, 1000 + rep)
        for k in range(folds_to_run):
            fn = out / (f"smoke_rep{rep}_fold{k}.csv.gz" if args.smoke else f"rep{rep}_fold{k}.csv.gz")
            if fn.exists() and not args.smoke:
                continue
            t0 = time.time()
            te_mask = df["group"].isin(parts[k])
            pool, test = df[~te_mask], df[te_mask]
            recs = []
            for regime in regimes:
                seed = 10_000 * rep + 100 * k + (0 if regime == "full" else int(regime))
                train = pool if regime == "full" else pool.sample(n=min(regime, len(pool)), random_state=seed)
                Xtr, Xte = train[ds.inputs], test[ds.inputs]
                for name in models:
                    if name == "catforge":
                        pred = CatForgeModel().fit(Xtr, train[ds.targets]).predict(Xte)
                        preds = {t: pred[t].to_numpy() for t in ds.targets}
                    else:
                        # Percent targets get the same physical bound CatFORGE builds in via its logit link, so no
                        # baseline is penalised for extrapolating outside 0-100 %.
                        preds = {t: bound(fit_predict_baseline(name, seed, Xtr, train[t], train["group"], Xte, t, nj),
                                          t, train[t]) for t in ds.targets}
                    for t in ds.targets:
                        recs.append(pd.DataFrame({"regime": str(regime), "repeat": rep, "fold": k, "model": name,
                                                  "target": t, "row": test.index.to_numpy(), "group": test["group"].to_numpy(),
                                                  "y": test[t].to_numpy(), "pred": np.asarray(preds[t], float)}))
            pd.concat(recs).to_csv(fn, index=False, compression="gzip")
            print(f"[{args.dataset}] rep {rep} fold {k} done in {time.time() - t0:.0f}s "
                  f"(train groups {df['group'].nunique() - len(parts[k])}, test rows {len(test)})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(rb.LOADERS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--regimes", default="50,200,full")
    ap.add_argument("--models", default="all")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    run(ap.parse_args())
