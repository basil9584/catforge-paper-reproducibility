"""CatFORGE predictor variants for ablation and repair.

A variant = a feature pipeline (the shipped one or a modified one) + several ensemble members fitted once on
the selected features. Different ensembles (shipped 3:2:1 vote, vote without the PLS member, single members,
tree-only votes) and an optional output guard are then formed from the same fitted members, so one fit yields
many candidate predictors.

Variant 'F0' with combination 'v321' is the shipped predictor; check_variants.py verifies it matches
run_sota.CatForgeModel (itself verified against generalized_predictor.train_and_predict_on_dataframe)."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

import run_sota as rs

gp = rs.gp
LGBM_SPACE = {"n_estimators": [100, 300, 600], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [7, 15, 31],
              "min_child_samples": [5, 10, 20], "subsample": [0.7, 0.9], "colsample_bytree": [0.6, 0.8, 1.0]}


class ClipToTrainRange(BaseEstimator, TransformerMixin):
    """Clips every feature to the range seen in training, so a value such as 1/(0+1e-6) in a new record cannot
    push a linear member far outside the data it was fitted on."""

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lo_, self.hi_ = X.min(axis=0), X.max(axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)


class Variant:
    def __init__(self, name, physics=True, clip=False, pls_aug=True, rfe=True, logit=True):
        self.name, self.physics, self.clip, self.pls_aug, self.rfe, self.logit = name, physics, clip, pls_aug, rfe, logit

    def _features(self, Xtr, Ytr_t):
        inputs = list(Xtr.columns)
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if self.physics:
            steps.append(("physics", gp.BlindPhysicsTransformer()))
        if self.clip:
            steps.append(("clip", ClipToTrainRange()))
        steps.append(("scaler", RobustScaler()))
        pre = ColumnTransformer([("num", Pipeline(steps), inputs)], remainder="drop")
        pls_comp = max(1, min(3, len(Xtr) - 1, len(inputs)))
        probe = pre.fit_transform(Xtr)
        n_total = (gp.PLSFeatureAugmenter(n_components=pls_comp).fit(probe, Ytr_t).transform(probe).shape[1]
                   if self.pls_aug else probe.shape[1])
        n_sel = max(1, min(15, max(3, (len(inputs) * 5) // 2), n_total - 1))
        parts = [("preprocessor", pre)]
        if self.pls_aug:
            parts.append(("pls_augment", gp.PLSFeatureAugmenter(n_components=pls_comp)))
        if self.rfe and n_total >= 2:
            parts.append(("selector", RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42),
                                          n_features_to_select=n_sel, step=0.1)))
        return Pipeline(parts).fit(Xtr, Ytr_t), pls_comp

    def fit_predict(self, Xtr, Ytr, gtr, Xte, seed, extras=True):
        """Returns {combination name: DataFrame of predictions on the original target scale}."""
        targets = list(Ytr.columns)
        Ytr_t, logit_cols = Ytr.copy(), []
        if self.logit:
            for t in targets:
                if rs.logit_ok(t, Ytr[t]):
                    logit_cols.append(t)
                    Ytr_t[t] = gp.logit_transform(Ytr[t])
        feat, pls_comp = self._features(Xtr, Ytr_t)

        def apply(X):  # step by step: Pipeline.transform refuses a pipeline ending in the custom PLS augmenter
            for _, step in feat.steps:
                X = step.transform(X)
            return np.asarray(X, float)
        Ztr, Zte = apply(Xtr), apply(Xte)
        out = {}
        for t in targets:
            y = Ytr_t[t].to_numpy()
            m = {"xgb": gp.AutoTunedXGB().fit(Ztr, y).predict(Zte),
                 "rf": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42).fit(Ztr, y).predict(Zte),
                 "pls": np.ravel(PLSRegression(n_components=pls_comp).fit(Ztr, y).predict(Zte))}
            if extras:
                m["lgbm"] = rs.Tuned(lgb.LGBMRegressor(random_state=seed, n_jobs=2, verbosity=-1, subsample_freq=1),
                                     LGBM_SPACE, 12, seed).fit(Ztr, y, gtr).predict(Zte)
                m["et"] = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=2).fit(Ztr, y).predict(Zte)
            combos = {"v321": (3 * m["xgb"] + 2 * m["rf"] + m["pls"]) / 6, "v32": (3 * m["xgb"] + 2 * m["rf"]) / 5,
                      "xgb": m["xgb"], "rf": m["rf"]}
            if extras:
                combos.update({"lgbm": m["lgbm"], "et": m["et"],
                               "trees4": (m["xgb"] + m["rf"] + m["lgbm"] + m["et"]) / 4,
                               "gbm_et": (m["xgb"] + m["lgbm"] + m["et"]) / 3})
            lo, hi = Ytr[t].min(), Ytr[t].max()
            for name, p in combos.items():
                p = gp.sigmoid_transform(p) if t in logit_cols else np.asarray(p, float)
                for suffix, q in (("", p), ("_c", np.clip(p, lo, hi))):
                    out.setdefault(name + suffix, {})[t] = q
        return {k: pd.DataFrame(v, index=Xte.index) for k, v in out.items()}


# Round 2 (decided after round 1 showed the logit link was the main cause of the accuracy gap): combinations of removals,
# all without the logit link and with the clipped-feature repair.
VARIANTS2 = {
    "N1": Variant("N1", clip=True, logit=False, physics=False),                                       # raw features + RFE
    "N2": Variant("N2", clip=True, logit=False, rfe=False),                                           # physics features, no RFE
    "N3": Variant("N3", clip=True, logit=False, physics=False, rfe=False),                            # raw features, all kept
    "N4": Variant("N4", clip=True, logit=False, physics=False, rfe=False, pls_aug=False),             # raw features only
}

VARIANTS = {
    "F0": Variant("F0"),                                # shipped feature pipeline
    "F1": Variant("F1", clip=True),                     # + features clipped to the training range (the repair)
    "F1-nologit": Variant("F1-nologit", clip=True, logit=False),
    "F1-nophys": Variant("F1-nophys", clip=True, physics=False),
    "F1-norfe": Variant("F1-norfe", clip=True, rfe=False),
}
