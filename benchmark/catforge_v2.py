"""Benchmark replica of the repaired CatFORGE predictor (generalized_predictor.py after the 2026-11-18 patch).

Mirrors train_and_predict_on_dataframe: features = impute -> physics expansion -> clip to training range -> robust scaling,
then PLS augmentation and RFE selection, then MultiOutputRegressor(VotingRegressor(tuned XGBoost, extra trees)),
with the logit link off. check_catforge_v2_equivalence.py verifies it against the patched function."""
import pandas as pd
from sklearn.cross_decomposition import PLSRegression  # noqa: F401  (kept for parity with the module's imports)
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

import run_sota as rs

gp = rs.gp


class CatForgeV2:
    def fit(self, X, Y):
        self.inputs, self.outputs = list(X.columns), list(Y.columns)
        Y = Y.copy()
        self.logit_cols = []
        if gp.USE_LOGIT_LINK:  # False after the patch
            for c in self.outputs:
                if ("%" in c or "conversion" in c.lower()) and Y[c].min() >= 0 and Y[c].max() <= 100:
                    self.logit_cols.append(c)
                    Y[c] = gp.logit_transform(Y[c])
        num_feats, cat_feats = self.inputs, []
        numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("physics", gp.BlindPhysicsTransformer()),
                            ("clip", gp.ClipToTrainRange()), ("scaler", RobustScaler())])
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
        selector = (RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=n_sel, step=0.1)
                    if n_total >= 2 else FunctionTransformer())
        voting = VotingRegressor(estimators=[("xgb", gp.AutoTunedXGB()),
                                             ("et", ExtraTreesRegressor(n_estimators=300, random_state=42))], weights=[1, 1])
        self.pipe = Pipeline([("preprocessor", pre), ("pls_augment", gp.PLSFeatureAugmenter(n_components=pls_comp)),
                              ("selector", selector), ("regressor", MultiOutputRegressor(voting))])
        self.pipe.fit(X, Y)
        return self

    def predict(self, X):
        out = pd.DataFrame(self.pipe.predict(X), columns=self.outputs, index=X.index)
        for c in self.logit_cols:
            out[c] = gp.sigmoid_transform(out[c])
        return out
