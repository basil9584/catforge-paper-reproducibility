"""(1) The benchmark replica CatForgeV2 must reproduce the patched shipped function exactly.
(2) Regression test for the reported failure: train without DRM paper 2020-53 (pore volume = 0 rows), predict that paper."""
import tempfile
from pathlib import Path

import numpy as np

import run_sota as rs
from catforge_v2 import CatForgeV2

rs.gp.cross_val_predict = lambda model, X, y, **kw: model.predict(X)  # skip the plot-only 5-fold refits

for name, n_train in (("drm", 300), ("ocm", 300)):
    ds = rs.rb.LOADERS[name]()
    df = ds.frame.reset_index(drop=True)
    parts = rs.partition(df["group"].unique(), 5, 1000)
    test = df[df["group"].isin(parts[0])].head(200)
    train = df[~df["group"].isin(parts[0])].sample(n=n_train, random_state=7)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        train[ds.inputs + ds.targets].to_csv(tmp / "train.csv", index=False)
        test[ds.inputs].to_csv(tmp / "test.csv", index=False)
        cfg = [{"name": c, "role": "Input", "dataType": "continuous"} for c in ds.inputs] + \
              [{"name": c, "role": "Output", "dataType": "continuous"} for c in ds.targets]
        res, *_rest, err = rs.gp.train_and_predict_on_dataframe(str(tmp / "train.csv"), str(tmp / "test.csv"), cfg, str(tmp))
    assert err is None, err
    real = res[ds.targets].to_numpy()
    mine = CatForgeV2().fit(train[ds.inputs], train[ds.targets]).predict(test[ds.inputs])[ds.targets].to_numpy()
    print(f"{name}: max |patched shipped - replica| = {np.abs(real - mine).max():.3e}; patched shipped range {real.min():.2f}..{real.max():.2f}", flush=True)

ds = rs.rb.LOADERS["drm"]()
df = ds.frame.reset_index(drop=True)
bad = df[df["group"] == "2020-53"]
train = df[df["group"] != "2020-53"]
y = bad["Syngas_Ratio"].to_numpy()
for label, model in (("before repair (V1)", rs.CatForgeModel()), ("after repair (V2)", CatForgeV2())):
    p = model.fit(train[ds.inputs], train[ds.targets]).predict(bad[ds.inputs])["Syngas_Ratio"].to_numpy()
    print(f"{label}: H2/CO predictions for paper 2020-53 range {p.min():.2f}..{p.max():.2f}; true {y.min():.2f}..{y.max():.2f}; RMSE {np.sqrt(np.mean((p - y) ** 2)):.3f}", flush=True)
