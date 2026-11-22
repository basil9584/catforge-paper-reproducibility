"""Check that the interface's composition-linked cost ranks the Fig. 6 candidates the same way as the raw-material cost used for Fig. 6.

(Fig. 6 in the working scripts = Fig. 5 of the manuscript, cost-aware ranking.)"""
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
loader = SourceFileLoader("occ", str(ROOT / "catforge/optimization_catcost.py"))
spec = importlib.util.spec_from_loader("occ", loader); occ = importlib.util.module_from_spec(spec); loader.exec_module(occ)

loadings = np.arange(2, 25.01, 0.5)
tea = {"materials": [{"name": "Ni", "cost": 17.0, "link_column": "Ni Loading", "link_factor": 0.01},
                     {"name": "Al2O3", "cost": 0.57, "balance": True}],
       "equipment": [], "labor": [], "scale": 1000, "project_life": 10}
app = np.array([occ.compute_row_tea_cost(tea, {"Ni Loading": w}) for w in loadings])
raw = loadings / 100 * 17.0 + (1 - loadings / 100) * 0.57
ratio = app / raw
print(f"raw-material cost US$ {raw.min():.2f}-{raw.max():.2f}/kg; app unit cost US$ {app.min():.2f}-{app.max():.2f}/kg; "
      f"ratio {ratio.min():.6f}-{ratio.max():.6f}")
rng = np.random.default_rng(0)
perf = pd.DataFrame({"ch4": rng.uniform(60, 70, len(loadings)), "co2": rng.uniform(65, 75, len(loadings))})
same = True
for wc in [0.0, 0.05, 0.15, 0.4, 0.8]:
    wts = {"ch4": (1 - wc) / 2, "co2": (1 - wc) / 2, "cost": wc}
    objs = {"ch4": "maximize", "co2": "maximize", "cost": "minimize"}
    s_raw = occ.calculate_topsis_score(perf.assign(cost=raw), ["ch4", "co2", "cost"], wts, objs)
    s_app = occ.calculate_topsis_score(perf.assign(cost=app), ["ch4", "co2", "cost"], wts, objs)
    same &= bool(np.allclose(s_raw, s_app))
print("TOPSIS scores identical for every cost weight:", same)
raise SystemExit(0 if same else 1)      # package addition: non-zero exit code if the check fails
