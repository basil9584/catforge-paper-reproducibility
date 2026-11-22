"""End-to-end test of the CatCost optimization module after the 22 Nov 2026 cost fix.

Runs steps 1-7 of CatFORGE/backend/models/optimization/optimization_catcost.py on the DRM literature table and checks:
  S0  backward compatibility: a fixed-quantity configuration gives the same costs and ranking as the original module
  S1  composition-linked materials: cost differs between records and between suggested candidates, and the
      user cost weight changes the TOPSIS ranking
  S2  the same with realistic fixed costs (equipment and labour), to see how much composition still matters
  S3  small data set (< 30 records, local perturbation path)
  S4  per-record override entered in the interface (row_cost_config) scales that record's cost
  S5  a record with no value in the linked column uses the median record's cost
  S6  a link to a column that is not a model input produces a warning
Usage (from the package root): python benchmark/test_cost_ranking.py [scratch dir]
Package paths: module catforge/optimization_catcost.py, original module (for S0)
catforge/previous_versions/optimization_catcost_before_2026-11-22_cost_fix.py, data data/drm_literature.csv; the result
is written to output/tests/cost_ranking_test.json (stored copy: results/cost_module/cost_ranking_test.json).
"""
import importlib.util
import json
import random
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "catforge/optimization_catcost.py"
OLD = ROOT / "catforge/previous_versions/optimization_catcost_before_2026-11-22_cost_fix.py"
DATA = ROOT / "data/drm_literature.csv"
OUT = ROOT / "output/tests/cost_ranking_test.json"
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output/tests/cost_test_scratch"

INPUTS = ["Ratio of CH4 in Feed", "Reaction Temperature", "Ni Loading", "Reaction Time", "Pore Size", "Pore Volume",
          "Surface Area", "H2-TPR Peak Temperature", "Ni Particle Size", "Modifier Electronegativity", "GHSV"]
TARGETS = ["CH4 Conversion", "CO2 Conversion"]


def load(path, name):
    loader = SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


new, old = load(NEW, "occ_new"), load(OLD, "occ_old")

raw = pd.read_csv(DATA)
raw.columns = raw.columns.str.strip()
base = raw.dropna(subset=INPUTS + TARGETS)[INPUTS + TARGETS].reset_index(drop=True)
base.loc[5, "Ni Loading"] = np.nan          # S5: one record without a nickel loading
SCRATCH.mkdir(parents=True, exist_ok=True)
full_csv, small_csv = SCRATCH / "drm_full.csv", SCRATCH / "drm_small.csv"
base.to_csv(full_csv, index=False)
base.iloc[:25].to_csv(small_csv, index=False)

STEP2 = {"inputs": INPUTS, "targets": TARGETS, "categoricals": [], "weights": {t: 1.0 for t in TARGETS},
         "objectives": {t: "maximize" for t in TARGETS}, "enable_cost": True}

LINKED_MATS = [
    {"name": "Ni (metal value)", "cost": 17.0, "qty": 1, "link_column": "Ni Loading", "link_factor": "0.01",
     "fraction_in_catalyst": "1"},
    {"name": "Al2O3 support", "cost": 0.57, "qty": 1, "balance": True},
]


def run(mod, tag, csv, tea, seed=42):
    job = SCRATCH / tag
    job.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed)
    mod.step1_upload_and_analyze(str(csv), str(job))
    mod.step2_configure_and_prepare(tag, str(job), dict(STEP2))
    r3 = mod.step3_configure_tea(tag, str(job), tea)
    d4 = mod.step4_filter_data(tag, str(job), {"filters": [], "logic": "AND"})
    mod.step5_rank_data(tag, str(job))
    mod.step6_suggest_candidates(tag, str(job), seed=seed)
    d7 = mod.step7_pareto_analysis(tag, str(job))
    d6 = pd.read_csv(job / "step6_suggested_candidates.csv")
    return r3, d4, d6, d7, job


def rerank(mod, job, cost_weight):
    """Re-run step 7 only, with a different cost weight."""
    cfg_path = job / "tea_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["cost_weight"] = cost_weight
    cfg_path.write_text(json.dumps(cfg))
    return mod.step7_pareto_analysis(job.name, str(job))


def summary_costs(s):
    s = pd.to_numeric(s, errors="coerce")
    return {"min": round(float(s.min()), 4), "max": round(float(s.max()), 4), "n_unique": int(s.round(8).nunique())}


res, checks = {}, []


def check(name, ok, detail=""):
    checks.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


# ---------------- S0: backward compatibility (fixed quantities, legacy CatCost format) ----------------
LEGACY = {"materials": [{"name": "Ni", "quantity": {"baseline": 100}, "unit_price": {"baseline": 17.0}},
                        {"name": "Al2O3", "quantity": {"baseline": 900}, "unit_price": {"baseline": 0.57}}],
          "equipment": [{"equipment_type": "reactor", "quantity": 1, "base_cost": 20000}],
          "labor": {"rate": 25, "operators": 1, "hours_per_year": 500, "utilities_annual": 1000},
          "scale": 1000, "project_life": 10}
_, o4, o6, o7, _ = run(old, "S0_old", full_csv, json.loads(json.dumps(LEGACY)))
_, n4, n6, n7, _ = run(new, "S0_new", full_csv, json.loads(json.dumps(LEGACY)))
check("S0 same record costs as the original module",
      np.allclose(o4["Total_Running_Cost"].values, n4["Total_Running_Cost"].values),
      f"old {summary_costs(o4['Total_Running_Cost'])}, new {summary_costs(n4['Total_Running_Cost'])}")
check("S0 same step-7 ranking as the original module",
      list(o7["TOPSIS_Score"].round(10)) == list(n7["TOPSIS_Score"].round(10)))
res["S0_fixed_quantities"] = {"record_cost": summary_costs(n4["Total_Running_Cost"]),
                              "candidate_cost": summary_costs(n6["Total_Running_Cost"])}

# ---------------- S1: composition-linked materials, materials-only cost ----------------
TEA1 = {"materials": LINKED_MATS, "equipment": [], "labor": [], "scale": 1000, "project_life": 10}
r3, d4, d6, d7, job1 = run(new, "S1_linked", full_csv, json.loads(json.dumps(TEA1)))
valid = d4["Ni Loading"].notna()
rho_rec = spearmanr(d4.loc[valid, "Ni Loading"], d4.loc[valid, "Total_Running_Cost"])[0]
rho_cand = spearmanr(d6["Ni Loading"], d6["Total_Running_Cost"])[0] if len(d6) > 2 else float("nan")
check("S1 record costs differ", d4["Total_Running_Cost"].round(8).nunique() > 10, str(summary_costs(d4["Total_Running_Cost"])))
check("S1 record cost rises with nickel loading", rho_rec > 0.999, f"Spearman {rho_rec:.4f}")
check("S1 candidate costs differ", d6["Total_Running_Cost"].round(8).nunique() == len(d6), str(summary_costs(d6["Total_Running_Cost"])))
check("S1 candidate cost rises with nickel loading", rho_cand > 0.999, f"Spearman {rho_cand:.4f}")
# Hand check of one record: materials per kg x opex overheads (1 + 0.5 overhead) x (1 + 0.25 general)
w = float(d4.loc[0, "Ni Loading"]) / 100
expected = (w * 17.0 + (1 - w) * 0.57) * 1.5 * 1.25
check("S1 hand calculation for record 0", abs(expected - d4.loc[0, "Total_Running_Cost"]) < 1e-9,
      f"expected {expected:.4f}, got {d4.loc[0, 'Total_Running_Cost']:.4f}")
check("S1 preview reports the per-record range", r3["preview"].get("per_record_cost") is not None, str(r3["preview"].get("per_record_cost")))

weights_out = {}
for cw in [0.0, None, 1.0, 3.0]:
    d = rerank(new, job1, cw)
    top = d.sort_values("TOPSIS_Score", ascending=False).head(10)
    cand = d[d["Source"] == "AI Candidate"].sort_values("TOPSIS_Score", ascending=False)
    weights_out[str(cw)] = {
        "top10_mean_Ni_loading": round(float(top["Ni Loading"].mean()), 2),
        "top10_mean_cost": round(float(top["Total_Running_Cost"].mean()), 3),
        "top10_mean_CH4_pred": round(float(top["CH4 Conversion_pred"].mean()), 1),
        "candidate_order": [int(i) for i in cand.index[:10]],
        "n_pareto": int(d["Is_Pareto_Optimal"].sum()),
    }
res["S1_linked_materials_only"] = {"record_cost": summary_costs(d4["Total_Running_Cost"]),
                                   "candidate_cost": summary_costs(d6["Total_Running_Cost"]),
                                   "spearman_cost_vs_Ni_records": round(float(rho_rec), 4),
                                   "preview": r3["preview"], "by_cost_weight": weights_out}
check("S1 cost weight changes the top-10 (0 vs 3)",
      weights_out["0.0"]["top10_mean_Ni_loading"] != weights_out["3.0"]["top10_mean_Ni_loading"],
      f"mean Ni loading of top 10: w=0 {weights_out['0.0']['top10_mean_Ni_loading']}, "
      f"w=avg {weights_out['None']['top10_mean_Ni_loading']}, w=3 {weights_out['3.0']['top10_mean_Ni_loading']}")
check("S1 higher cost weight gives a cheaper top 10",
      weights_out["3.0"]["top10_mean_cost"] < weights_out["0.0"]["top10_mean_cost"],
      f"w=0 {weights_out['0.0']['top10_mean_cost']}, w=3 {weights_out['3.0']['top10_mean_cost']} US$/kg")

# ---------------- S2: linked materials plus realistic fixed costs ----------------
TEA2 = {"materials": LINKED_MATS,
        "equipment": [{"name": "Calcination furnace", "costType": "one_time", "cost": 50000}],
        "labor": [{"name": "Technician", "costType": "per_hour", "cost": 25, "duration": 2080, "durationUnit": "hours"}],
        "lab_cost": [{"name": "Characterization", "costType": "one_time", "cost": 5000}],
        "scale": 10000, "project_life": 10}
r3b, e4, e6, e7, job2 = run(new, "S2_fixed_costs", full_csv, json.loads(json.dumps(TEA2)))
s2 = {}
for cw in [None, 3.0]:
    d = rerank(new, job2, cw)
    top = d.sort_values("TOPSIS_Score", ascending=False).head(10)
    s2[str(cw)] = {"top10_mean_Ni_loading": round(float(top["Ni Loading"].mean()), 2),
                   "top10_mean_cost": round(float(top["Total_Running_Cost"].mean()), 3)}
rc = e4["Total_Running_Cost"]
res["S2_with_fixed_costs"] = {"record_cost": summary_costs(rc), "relative_spread_pct": round(float((rc.max() - rc.min()) / rc.median() * 100), 1),
                              "preview": r3b["preview"], "by_cost_weight": s2}
check("S2 runs with interface-format labour, equipment and lab items", rc.round(8).nunique() > 10,
      f"{summary_costs(rc)}, spread {res['S2_with_fixed_costs']['relative_spread_pct']}% of median")

# ---------------- S3: small data set (local search path) ----------------
_, f4, f6, _, _ = run(new, "S3_small", small_csv, json.loads(json.dumps(TEA1)))
check("S3 small-data candidates get their own cost", f6["Total_Running_Cost"].round(8).nunique() > 1,
      f"{len(f6)} candidates, {summary_costs(f6['Total_Running_Cost'])}")
res["S3_small"] = {"n_candidates": int(len(f6)), "candidate_cost": summary_costs(f6["Total_Running_Cost"])}

# ---------------- S4: per-record override from the interface ----------------
TEA4 = json.loads(json.dumps(TEA1))
TEA4.update({"row_cost_config": {"0": {"total": 200.0}, "1": {"total": 100.0}}, "row_cost_reference_total": 100.0})
_, g4, _, _, _ = run(new, "S4_override", full_csv, TEA4)
check("S4 overridden record 0 costs twice its composition cost",
      abs(g4.loc[0, "Total_Running_Cost"] - 2 * d4.loc[0, "Total_Running_Cost"]) < 1e-9,
      f"{d4.loc[0, 'Total_Running_Cost']:.4f} -> {g4.loc[0, 'Total_Running_Cost']:.4f}")
check("S4 record 1 (override equal to reference) unchanged",
      abs(g4.loc[1, "Total_Running_Cost"] - d4.loc[1, "Total_Running_Cost"]) < 1e-9)

# ---------------- S5: record without a nickel loading ----------------
fallback = json.loads((job1 / "tea_config.json").read_text()).get("linked_fallback_unit_cost")
check("S5 record without a linked value uses the median record's cost",
      fallback is not None and abs(d4.loc[5, "Total_Running_Cost"] - fallback) < 1e-9,
      f"cost {d4.loc[5, 'Total_Running_Cost']:.4f}, median record {fallback}")

# ---------------- S6: link to a column that is not an input ----------------
TEA6 = json.loads(json.dumps(TEA1)); TEA6["materials"][0]["link_column"] = "CO2 Conversion"
r6, _, _, _, _ = run(new, "S6_bad_link", full_csv, TEA6)
check("S6 warning for a link to a non-input column", any("not a model input" in w for w in r6.get("warnings", [])),
      "; ".join(r6.get("warnings", [])))

res["checks"] = checks
res["n_pass"] = sum(c["pass"] for c in checks)
res["n_checks"] = len(checks)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2, default=float))
print(f"\n{res['n_pass']}/{res['n_checks']} checks passed; details in {OUT}")
