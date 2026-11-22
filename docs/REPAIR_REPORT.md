# Repair of the CatFORGE predictor and one-shot re-benchmark (18 Nov 2026)

**Short version.** The first benchmark ([SOTA_BENCHMARK_REPORT.md](SOTA_BENCHMARK_REPORT.md)) showed the shipped predictor trailing tuned tree ensembles and failing badly on one DRM paper. I found two causes, repaired them, and replaced a weak ensemble member, using development groups only. On 30% of the groups that no design decision ever used, the repaired predictor is **statistically tied with the best of ten tuned baselines on 4 of 7 targets and 6-8% worse on the 3 OCM targets**. It is now competitive, **not better than** tuned tree ensembles, so "state of the art" is still not a supportable claim.

## What was changed (`CatFORGE/backend/models/prediction/generalized_predictor.py`, diff in `generalized_predictor.patch`)
Original saved at `benchmark/_backup/generalized_predictor.py.2026-11-18.orig` (SHA-256 8B122083...C0205C).

| # | Change | Why (evidence) |
|---|---|---|
| 1 | Logit link on %-like targets off by default (`USE_LOGIT_LINK = False`; set True to restore) | Main cause of the accuracy gap. Dev ablation, geometric-mean RMSE vs tuned LightGBM: 1.13 with the link, 1.04 without. XGBoost with the link was also worse in the first benchmark. |
| 2 | New `ClipToTrainRange` step after the physics features | `1/(x+1e-6)` turns a pore volume of 0 into 10^6; the linear PLS member then extrapolated. Regression test (train without DRM paper 2020-53, predict it): H2/CO predictions -301.7 before, 0.62-0.69 after (true 0.39-0.92; RMSE 302 to 0.19). |
| 3 | Ensemble XGBoost 3 : random forest 2 : PLS 1 replaced by XGBoost 1 : extra trees 1 | In dev, the unbounded PLS member added error; XGBoost + extra trees was the best ensemble without a new dependency (about 1.01 vs LightGBM; LightGBM is not in the platform's environment). |

Not changed: physics features, PLS augmentation, RFE selection. In dev they neither helped nor hurt measurably once the logit link was gone (raw inputs only scored about 1.00 vs LightGBM, the engineered pipeline about 1.01).

## Protocol
- **Locked test.** 30% of the groups of each dataset, seeded shuffle (seed 4242): DRM 61 papers / 477 rows, WGS 25 references / 888 rows, OCM 232 element systems / 483 rows. DRM paper 2020-53 (the failure that motivated the repair) was kept in development.
- **Development.** The other 70% (1,087 / 3,472 / 1,261 rows). Grouped 5-fold CV, run on 3 (DRM, OCM) or 4 (WGS) folds before I stopped for time. Round 1 tested the shipped design, the clip repair, and removing the logit link, physics features or RFE; round 2 tested combinations (no logit with raw features, no RFE, no PLS augmentation). Ensembles were formed from stored member predictions.
- **Fidelity.** The benchmark replica reproduces the patched shipped function exactly (difference 0.0 on DRM and OCM; `check_catforge_v2_equivalence.py`).
- **Locked evaluation (once).** `run_final.py`: models trained on development groups (all rows, or 200 / 50 rows drawn at random, 5 draws each) and scored on every locked-test record. Verdict rule unchanged: 95% group-bootstrap CI of the RMSE ratio entirely below 1 = win, above 1 = loss, else tie.

## Results on the locked test (all development rows for training)
| Target | Repaired RMSE | Best other model | Ratio [95% CI] | Verdict | Before repair |
|---|---|---|---|---|---|
| DRM CH4 conv. | 15.20 | SVR 14.76 | 1.030 [0.932, 1.150] | tie | 0.999, tie |
| DRM CO2 conv. | 13.73 | ridge 13.64 | 1.007 [0.901, 1.107] | tie | 0.978, tie |
| DRM H2/CO | 0.158 | LightGBM 0.154 | 1.026 [0.965, 1.079] | tie | 1.071, loss |
| WGS CO conv. | 29.71 | LightGBM 30.38 (repaired is best of all) | 0.978 [0.917, 1.032] | tie | 1.122, loss |
| OCM CH4 conv. | 9.07 | LightGBM 8.57 | 1.058 [1.003, 1.113] | loss | 1.115, loss |
| OCM C2 selectivity | 17.81 | XGBoost 16.75 | 1.063 [1.012, 1.122] | loss | 1.168, loss |
| OCM C2 yield | 5.64 | LightGBM 5.24 | 1.077 [1.030, 1.133] | loss | 1.192, loss |

- **Totals:** 0 wins, 4 ties, 3 losses against the best other model (before repair: 0 / 2 / 5). Mean rank among the 12 models: LightGBM 3.0, **repaired CatFORGE 3.4**, XGBoost 3.8, extra trees 4.4, pre-repair CatFORGE 5.9.
- **Against LightGBM only:** all rows 1 win (DRM CH4, 0.915 [0.847, 0.992]), 4 ties, 2 losses (OCM CH4 and yield). 200 rows: 0 / 6 / 1. 50 rows: 3 wins (all DRM targets, ratios 0.82-0.92), 2 ties, 2 losses (WGS 1.07, OCM selectivity 1.04).
- **Small data:** against the best other model, 200 rows gives 5 ties and 2 losses; 50 rows gives 3 ties and 4 losses, all within 4-8%. The pre-repair predictor collapsed on OCM at 50 rows (ratios 1.90, 1.29, 3.32); the repaired one is at 1.05-1.07.
- **Not a clean improvement everywhere:** on DRM conversions the pre-repair predictor had slightly lower RMSE (14.74 and 13.34 vs 15.20 and 13.73). Both are ties with the best model, so this is within noise.
- **Ranking within a publication (Spearman, point estimates only):** repaired 0.44 / 0.47 / 0.26 on DRM and 0.54 on WGS, similar to or above LightGBM; on OCM 0.42-0.45 vs 0.34-0.38.

## What the paper can and cannot say
- **Supportable:** the predictor is statistically indistinguishable from the best of ten tuned baselines on DRM and WGS (4 of 7 targets) and within 6-8% on OCM; the benchmark exposed and fixed two defects; held-out-group evaluation was used.
- **Not supportable:** "state of the art", "outperforms", or "proves SOTA".

## Caveats
- The locked sets are small (25 WGS references, 61 DRM papers), so CIs are wide and a 3-5% gap cannot be resolved there.
- The design was chosen in development, on the same three tables the benchmark uses, and development folds were not run to completion (3 or 4 of 5). The locked evaluation is the only unbiased number; it was run once.
- **The remaining OCM gap has a likely cause that I cannot test honestly on this data.** In development, raw inputs with XGBoost + extra trees did about 3-5% better on OCM than the engineered features (relative to LightGBM 1.006 / 1.039 / 1.037 vs 1.039 / 1.085 / 1.065), probably because RFE discards composition columns. Changing the predictor again and re-scoring on the locked groups would invalidate the one-shot test; it needs fresh data (new literature records, or the QEERI data once its mapping is resolved).
- The same limits as the first benchmark apply: TabPFN, CatBoost and deep tabular models were not run; the tables are not a published leaderboard; baselines had a modest tuning budget.

## Files
- Code: `run_ablation.py`, `catforge_variants.py`, `analyze_ablation.py`, `catforge_v2.py`, `check_catforge_v2_equivalence.py`, `run_final.py`, `analyze_final.py`, `plot_final.py`, `pause_resume.ps1`.
- Data: `results/ablation/` (split definitions, development folds), `results/final/` (locked-test predictions), `results/final_summary.csv`, `results/final_catforge_vs_baselines.csv`, `results/final_headline.json`, `results/final_figure.png` / `.pdf`.
