# CatFORGE predictor vs state-of-the-art tabular baselines (run 18 Nov 2026)

> **Superseded for the predictor's current status:** this is the first benchmark, of the predictor before repair. The repair, its development ablation and the one-shot locked-test result are in [REPAIR_REPORT.md](REPAIR_REPORT.md).

**Question.** Does the CatFORGE predictor, as shipped, beat tuned state-of-the-art tabular models on catalysts from publications (or element systems) it has never seen?

**Answer.** No. On the pre-specified primary comparison (all training rows, 7 targets, held-out groups) CatFORGE won 0, tied 3 and lost 4 against the best other model. Its mean rank among 11 models was 8.3; LightGBM's was 1.3. A paper cannot claim "state of the art" for the predictor on this evidence.

## Protocol (fixed in `run_sota.py` before any result was seen)

| Item | Setting |
|---|---|
| Datasets | DRM (1,564 rows, 204 papers held out by paper), WGS (4,360 rows, 84 references), OCM (1,744 rows, 773 element systems; the table records no papers). Same rows and group keys as `run_benchmark.py`. |
| Targets | DRM: CH4 conversion, CO2 conversion, H2/CO. WGS: CO conversion. OCM: CH4 conversion, C2 selectivity, C2 yield. |
| Splits | Grouped 5-fold, 3 repeats (new random group partition each repeat). Whole groups are held out; every model sees the same splits. |
| Regimes | All training rows, or 200 or 50 rows drawn at random from the training groups (small-data setting). |
| Models | **catforge**: the shipped predictor (PLS-augmented XGBoost + random forest + PLS voting ensemble, logit link on % targets), fitted on all targets of a dataset jointly. **lgbm, xgb**: tuned (12 random configurations, inner grouped 3-fold CV on training groups only). **xgb_logit**: xgb with CatFORGE's logit link. **rf, et**: 300 trees. **ridge, knn, svr, mlp**: simple references. **mean**: training mean. |
| Primary outcome | RMSE on held-out groups, all training rows, per target. WIN if the 95% paired group-bootstrap CI of RMSE_catforge / RMSE_baseline is entirely below 1, LOSS if entirely above 1, otherwise TIE. |
| Leakage | Groups are split before any fitting; tuning uses training groups only; percent-target predictions of all baselines are clipped to 0-100 %, the same bound CatFORGE's logit link enforces. |

**Fidelity check.** `check_catforge_equivalence.py` runs the shipped `train_and_predict_on_dataframe` and the benchmark's rebuilt pipeline on the same data: the largest prediction difference is 0.0 on DRM and OCM.

## Results (all training rows)

| Dataset: target | CatFORGE RMSE | Best other | Ratio [95% CI] | Verdict |
|---|---|---|---|---|
| DRM: CH4 conv. | 18.10 | LightGBM 17.24 | 1.050 [0.989, 1.117] | tie |
| DRM: CO2 conv. | 16.76 | Extra trees 15.63 | 1.072 [0.991, 1.166] | tie |
| DRM: H2/CO | 17.14 | Extra trees 0.168 | 102 [0.97, 190] | tie by CI, but a failure (below) |
| WGS: CO conv. | 30.32 | LightGBM 26.88 | 1.128 [1.069, 1.184] | loss |
| OCM: CH4 conv. | 9.85 | LightGBM 8.55 | 1.151 [1.110, 1.200] | loss |
| OCM: C2 selectivity | 20.06 | LightGBM 16.52 | 1.214 [1.173, 1.261] | loss |
| OCM: C2 yield | 6.34 | LightGBM 5.04 | 1.257 [1.210, 1.305] | loss |

- **Rank across the 7 targets (1 = best of 11):** LightGBM 1.3, XGBoost 2.4, extra trees 3.0, XGBoost+logit 4.2, random forest 4.3, SVR 6.6, ridge 7.3, **CatFORGE 8.3**, k-NN 9.0, MLP 9.0, training mean 10.7.
- **R2:** DRM conversions 0.50 / 0.49 vs LightGBM 0.55 / 0.55; WGS 0.13 vs 0.32; OCM 0.20 / 0.36 / 0.10 vs 0.46 / 0.52 / 0.43.
- **Small data (50 and 200 rows):** CatFORGE loses to the best other model in 5 of 7 targets at each size and ties in 2. Its one win is against LightGBM only (DRM CH4, 50 rows, ratio 0.927 [0.866, 0.991]); extra trees is better still there.
- **Ranking within a publication (Spearman, point estimates, no CIs):** on par on DRM and WGS (e.g. WGS 0.546 vs LightGBM 0.558), lower on OCM (0.24-0.39 vs 0.40-0.49). Top-1 regret on DRM: CatFORGE 6.5 and 5.2 vs LightGBM 7.0 and 5.4 (lower is better).
- **Logit link:** XGBoost with the logit link is no better than without it, and worse on OCM (C2 selectivity RMSE 18.95 vs 16.67).

### The DRM H2/CO failure (diagnosed)
CatFORGE's RMSE there is 100x the others because of six rows from one held-out paper (2020-53) where `Pore Volume` is 0. The pipeline's `BlindPhysicsTransformer` builds `1/(x + 1e-6)` features, which become 10^6 for those rows, and the linear PLS member of the voting ensemble extrapolates to a prediction of about -144 for a ratio that lies between 0 and 1.6. **Post hoc**, without that paper CatFORGE's RMSE is 0.169, the same as LightGBM (0.169). The bootstrap "tie" for that target is an artifact of one group dominating the error, so it should not be read as parity.

## Caveats
- "State of the art" here means tuned gradient-boosted trees and other tree ensembles, which are the usual winners on small tabular data. TabPFN, CatBoost and tabular deep-learning models were not run.
- The three tables are not a published leaderboard. OCM is the public CADS data the manuscript cites; DRM and WGS are literature tables compiled for CatFORGE. The QEERI laboratory data were not used because their mapping to the model inputs is unresolved.
- Baselines got a modest tuning budget. CatFORGE was run exactly as shipped, including its own internal grid search (which uses ungrouped CV).
- The best other model is picked per cell after seeing results, a strict comparator that makes verdicts conservative for CatFORGE. The three repeats share data, so CIs are approximate.
- The H2/CO exclusion is a post-hoc diagnostic, not a pre-specified analysis.
- LightGBM 4.7.0 was installed into `benchmark/.venv` for this run.

## Files
- Code: `run_sota.py`, `check_catforge_equivalence.py`, `analyze_sota.py`, `plot_sota.py`.
- Out-of-fold predictions (45 files, 18 MB): `results/sota/<dataset>/rep*_fold*.csv.gz`.
- Tables: `results/sota_summary.csv`, `results/sota_catforge_vs_baselines.csv`, `results/sota_headline.json`.
- Figure: `results/sota_figure.pdf` and `.png`.
