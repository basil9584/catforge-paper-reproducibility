# CatFORGE reproducibility package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24452477.svg)](https://doi.org/10.5281/zenodo.24452477)

This package regenerates every figure, table and number in the **Results** of the manuscript that describes
CatFORGE, a browser-based platform for data-guided, cost-aware catalyst design. It contains the analysis code, the
evaluated versions of the three platform modules the results depend on, the datasets the analyses read, and the
stored results of the long benchmark runs. It does **not** contain the CatFORGE web application (see
[What is not included](#what-is-not-included-and-why)).

One command regenerates Figs 1-6 from the stored results and checks every reported number:

```
python reproduce.py --figures --check
```

> Licence: MIT (code and the authors' own data compilation); the published third-party tables keep the attribution
> given in [Data sources and licences](#data-sources-and-licences). Archived on Zenodo: version 1.0.0,
> https://doi.org/10.5281/zenodo.24452477 (all versions: https://doi.org/10.5281/zenodo.24452476).

## Quick start

Python 3.12 (the results were produced with Python 3.12.10 on Windows 11, x86-64).

```
python -m venv .venv
.venv\Scripts\activate              (Windows)      or      source .venv/bin/activate      (Linux, macOS)
pip install -r requirements.txt
python reproduce.py --figures --check
```

`requirements.txt` pins the packages the code imports; `requirements-lock.txt` pins the complete environment
(including indirect dependencies such as the SDV libraries), for the closest possible re-run of the benchmarks.
torch is only needed for the benchmark re-runs. On macOS replace `torch==2.13.0+cpu` by `torch==2.13.0`.

All commands run from the package root. Everything is written below `output/` (logs in `output/logs/`); `data/` and
`results/` are only read.

## What `reproduce.py` does, and how long it takes

| Command | What it does | Time (authors' laptop, Intel Core i7-1185G7, 4 cores) |
|---|---|---|
| `--figures` | Regenerates Figs 1-6 as PDF, SVG and PNG (600 dpi) in `output/figures/`, from the stored results and the data. Each figure script recomputes every plotted value and asserts that what it draws equals what its computation script computed. | about 45 s |
| `--check` | Runs `--figures`, then `checks/check_results.py`: (1) the figure scripts' own verification; (2) 148 key numbers (in 53 checks, plus 25 text labels such as model names and verdicts) transcribed from the manuscript (`results/reference_numbers/manuscript_key_numbers.json`), each recomputed and required to round to the printed value; (3) the stored summary tables behind Figs 2-3 and Table 1 re-derived from the stored per-split and per-record benchmark outputs; (4) data files and predictor code against the SHA-256 recorded when the benchmarks ran, and the locked test groups recomputed from the code; (5) Fig. 1 labels and format; (6) regenerated PNGs compared with the manuscript figures in `results/reference_figures/` (informational). Prints PASS/FAIL per figure and table; report in `output/check/`. | about 1-2 min in total |
| `--tests` | Cost-module tests (`benchmark/test_cost_ranking.py`, 16 checks; result compared with `results/cost_module/cost_ranking_test.json`), `benchmark/check_fig6_equivalence.py`, and `benchmark/check_catforge_v2_equivalence.py` (benchmark replica = platform predictor; regression test of the repaired defect). | about 5-6 min |
| `--quick` | Smoke run of the benchmark code on a small subset: DRM only; augmentation main grid for the first of the three repeats (5 group folds) at 50 real records; locked-test evaluation of all 12 models on the first of the five 50-record training draws. **Its summary numbers are not the manuscript's** (they cover a small part of the design). Because the subset uses the same seeds as the matching part of the manuscript run, it is then compared row by row with the stored results (on the authors' machine: augmentation MAE, RMSE, R2 and quality score equal to 1e-14; the 17,172 locked-test predictions equal to 2e-14; the within-publication Spearman correlation, which the manuscript does not use, differed by up to 0.01 in 2-6 of 330 results, varying between runs, because random-forest predictions that differ in the 15th digit swap nearly tied ranks). Output in `output/quick/`. | about 3-4 min |
| `--benchmarks` | Full re-run of the augmentation benchmark (main and ablation grids, three datasets) and the locked-test predictor benchmark, with the original seeds and settings, then comparison of the re-run tables with the stored ones. Prints an estimate and asks for confirmation (`--yes` skips the question; `--jobs N` runs N jobs in parallel). Output in `output/benchmarks/`; `python reproduce.py --figures --results-dir output/benchmarks` then redraws Figs 2-3 from the re-run. **Not re-run for this release** (see "What was verified"). | about 2.5 h with `--jobs 6` (measured for the original runs); roughly 5-10 h with one job (estimate) |

The development steps that led to the repaired predictor (first-round benchmark `benchmark/run_sota.py` +
`analyze_sota.py`, development ablation `benchmark/run_ablation.py` + `catforge_variants.py`) are included because the
final benchmark imports them, but `reproduce.py` does not re-run them (several further hours; their outputs are not
included, see below). They run from the package root, e.g. `python benchmark/run_sota.py --dataset drm`, and write to
`output/benchmarks/`.

## Where every result comes from

Manuscript numbering is used throughout. The authors' working folders called the cost figure "Fig. 6" and the
laboratory figure "Fig. 5"; here they are Fig. 5 and Fig. 6, as in the manuscript.

Each data figure has two scripts: a **computation script** in `figures/compute/` that computes every value (it also
draws a first-draft figure, which is never saved when it is run by the figure script) and a **figure script** in
`figures/` that runs the computation script in a separate process, plots exactly the values it computed, saves the
figure and verifies it.

| Manuscript item | Computation | Figure script | Inputs (stored results or data) | Output | Checks |
|---|---|---|---|---|---|
| Fig. 1 (schematic) | none (no data) | `figures/fig1_pipeline.py`, `figures/fig1_icons.py`, icons in `figures/assets/` | none | `output/figures/Fig1_CatFORGE_pipeline.*` | labels present as text, 183 mm wide, vector only; image |
| Fig. 2 (synthetic-data augmentation) | `figures/compute/fig2_synthetic_compute.py` | `figures/fig2_synthetic.py` | `results/augmentation_benchmark/benchmark_summary.csv`, `benchmark_fidelity_vs_utility.csv`, produced by `benchmark/run_benchmark.py` (per-split results: `*_raw.csv` in the same folder) | `output/figures/Fig2_synthetic_fidelity_vs_usefulness.*`, `output/numbers/fig2_numbers.txt` | printed numbers = `results/reference_numbers/fig2_numbers.txt`; key numbers; summary tables re-derived from `*_raw.csv` |
| Fig. 3 (predictor vs ten baselines) | `figures/compute/fig3_benchmark_compute.py` | `figures/fig3_benchmark.py` | `results/predictor_benchmark/final_summary.csv`, `final_catforge_vs_baselines.csv`, produced by `benchmark/run_final.py` + `benchmark/analyze_final.py` (locked-test predictions: `results/predictor_benchmark/final/*.csv.gz`; locked groups: `splits/`) | `output/figures/Fig3_benchmark_heldout.*` | key numbers; tables re-derived from the predictions; locked groups recomputed |
| Table 1 (predictor accuracy on locked test groups) | same tables as Fig. 3 | (table in the manuscript text) | as Fig. 3 | values in `output/check/check_report.txt` | every cell (RMSE, best other model and its RMSE, ratio, 95% interval, before-repair ratio) |
| Fig. 4 (SHAP on the DRM compilation) | `figures/compute/fig4_shap_compute.py` (CatFORGE's SHAP settings) | `figures/fig4_shap.py` | `data/drm_literature.csv` | `output/figures/Fig4_interpretation_shap.*`, `output/numbers/fig4_numbers.txt` | printed numbers = `results/reference_numbers/fig4_numbers.txt`; key numbers |
| Fig. 5 (cost-aware ranking) | `figures/compute/fig5_cost_compute.py`, using `catforge/optimization_catcost.py` (`calculate_materials_cost`, `calculate_topsis_score`) | `figures/fig5_cost.py` (passes the USGS prices `{"ni_low":17.0,"ni_base":17.0,"ni_high":21.495,"al_low":0.481,"al_base":0.570,"al_high":0.570}` US$/kg) | `data/drm_literature.csv` | `output/figures/Fig5_cost_aware_ranking.*`, `output/numbers/fig5_cost_numbers.json` | numbers file JSON-equal to `results/reference_numbers/fig5_cost_numbers.json`; key numbers |
| Fig. 6 (laboratory comparisons) | `figures/compute/fig6_lab_compute.py` | `figures/fig6_lab.py` | `data/qeeri_laboratory_comparisons.csv` | `output/figures/Fig6_laboratory_comparison.*`, `output/numbers/fig6_lab_numbers.txt` | printed numbers = `results/reference_numbers/fig6_lab_numbers.txt`; key numbers |
| Table 2 (four catalytic systems, earlier platform version) | **not reproducible from code** (historical values, see below) | - | - | - | only the laboratory-test column (predicted minus measured conversion) is recomputed from `data/qeeri_laboratory_comparisons.csv` |
| Cost module behaviour (per-record and per-candidate cost, user cost weight; interface cost ranks the Fig. 5 candidates as the raw-material cost does) | `benchmark/test_cost_ranking.py`, `benchmark/check_fig6_equivalence.py` | - | `data/drm_literature.csv`; `catforge/optimization_catcost.py` and, for the backward-compatibility test, `catforge/previous_versions/` | `output/tests/cost_ranking_test.json` | `--tests`: 16/16 checks; JSON equal to `results/cost_module/cost_ranking_test.json` |
| Defect found by the first benchmark round ("H2/CO predictions of -140 to -300") | `benchmark/check_catforge_v2_equivalence.py` | - | `data/drm_literature.csv` | console / `output/logs/` | `--tests`: -301.7 before repair, 0.62-0.69 after (the -140 value, about -144, comes from the first-round run; see below) |

Key numbers checked (all from the manuscript's Results text, figure captions and Table 1; ids in
`results/reference_numbers/manuscript_key_numbers.json`):

| Item | Numbers |
|---|---|
| Fig. 2 | dataset sizes (DRM 1,564 records / 204 publications; WGS 4,360 / 84 references; OCM 1,744); 25-400 training records; change in MAE within 2% in 21 of 30 settings, significant decrease in 2, increase in 4; maximum +12.8% (7.2-19.8%) with 25 records from a few DRM publications; synthetic-only worse in 29 of 30, up to 106%; CTGAN and TVAE up to 10.6% and 6.1%; Spearman -0.39 to -0.52 (DRM) and -0.18 to 0.32 (WGS, OCM); TVAE highest quality for WGS and OCM yet worse; 15 splits |
| Fig. 3 | locked test groups 61 / 25 / 232; twelve models; tie for DRM and WGS, loss for OCM; ratios 0.98-1.03; OCM 6-8% worse, intervals 1.003-1.133; mean rank 3.4 (LightGBM 3.0, XGBoost 3.8, before repair 5.9); within 8.2% with 50 or 200 records |
| Table 1 | all 7 rows x 6 numbers and the name of the best other model |
| Fig. 4 | 1,564 records, 204 publications; temperature 39%, 42%, 23% and ranked first in all five refits; 400-1,000 C; Spearman 0.96 and -0.74; R2 0.85-0.92 (fitted) and 0.16-0.49 (held-out publications) |
| Fig. 5 | 1,564 records; prices US$17.0 and 0.57 per kg; ten-model surrogate; 63.9% at 2 wt% and 66.9% at 5 wt%, then within 0.8 points of the 5 wt% value (largest distance 0.72); cost US$0.90-4.68 per kg; measured records from 9 publications, 32-77%, GHSV 12,000-72,000; performance-only choice 15 wt%, gain 0.7 points; held-out RMSE 17.6; cost weight 0.05 -> 5 wt%, 0.15 or more -> 2.5 then 2 wt%, same for every price case |
| Fig. 6 | 7 of 14 conversion predictions within 5 points; largest difference 12.2; calcination: measured peak 600 C, predicted peak 300 C with monotonic decline; Ni-HT, Ni-AC and Ni-alumina predicted and measured conversions and H2/CO ratios |
| Table 2 | laboratory-test column (+5.9; -2.2 / +4.5; -0.3 / -1.7; -6.8) |

Rounding note: the check accepts a value if it rounds to the printed one (within half a unit of the last printed
digit). One value sits on the rounding boundary: the lowest WGS/OCM Spearman correlation is exactly -0.175 (printed
-0.18). The bound "within 0.8 points of that value" is checked as an upper bound: above 5 wt% the predicted CH4
conversion stays within 0.72 points of its 5 wt% value (the full spread, highest minus lowest, is 0.84 points).

## Numbers that this package does not reproduce

These values are historical: they were transcribed from the authors' earlier merged manuscript, which used earlier
platform versions, models trained on augmented data and undocumented splits. The exact inputs, versions and settings
of those runs are not documented, so no code in this package regenerates them.

- **Table 2**, all columns except the laboratory-test column: real/synthetic record counts, quality scores (78.2%,
  71.8%, 75.5%, 77.0%), cross-validated R2 and SHAP drivers of the earlier runs.
- Results, "Synthetic data reproduce the main statistical structure": CTGAN 88.3% vs CopulaGAN 81.8% quality,
  correlation similarity 0.94 vs 0.88, Kolmogorov-Smirnov complement 0.80 and 0.004, total-variation complement 0.87
  and 0.82.
- Results, first paragraph of "Benchmarking the predictive models": the earlier comparison (R2 0.812, MSE 57.95;
  LightGBM 0.817 / 53.78; XGBoost 0.803 / 60.30; random forest 0.770 / 71.58; linear regression 0.572 / 151.38).
- Results, first paragraph of "Interpretability and physical consistency": SHAP findings of the earlier model on
  augmented data (for example the plateau near 14 wt% Ni).
- Results, case study in "End-to-end candidate design and cost": the suggested candidates and their predicted
  properties (CB-LDO variant, Ni-Ce/SiO2 variant, Ni/ZrO2 and NAC-0.8% Gd2O3 variants).
- Results, "Validation across four catalytic systems": the per-dataset quality scores, cross-validated R2 and
  SHAP/LIME findings of the earlier runs. (The measured and predicted laboratory values are in
  `data/qeeri_laboratory_comparisons.csv` and are checked; the predictions themselves come from the earlier runs and
  were not re-predicted.)
- The value -140 in "H2/CO predictions of -140 to -300" comes from the first-round benchmark (about -144 in
  `docs/SOTA_BENCHMARK_REPORT.md`); its out-of-fold predictions (45 files, 19 MB) are not included, but
  `benchmark/run_sota.py` regenerates them (several hours). The -300 value is reproduced by `--tests`.

## Package layout

```
README.md, LICENSE, CITATION.cff, .zenodo.json, .gitignore, requirements.txt, requirements-lock.txt, reproduce.py
catforge/        verbatim copies of the platform modules the results depend on (see below)
benchmark/       augmentation benchmark, predictor benchmark (first round, development ablation, locked test),
                 analyses, cost-module tests; repro_paths.py holds all package paths
figures/         figure scripts (final design), style module, Fig. 1 icons; compute/ = computation scripts
checks/          check_results.py (used by reproduce.py --check)
data/            the four input tables (README.md: sources and licences)
results/         stored results: augmentation_benchmark/ (summary tables + per-split results),
                 predictor_benchmark/ (summary tables, locked-test predictions, locked splits), cost_module/,
                 reference_numbers/ (stored printed numbers + manuscript key numbers), reference_figures/ (the PNGs
                 of the manuscript figures), run_info.json (run metadata: versions, seeds, data SHA-256)
docs/            the authors' benchmark reports (protocol of the predictor repair and the locked test)
```

`docs/REPAIR_REPORT.md` and `docs/SOTA_BENCHMARK_REPORT.md` are the benchmark reports as written during the work;
file paths in them refer to the authors' working folders (for example `benchmark/results/` there is
`results/` here).

### Platform code (verbatim)

| File in this package | Platform path | SHA-256 | Used by |
|---|---|---|---|
| `catforge/generalized_predictor.py` | `backend/models/prediction/generalized_predictor.py` | `02b2ed3250680680f4d50577635972afe89faafad7bef80bb579764167d8c535` (the hash recorded by the locked-test run) | predictor benchmark (Fig. 3, Table 1) |
| `catforge/optimization_catcost.py` | `backend/models/optimization/optimization_catcost.py` (after the 22 Nov 2026 cost fix) | `8e3f4d16fc6166b02c44b4dc4433cafb5ce3c21674ef543ba239059a7e6c74ca` | Fig. 5, cost tests |
| `catforge/data_synthesizer.py` | `backend/models/synthesis/data_synthesizer.py` | `f98746bba7da8925bd7e1a011b741bf07cfa3102b037a350ac78aa74d56c360a` | augmentation benchmark (Fig. 2): CatFORGE's synthesizer |
| `catforge/previous_versions/optimization_catcost_before_2026-11-22_cost_fix.py` | same module before the cost fix | `50112fd55ca72bf96d1259bad9dea07c0a1500c516797f2e88abc77169071012` | only the backward-compatibility test S0 in `benchmark/test_cost_ranking.py` |

The modules import only third-party packages (no other platform code). `generalized_predictor.py` imports `shap` for
explanation plots only; the benchmark scripts put a stub on the path (`benchmark/_stubs/shap.py`), as the original
runs did. `lime` is optional (the module prints "Warning: LIME not found" and continues).

### Changes to the analysis scripts for this package

The analysis code is the code that produced the stored results, with these changes only (no change to models,
settings, seeds, splits or metrics; `--check` and `--tests` confirm identical outputs):
- absolute paths of the authors' working folders replaced by package paths (`benchmark/repro_paths.py`); re-runs
  write below `output/`, never into `results/`;
- `benchmark/run_final.py`: options `--out`, `--regimes`, `--draws` (defaults = the manuscript run; the last two
  are used only by `--quick`); `benchmark/analyze_final.py`: input and output folders as arguments;
  `benchmark/run_benchmark.py`: can skip its closing summary when several runs go in parallel
  (`CATFORGE_SKIP_SUMMARY=1`); `benchmark/check_fig6_equivalence.py`: non-zero exit code on failure;
  `benchmark/run_sota.py`: writes the shap stub only if it is missing;
- figure scripts renamed to manuscript numbering (working names: `fig1_v2.py` ... `fig6_v2.py` and, in
  `figures/compute/`, `fig2_synthetic.py`, `fig3_benchmark.py`, `fig4_shap.py`, `fig6_cost.py` -> Fig. 5,
  `fig5_lab.py` -> Fig. 6); they now also write the regenerated numbers to `output/numbers/`; the v1/v2 contact
  sheets of the design work are not produced;
- stored reference numbers: the first line of the three text files (a "saved <path>" message of the original run)
  is removed; `fig6_numbers.json` is renamed `fig5_cost_numbers.json`, content unchanged; the nine run-info files
  are merged into `results/run_info.json` with the data-file paths replaced by package paths;
- `benchmark/recompute_quality.py` is included for provenance only: it corrected the quality-score column of the
  first augmentation runs (2 Nov 2026), so the stored per-split results already carry the corrected score, and
  `run_benchmark.py` now computes it directly.

## Data sources and licences

| File | Content | Source | Licence / permission |
|---|---|---|---|
| `data/drm_literature.csv` | DRM literature compilation, 1,714 rows (1,564 complete records, 204 publications used) | compiled by the CatFORGE authors from 220 published DRM studies (`Paper no.` codes) | the authors' own compilation, released with this package under its MIT licence; please also cite the original studies |
| `data/wgs_literature.csv` | water-gas shift compilation, 4,360 records, 84 references | Odabasi, Gunay & Yildirim, *Int. J. Hydrogen Energy* 39, 5733 (2014) | published compilation, redistributed unchanged with attribution for reproducibility; cite the original publication when reusing |
| `data/ocm_literature.csv` | OCM compilation, 1,831 rows (1,744 used) | Zavyalova et al., *ChemCatChem* 3, 1935 (2011), via CADS (Fujima et al., *React. Chem. Eng.* 5, 903 (2020)) | published dataset, redistributed unchanged with attribution for reproducibility; cite the original publications when reusing |
| `data/qeeri_laboratory_comparisons.csv` | QEERI laboratory measurements and earlier CatFORGE predictions | QEERI (Hamad Bin Khalifa University); values from the authors' earlier manuscript | the authors state disclosure is permitted |
| `figures/assets/lucide-static-0.469.0/` | 8 line icons for Fig. 1 | Lucide 0.469.0 (https://lucide.dev) | ISC (portions MIT, Feather); licence file kept with the icons |

Details in `data/README.md`.

## What is not included, and why

- **The CatFORGE web application**: browser interface, back-end web service, user accounts and authentication,
  databases, hosting and deployment configuration, and all credentials and keys. None of the reported numbers needs
  them (the results come from the three platform modules above, called directly); running them would require a
  server and secrets, and publishing them would expose account data and keys.
- **Other platform code and other users' uploads**: only modules that the results depend on are included.
- **Large intermediates that no reported number needs**: out-of-fold predictions of the first-round benchmark
  (45 files, 19 MB) and of the development ablation (19 files, 31 MB), and the quality-score recomputation
  tables. The scripts that produce them are included.
- **The earlier runs behind Table 2** and the other historical values listed above: their inputs and settings are
  not documented (see above).
- **First-draft figures** and design contact sheets.

## Notes on running elsewhere

- The figures use Arial. On a machine without Arial, matplotlib falls back to another font: the numbers and checks
  are unaffected, but the PNGs will not be pixel-identical to `results/reference_figures/`, the layout checks may
  report overlaps, and Fig. 1, which measures its text, may stop with "card text wider than the card". On Linux,
  install the Microsoft core fonts (e.g. the `ttf-mscorefonts-installer` package) to get identical figures.
- The package was run on Windows 11. It uses only portable Python, but Linux and macOS were not tested.
- Benchmark re-runs use fixed seeds; tree ensembles with several threads and CTGAN/TVAE training can still differ in
  the last digits between machines and library builds, and the within-publication Spearman correlations (not used in
  the manuscript) can shift slightly when nearly tied predictions swap rank. `--benchmarks` reports any difference
  from the stored tables, with the column in which it is largest.

## What was verified for this release (22 Nov 2026, authors' laptop, Windows 11, Python 3.12.10)

- `python reproduce.py --figures --check`: all six figures regenerated; every check passed (Figs 1-6, Table 1,
  Table 2 laboratory column, data and code integrity); all six PNGs byte-identical to the manuscript figures; the
  re-derived summary tables byte-identical to the stored ones.
- `python reproduce.py --tests`: 16/16 cost-module checks, result file equal to the stored one; interface cost ranks
  the Fig. 5 candidates identically; benchmark replica = platform predictor (difference 0); regression test as in
  `docs/REPAIR_REPORT.md`.
- `python reproduce.py --quick`: completed; the subset matched the stored run as described in the table above.
- `python reproduce.py --benchmarks` was **not** run for this release (hours). The stored results it would regenerate
  were produced by this code (apart from the path changes and the quality-score correction described above) on
  2 Nov 2026 (augmentation) and 18 Nov 2026 (locked test); `--quick` replays a slice of both with the original seeds.

## How to cite

Please cite the article (reference to be added on publication) and this package:

> Elshenawy, B., Khan, M. Q. A., Varlani, M., Abotaleb, A. & Elshrif, M. CatFORGE reproducibility package (version
> 1.0.0). Zenodo https://doi.org/10.5281/zenodo.24452477 (2026).

`CITATION.cff` and `.zenodo.json` hold the same metadata. The DOI above identifies version 1.0.0; the concept DOI
https://doi.org/10.5281/zenodo.24452476 always resolves to the latest version.

## Licence

Code in this package, including the platform modules in `catforge/`, and the authors' DRM literature compilation:
MIT licence (`LICENSE`, "Copyright (c) 2026 the authors"). The WGS and OCM tables are third-party published data,
redistributed with attribution (see above); the Lucide icons are ISC-licensed.
