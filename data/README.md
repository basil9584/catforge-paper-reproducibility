# Data

Only the four tables that the scripts in this package actually read are included. They are byte-for-byte copies of
the files the analyses used, renamed to neutral names. For the three literature tables the SHA-256 recorded when the
benchmarks ran is kept in `../results/run_info.json`, and `python reproduce.py --check` confirms that the files still
match it.

| File | What it is | Size | Used by | Source | Licence / permission status |
|---|---|---|---|---|---|
| `drm_literature.csv` | Dry reforming of methane (DRM) over Ni catalysts: catalyst descriptors, reaction conditions, CH4 and CO2 conversion, H2/CO ratio; `Paper no.` identifies the publication | 1,714 rows x 17 columns; 1,564 complete records from 204 publications are used | Figs 2-5, Table 1 (`benchmark/run_benchmark.py`, `figures/compute/fig4_shap_compute.py`, `figures/compute/fig5_cost_compute.py`, `benchmark/test_cost_ranking.py`) | Compiled by the CatFORGE authors from 220 published DRM studies; `Paper no.` is the authors' code for each study (a code-to-reference list is available from the authors). | The authors' own compilation, released with this package under its MIT licence. Please also cite the original studies when reusing it. |
| `wgs_literature.csv` | Low-temperature water-gas shift (WGS) over noble-metal catalysts: composition, preparation, pretreatment and test conditions, CO conversion; `Reference` identifies the publication (given on its first row only) | 4,360 rows x 84 columns; 84 references | Figs 2-3, Table 1 | Compilation of Odabasi, C., Gunay, M. E. & Yildirim, R. *Int. J. Hydrogen Energy* **39**, 5733-5746 (2014), https://doi.org/10.1016/j.ijhydene.2014.01.160 (manuscript ref. odabasi2014) | Published compilation, redistributed unchanged with attribution for reproducibility. Cite the original publication when reusing it. |
| `ocm_literature.csv` | Oxidative coupling of methane (OCM): catalyst composition (cations, anions, promoter, support, mol%), conditions, CH4 conversion, C2 selectivity and yield; no publication column (groups are element sets) | 1,831 rows x 27 columns; 1,744 complete records, 773 element sets | Figs 2-3, Table 1 | Zavyalova, U., Holena, M., Schlogl, R. & Baerns, M. *ChemCatChem* **3**, 1935-1947 (2011), https://doi.org/10.1002/cctc.201100186, as distributed by Catalyst Acquisition by Data Science (CADS): Fujima, J. et al. *React. Chem. Eng.* **5**, 903-911 (2020), https://doi.org/10.1039/D0RE00098A | Public dataset, redistributed unchanged with attribution for reproducibility. Cite the original publications when reusing it. |
| `qeeri_laboratory_comparisons.csv` | Measured (QEERI laboratory) and predicted (earlier CatFORGE runs) CH4 and CO2 conversion and H2/CO ratio for Ni-hydrotalcite, Ni-activated carbon and Ni-alumina catalysts and a Ni-HT calcination series; differences recalculated | 21 rows x 10 columns | Fig. 6, Table 2 (laboratory-test column) | Values as reported in the authors' earlier merged CatFORGE manuscript (`source_pdf_page` gives the page of that internal draft); measurements made at the Qatar Environment and Energy Research Institute (QEERI) | The authors state that disclosure of these measurements is permitted. |

Original file names in the authors' working copy of the platform (uploads): `157310cd-44f1-4842-8af6-8384385c4fbe_Final_Complied.csv`
(DRM), `41a6cee2-a143-408c-b923-07dbb67bf756_big_dataset_trimmed_-_Sheet1_3.csv` (WGS), `3ac13314_OCM_dataset_-2.csv` (OCM),
`Laboratory_Comparisons_Recalculated.csv` (laboratory comparisons).

Notes
- Rows with a missing value in any model input or response are dropped by the loaders (`benchmark/run_benchmark.py`,
  `load_drm`, `load_wgs`, `load_ocm`); the DRM column `Ni Dispersion` is not used (1,056 of 1,714 values blank).
- The laboratory values are historical: the predictions come from earlier platform versions and were not re-run with the
  repaired predictor. The stand-alone `NiHT-600 separate case` (3 rows) is excluded from Fig. 6 because its reported
  time on stream is inconsistent (12 h or 20 h).
- The QEERI DRM modelling tables of the earlier runs (Table 2 of the manuscript) are not included: Table 2 is not
  reproducible from code (see the main README).
- Prices for the cost illustration (Fig. 5) are not a data file: US Geological Survey Mineral Commodity Summaries 2025
  values (nickel US$17,000 and 21,495 per tonne; alumina US$570 and 481 per tonne) are passed to the computation in
  `figures/fig5_cost.py` (US government work, public domain).
