# Legacy-DWTP ClO2 Pre-oxidation ML — Reusable Time-aware Validation Template

Accompanying data and code for:

> Cold-start Machine Learning for Chlorine Dioxide Pre-oxidation Management in a Legacy Water Treatment Plant: A Reusable Time-aware Validation Template for Joint Soft-sensing of Pre-oxidation Dose and Chlorite
> M. Wu, Y. Liu, T. Liu, M. Li, J. Liu, S. Liu, S. Liu

This repository provides a **de-identified** operational archive and the analysis
pipeline that implements the paper's four-part template:
(1) column-lineage audit (Box 1), (2) strictly time-aware validation
(sliding-window, fold-nested KNN imputation, persistence/climatology baselines,
Diebold–Mariano tests), (3) joint soft-sensing of pre-oxidation dose and
finished-water chlorite, and (4) honest uncertainty + skill reporting
(bootstrap 95% CI, Y-randomization, skill scores vs persistence).

## De-identification note
The raw operational archive has been **de-identified**: absolute calendar dates
(`date_raw`, `date` / `dt`) were removed. Chronological order is preserved via
`record_id`, and `year`/`month`/`day` (or `doy`) are retained for seasonality.
The treatment plant's location (Dongying, Shandong, China) is disclosed in the
paper text by the authors' choice and is **not** present in these data files.
No operator names or other direct identifiers are included.

## ⚠️ Column-label audit note (why two data files)
The paper's contribution includes a column-lineage audit (Box 1) that catches
"column confusion" between similarly named dosing channels. This repository
ships TWO files for full transparency:
- `modeling_dataset_729.csv` — the analysis-ready, QC'd, **correctly labeled**
  set (729 daily records). The pre-oxidation dose is the `pre` column
  (0.07–1.93 mg/L, actively adjusted); the fixed terminal setpoint is `post`
  (0.10–0.20 mg/L). This is the file all reported numbers are computed from.
- `plant_operations_raw_deidentified.csv` — a faithful de-identification of the
  **earlier raw parse**. Its `pre_dose` column was found by the audit to be the
  *fixed terminal dose* (0.10–0.20 mg/L), **not** the pre-oxidation dose — exactly
  the confusion Box 1 guards against. We renamed it `terminal_dose_logged` so the
  repository does not propagate the mislabel. The genuine pre-oxidation dose
  time series lives only in `modeling_dataset_729.csv` (`pre`). Re-running
  `audit_column_lineage.py` on the raw file reproduces the flag; on the modeling
  file it passes.

## Repository layout
```
data/
  modeling_dataset_729.csv                # analysis-ready set (729 records, dates removed) — PRIMARY
  plant_operations_raw_deidentified.csv   # de-identified earlier raw parse (1073 records, dates removed)
code/
  analyze_ly0812_Afix.py                 # A-class results: change-day stratification, seasonal ablation,
                                         #   lagged persistence (lag-1/lag-2), Diebold–Mariano, MASE
  analyze_ly0812_review.py               # bootstrap 95% CI, skill scores (ΔR², MASE), sMAPE,
                                         #   stratified MAPE, 50-iter Y-randomization, window sensitivity
  analyze_ly0812_supp.py                 # supplementary: global Y-randomization + aligned persistence
  audit_column_lineage.py                # Box 1 column-lineage audit (transferable preprocessing gate)
results/                                 # JSON outputs written by the scripts above
```

## Data dictionary

### modeling_dataset_729.csv (729 rows) — PRIMARY
| column | meaning |
|---|---|
| record_id | chronological index (1..729), preserves time order |
| year | calendar year |
| doy | day of year |
| finres | finished-water residual ClO2 (mg/L) |
| turb | raw-water turbidity (NTU) |
| finturb | finished turbidity (NTU) |
| nh3 | raw-water NH3-N (mg/L) |
| codmn | raw-water permanganate index CODMn (mg/L) |
| fincod | finished-water CODMn (mg/L) |
| bact | raw-water bacterial count (CFU/mL) |
| flow | influent flow (m3/d) |
| clear | clearwell residual ClO2 (mg/L) |
| post | terminal (post) ClO2 dose (mg/L) — fixed setpoint |
| pre | pre-oxidation ClO2 dose (mg/L) — TARGET 1 (0.07–1.93) |
| doseT | total ClO2 dose = pre + post (mg/L) |
| chlo | finished-water chlorite (mg/L) — TARGET 2 |
| doy_sin, doy_cos | seasonality features (sine/cosine of day-of-year) |

### plant_operations_raw_deidentified.csv (1073 rows)
De-identified earlier raw operational parse (dates removed). Columns:
record_id, res_ClO2, raw_turb, fin_turb, NH3N, CODMn_raw, CODMn_fin, bacteria,
flow, clearwell_ClO2, terminal_dose_logged (renamed from the mislabeled
`pre_dose`; this is the fixed terminal dose, NOT the pre-oxidation dose),
year, month, day.

## Reproduce
```bash
pip install -r requirements.txt
cd code
python analyze_ly0812_Afix.py      # -> ../results/results_ly0812_Afix.json
python analyze_ly0812_review.py    # -> ../results/results_ly0812_review.json  (bootstrap CI, Y-rand)
python analyze_ly0812_supp.py      # -> ../results/results_ly0812_supp.json
python audit_column_lineage.py     # runs Box 1 audit on the de-identified modeling dataset (passes)
```
Expected headline figures (sliding-window, seed 42): pre-oxidation dose
R2 = 0.699 (95% CI 0.60–0.78); chlorite R2 = 0.840 (95% CI 0.81–0.87); both
negative vs lag-1 persistence (MASE > 1), indistinguishable from lag-2 persistence.

## License
Code: MIT. Data: CC-BY-4.0. Please cite the paper above when using this material.
