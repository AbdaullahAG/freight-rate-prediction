# Freight Rate Prediction (Spotter ML Engineer take-home)

Predict the posted rate of 12,000 future freight loads (`data/validation.csv`, Nov-Dec 2025) from 48,000 labelled loads
(`data/train_test.csv`, Jan-Oct 2025), validate with time-based splits only, and produce the fixed December chart.
**No accuracy is claimed on `validation.csv`** (it has no labels; Spotter computes the final score).

## Where to put the raw files
`data/` is git-ignored on purpose (raw assessment data). Copy the four assessment CSVs into `data/` using these names:
`train_test.csv`, `validation.csv`, `validation_predictions_template.csv`, `december_chart_inputs.csv`.
(The uploaded files use hyphens; the assessment README uses underscores.)

## Setup (tested: Python 3.12.3, Linux, single CPU)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # pinned; pandas 2.3.3 because the provided requirements cap pandas at <3
```

## Run
One command (about 12 minutes, seed 42; installs requirements, runs tests, the whole pipeline, score.py and builds the report):
```bash
bash run_all.sh                 # add --skip-tune to reuse outputs/best_params.json instead of re-running Optuna
```
Step by step:
```bash
python -m pytest -q tests
python scripts/01_eda.py                    # EDA, data-quality table, plots, adversarial validation
python scripts/02_validation_design.py      # time-based folds + baseline cleaning effects
python scripts/03_features_ablation.py      # target choice, signal/feature ablations (CV folds only)
python scripts/04_models.py                 # Optuna (<=20 trials/model), CV + Sep-Oct holdout, model selection
python scripts/04b_hybrid_tree_features.py  # follow-up check on the hybrid
python scripts/05_final_predictions.py      # refit, December scenario, validation_predictions.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python scripts/06_report.py                 # report/Freight_Rate_Report.docx
```

## Folder tree
```
src/freight/   config, data, quality (label outliers, haversine), features (leak-free lane stats), cv (time splits, metrics),
               models (ridge, lightgbm, catboost, hybrid, routed fallback), evaluate, december, predict, report
scripts/       01_eda ... 06_report (numbered pipeline steps)
tests/         time-ordering per fold, leakage of lane features, shapes, ID match, December fixed values, monotonicity
outputs/       tables (csv/json), plots/, checkpoints.md, decision_log.md
docs/          validation_design.md, original assessment files, PROJECT_EXPLAINED_AR.md (Arabic study guide)
report/        Freight_Rate_Report.docx
score.py       provided scorer, unmodified
```

## Outputs
`validation_predictions.csv` (exactly `load_id,predicted_rate`), `data/december_chart_inputs.csv` (filled), `scorer_results/candidate_december.png`,
`report/Freight_Rate_Report.docx`, `outputs/*` (metrics, issue table, importance, residuals).

## Approach in short
* Time-based validation: 6 expanding monthly folds (Mar-Aug) for tuning, Sep-Oct holdout, refit on all train. Random splits are never used.
* Cleaning: 677 corrupted labels (1.41%) removed from training, `abs(weight)`, missing values handled, `quote_signal` excluded (unstable), `market_index` kept.
* Model: hybrid = Ridge on log(rate) (distance, equipment, market_index) + LightGBM on its residuals (geometry, weight, coordinates, time-respecting lane statistics),
  with a fallback model without lane statistics for lanes/cities without history.
* `validation.csv` features are used only for adversarial validation and for the December `market_index`/`quote_signal` fill.

## Results (Mar-Aug CV, Sep-Oct holdout; MAPE % on clean rows)
| model | CV mean | CV std | last fold | Sep-Oct holdout |
|---|---|---|---|---|
| ridge | 3.083 | 0.481 | 3.120 | 4.635 |
| lightgbm | 2.300 | 0.663 | 1.764 | 2.724 |
| catboost | 2.272 | 0.660 | 1.763 | 2.770 |
| **hybrid (chosen)** | 2.178 | 0.705 | 1.608 | 2.696 |
| blend | 2.160 | 0.684 | 1.653 | 2.762 |

## Limitations
Systematic under-prediction of ~2.2-3.2% on the Sep-Oct holdout; December seasonality is not learned and trees do not extrapolate;
8 cities exist only in validation (fallback model, ~4.6% MAPE in the cold-start diagnostic); corrupted labels would dominate the error if present in validation;
the model-selection rule was revised after seeing holdout results (see `outputs/decision_log.md` #17). Full list in the report.
