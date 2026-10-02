#!/usr/bin/env bash
# One-command reproduction from the raw CSVs in data/ (see data/README.txt). Seed 42, single CPU, ~12 minutes.
# Usage:  bash run_all.sh            (uses `python`; override with PYTHON=/path/to/python)
#         bash run_all.sh --skip-tune   (reuse outputs/best_params.json instead of re-running Optuna)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
TUNE_FLAG="${1:-}"
$PY -m pip install -q -r requirements.txt
for f in train_test validation validation_predictions_template december_chart_inputs; do
  [ -f "data/$f.csv" ] || { echo "ERROR: data/$f.csv missing - copy the raw assessment CSVs into data/"; exit 1; }
done
$PY -m pytest -q tests --ignore=tests/test_predictions.py   # data/leakage tests (prediction tests need the outputs)
$PY scripts/01_eda.py                       # Stage 2: EDA, data-quality table, plots, adversarial validation
$PY scripts/02_validation_design.py         # Stage 3: time-based folds, baseline cleaning effects
$PY scripts/03_features_ablation.py         # Stage 4: target + feature ablation (CV folds only)
$PY scripts/04_models.py $TUNE_FLAG         # Stage 5: Optuna (<=20 trials/model), CV + Sep-Oct holdout, selection, importance
$PY scripts/04b_hybrid_tree_features.py     # follow-up check on hybrid trees
$PY scripts/05_final_predictions.py         # Stages 6-7: refit, December, validation_predictions.csv
$PY score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
$PY scripts/06_report.py                    # Stage 9: report/Freight_Rate_Report.docx
$PY -m pytest -q tests                      # full test suite incl. prediction/December/monotonicity tests
