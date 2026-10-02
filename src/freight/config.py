"""Single source of truth for paths, constants and seeds. No absolute paths anywhere else."""
from __future__ import annotations

from pathlib import Path

SEED: int = 42

ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT / "data"
OUTPUTS_DIR: Path = ROOT / "outputs"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
SCORER_DIR: Path = ROOT / "scorer_results"
REPORT_DIR: Path = ROOT / "report"
DOCS_DIR: Path = ROOT / "docs"

TRAIN_PATH: Path = DATA_DIR / "train_test.csv"
VALIDATION_PATH: Path = DATA_DIR / "validation.csv"
TEMPLATE_PATH: Path = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH: Path = DATA_DIR / "december_chart_inputs.csv"
PREDICTIONS_PATH: Path = ROOT / "validation_predictions.csv"

TARGET: str = "posted_rate"
ID_COL: str = "load_id"
EQUIPMENT: tuple[str, ...] = ("Dry Van", "Flatbed", "Reefer")

# Validation design (all dates in 2025)
CV_FIRST_EVAL_MONTH: int = 3   # folds evaluate on months 3..8 (Mar..Aug); each trains on all earlier months
CV_LAST_EVAL_MONTH: int = 8
HOLDOUT_MONTHS: tuple[int, ...] = (9, 10)  # Sep-Oct: touched once, after tuning on Jan-Aug only
YEAR: int = 2025

# December fixed scenario (must match score.py)
DEC_PICKUP: str = "Lexington"
DEC_DELIVERY: str = "Fort Wayne"

# Data-quality constants
WEIGHT_CAP: float = 47_500.0       # observed censoring level of |weight|
LABEL_OUTLIER_K: float = 10.0      # robust-sigma multiplier for corrupted-label detection
EARTH_RADIUS_MILES: float = 3958.8
