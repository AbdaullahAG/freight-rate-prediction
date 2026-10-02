"""Loading raw CSVs and light, label-free normalisation."""
from __future__ import annotations

import pandas as pd

from . import config as C


def _read(path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}. Copy the raw CSVs into data/ (see data/README.txt).")
    return pd.read_csv(path)


def load_train() -> pd.DataFrame:
    df = _read(C.TRAIN_PATH, "train_test.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_validation() -> pd.DataFrame:
    df = _read(C.VALIDATION_PATH, "validation.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_template() -> pd.DataFrame:
    return _read(C.TEMPLATE_PATH, "validation_predictions_template.csv")


def load_december() -> pd.DataFrame:
    df = _read(C.DECEMBER_PATH, "december_chart_inputs.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def fix_weight_sign(df: pd.DataFrame) -> pd.DataFrame:
    """Negative weights are sign errors (|weight| distribution matches the positives): take abs()."""
    out = df.copy()
    out["weight"] = out["weight"].abs()
    return out


def city_coordinates(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Median lat/lon per city using both pickup and delivery columns of all given frames."""
    parts = []
    for f in frames:
        parts.append(f[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
        parts.append(f[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).groupby("city").median()
