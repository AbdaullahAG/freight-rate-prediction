"""Time-based validation design. Random splits are never used (validation.csv is the future).

* Expanding-window monthly folds on Jan-Aug only (Optuna tunes only here).
* Final holdout: train Jan-Aug, evaluate once on Sep-Oct (selection among tuned candidates / reporting).
* Then refit on all of train (Jan-Oct) for the final predictions.
* Optional cold-start holdout: like the holdout, but a set of cities is excluded from training,
  mimicking the 8 cities that appear only in validation.csv.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass(frozen=True)
class Split:
    name: str
    train_idx: np.ndarray
    eval_idx: np.ndarray


def _month(df: pd.DataFrame) -> pd.Series:
    return df["date"].dt.month


def expanding_monthly_folds(df: pd.DataFrame, first: int = C.CV_FIRST_EVAL_MONTH,
                            last: int = C.CV_LAST_EVAL_MONTH) -> list[Split]:
    """Fold k: train = all rows in months < k, eval = rows in month k. Only months <= `last` are used."""
    m = _month(df).to_numpy()
    folds = []
    for k in range(first, last + 1):
        tr, ev = np.flatnonzero(m < k), np.flatnonzero(m == k)
        if len(tr) and len(ev):
            folds.append(Split(f"fold_m{k:02d}", tr, ev))
    return folds


def final_holdout(df: pd.DataFrame) -> Split:
    m = _month(df).to_numpy()
    return Split("holdout_sep_oct", np.flatnonzero(m < C.HOLDOUT_MONTHS[0]),
                 np.flatnonzero(np.isin(m, C.HOLDOUT_MONTHS)))


def cold_start_holdout(df: pd.DataFrame, n_cities: int = 8, seed: int = C.SEED) -> tuple[Split, list[str]]:
    """Train Jan-Aug on rows whose cities are NOT in the held-out set; eval Sep-Oct rows touching them."""
    cities = sorted(set(df["pickup"]) | set(df["delivery"]))
    rng = np.random.default_rng(seed)
    held = sorted(rng.choice(cities, size=n_cities, replace=False).tolist())
    touches = (df["pickup"].isin(held) | df["delivery"].isin(held)).to_numpy()
    m = _month(df).to_numpy()
    tr = np.flatnonzero((m < C.HOLDOUT_MONTHS[0]) & ~touches)
    ev = np.flatnonzero(np.isin(m, C.HOLDOUT_MONTHS) & touches)
    return Split("cold_start_sep_oct", tr, ev), held


def assert_no_time_leak(df: pd.DataFrame, split: Split) -> None:
    tr_max, ev_min = df["date"].iloc[split.train_idx].max(), df["date"].iloc[split.eval_idx].min()
    if not tr_max < ev_min:
        raise AssertionError(f"{split.name}: max train date {tr_max} >= min eval date {ev_min}")
    if np.intersect1d(split.train_idx, split.eval_idx).size:
        raise AssertionError(f"{split.name}: train/eval row overlap")


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Spotter's final metric is not disclosed, so report several complementary ones."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = p - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape_pct": float(100 * np.mean(np.abs(err) / y)),
        "smape_pct": float(100 * np.mean(2 * np.abs(err) / (np.abs(y) + np.abs(p)))),
        "rmsle": float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2))),
        "r2": float(1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2)),
        "median_ape_pct": float(100 * np.median(np.abs(err) / y)),
    }
