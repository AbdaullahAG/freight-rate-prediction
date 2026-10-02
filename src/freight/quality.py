"""Data-quality helpers: geometry, corrupted-label detection, robust statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    a1, b1, a2, b2 = map(np.radians, (np.asarray(lat1), np.asarray(lon1), np.asarray(lat2), np.asarray(lon2)))
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((b2 - b1) / 2) ** 2
    return 2 * C.EARTH_RADIUS_MILES * np.arcsin(np.sqrt(h))


def robust_sigma(x: pd.Series | np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def label_residuals(df: pd.DataFrame) -> pd.Series:
    """Residual of log(rate) from a deliberately simple, outlier-insensitive structural fit.

    Features: equipment, log-distance (quadratic), market_index (date-median imputed).
    The fit is used ONLY to flag corrupted training labels, never to predict.
    Outliers are so extreme (x0.17-0.47 or x2.1-5.3) that a plain least-squares fit followed by
    median-centering is sufficient; the clean residual spread is ~0.047 (robust sigma).
    """
    mi = df["market_index"].fillna(df.groupby("date")["market_index"].transform("median"))
    mi = mi.fillna(mi.median())
    ld = np.log(df["distance"])
    X = pd.get_dummies(df["equipment"], dtype=float)
    X = X.assign(ld=ld, ld2=ld**2, mi=mi)
    y = np.log(df[C.TARGET])
    # two-pass fit: refit after dropping gross outliers from the first pass
    beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
    r = y - X.values @ beta
    keep = (r - r.median()).abs() < 0.5
    beta = np.linalg.lstsq(X.values[keep.values], y.values[keep.values], rcond=None)[0]
    r = y - X.values @ beta
    return (r - r.median()).rename("log_resid")


def flag_label_outliers(df: pd.DataFrame, k: float = C.LABEL_OUTLIER_K) -> pd.DataFrame:
    """Return frame with log_resid and is_label_outlier (|resid| > k * robust sigma)."""
    r = label_residuals(df)
    sigma = robust_sigma(r[(r.abs() < 0.5)])
    out = pd.DataFrame({"log_resid": r, "is_label_outlier": r.abs() > k * sigma})
    out.attrs["sigma"] = sigma
    return out
