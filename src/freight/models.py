"""Model wrappers. All take prepared feature frames and return predictions on the RATE scale.

Target modes: log_rate -> exp(pred); rpm -> pred*distance; log_rpm -> exp(pred)*distance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import config as C

LGB_DEFAULT = dict(n_estimators=500, learning_rate=0.04, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, reg_lambda=1.0, random_state=C.SEED, deterministic=True, force_row_wise=True,
                   n_jobs=1, verbose=-1)
CAT_DEFAULT = dict(iterations=500, learning_rate=0.06, depth=6, l2_leaf_reg=3.0, random_seed=C.SEED, thread_count=1, verbose=0,
                   allow_writing_files=False)


def to_target(y_rate: pd.Series, distance: pd.Series, mode: str) -> np.ndarray:
    y, d = np.asarray(y_rate, float), np.asarray(distance, float)
    return {"log_rate": np.log(y), "rpm": y / d, "log_rpm": np.log(y / d)}[mode]


def from_target(pred: np.ndarray, distance: pd.Series, mode: str) -> np.ndarray:
    d = np.asarray(distance, float)
    return {"log_rate": np.exp(pred), "rpm": pred * d, "log_rpm": np.exp(pred) * d}[mode]


def linear_design(X: pd.DataFrame, use_market: bool, use_weight: bool = False, use_trend: bool = False) -> pd.DataFrame:
    """Distance + equipment (+ optional market_index / weight) design for the linear component."""
    ld = X["log_distance"]
    D = pd.DataFrame(index=X.index)
    for e in range(3):
        m = (X["equip"] == e).astype(float)
        D[f"eq{e}"], D[f"eq{e}_ld"], D[f"eq{e}_ld2"] = m, m * ld, m * ld**2
    if use_market:
        D["mi"] = X["market_index"]
        for e in (1, 2):
            D[f"mi_eq{e}"] = X["market_index"] * (X["equip"] == e)
        D["mi_ld"] = X["market_index"] * ld
    if use_trend:
        D["tday"] = X["tday"]
    if use_weight:
        w = X["weight"].fillna(X["weight"].median()) / 1e4
        D["w"], D["w2"] = w, w**2
    return D


@dataclass
class RidgeModel:
    features: list[str]
    alpha: float = 1.0
    mode: str = "log_rate"
    use_market: bool = True
    use_weight: bool = True
    use_trend: bool = False
    name: str = "ridge"
    _m: Ridge = field(default=None, init=False, repr=False)
    _sc: StandardScaler = field(default=None, init=False, repr=False)
    _wmed: float = field(default=0.0, init=False, repr=False)

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        D = linear_design(X.assign(weight=X["weight"].fillna(self._wmed)), self.use_market, self.use_weight, self.use_trend)
        extra = [c for c in self.features if c in ("dow", "quote_signal", "weight_missing", "weight_capped")]
        if "dow" in extra:
            for k in range(1, 7):
                D[f"dow{k}"] = (X["dow"] == k).astype(float)
            extra.remove("dow")
        for c in extra:
            D[c] = X[c]
        return D.values

    def fit(self, X: pd.DataFrame, y_rate: pd.Series) -> "RidgeModel":
        self._wmed = float(X["weight"].median())
        A = self._design(X)
        self._sc = StandardScaler().fit(A)
        self._m = Ridge(alpha=self.alpha, random_state=C.SEED).fit(self._sc.transform(A), to_target(y_rate, X["distance"], self.mode))
        return self

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self._m.predict(self._sc.transform(self._design(X)))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return from_target(self.predict_raw(X), X["distance"], self.mode)


@dataclass
class LGBModel:
    features: list[str]
    params: dict = field(default_factory=dict)
    mode: str = "log_rate"
    name: str = "lightgbm"
    _m: lgb.LGBMRegressor = field(default=None, init=False, repr=False)

    def fit(self, X, y_rate):
        self._m = lgb.LGBMRegressor(**{**LGB_DEFAULT, **self.params}).fit(X[self.features], to_target(y_rate, X["distance"], self.mode))
        return self

    def predict_raw(self, X):
        return self._m.predict(X[self.features])

    def predict(self, X):
        return from_target(self.predict_raw(X), X["distance"], self.mode)


@dataclass
class CatModel:
    features: list[str]
    params: dict = field(default_factory=dict)
    mode: str = "log_rate"
    name: str = "catboost"
    _m: CatBoostRegressor = field(default=None, init=False, repr=False)

    def fit(self, X, y_rate):
        self._m = CatBoostRegressor(**{**CAT_DEFAULT, **self.params}).fit(X[self.features], to_target(y_rate, X["distance"], self.mode))
        return self

    def predict_raw(self, X):
        return self._m.predict(X[self.features])

    def predict(self, X):
        return from_target(self.predict_raw(X), X["distance"], self.mode)


@dataclass
class HybridModel:
    """Linear model on distance + equipment (+market_index, which extrapolates linearly), trees on its residuals.

    The linear part carries the global trend/scale (trees cannot extrapolate beyond training range); the trees only
    model the bounded residual structure (weight, lane effects, interactions).
    """
    features: list[str]
    ridge_alpha: float = 1.0
    params: dict = field(default_factory=dict)
    mode: str = "log_rate"
    linear_use_market: bool = True
    use_trend: bool = False
    name: str = "hybrid"
    _lin: RidgeModel = field(default=None, init=False, repr=False)
    _tree: lgb.LGBMRegressor = field(default=None, init=False, repr=False)

    def fit(self, X, y_rate):
        self._lin = RidgeModel(features=[], alpha=self.ridge_alpha, mode=self.mode, use_market=self.linear_use_market, use_weight=False, use_trend=self.use_trend).fit(X, y_rate)
        resid = to_target(y_rate, X["distance"], self.mode) - self._lin.predict_raw(X)
        self._tree = lgb.LGBMRegressor(**{**LGB_DEFAULT, **self.params}).fit(X[self.features], resid)
        return self

    def predict_raw(self, X):
        return self._lin.predict_raw(X) + self._tree.predict(X[self.features])

    def predict(self, X):
        return from_target(self.predict_raw(X), X["distance"], self.mode)


@dataclass
class RoutedModel:
    """Use `primary` (has lane-history features) where the lane has history, else `fallback` (no lane features).

    Needed because validation has 8 unseen cities (~12% of rows) whose lanes have no history; in the cold-start CV the
    lane-feature model was worse there (3.13% vs 2.75% MAPE) while better on seen lanes (2.28% vs 2.41%).
    """
    primary: object
    fallback: object
    name: str = "routed"

    def fit(self, X, y_rate):
        self.primary.fit(X, y_rate)
        self.fallback.fit(X, y_rate)
        return self

    def predict(self, X):
        p, f = self.primary.predict(X), self.fallback.predict(X)
        has_hist = ((X["lane_n"] + X["rev_n"]) > 0).to_numpy()
        return np.where(has_hist, p, f)


def nnls_blend_weights(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative weights minimising relative squared error, normalised to sum to 1. preds: (n, k)."""
    from scipy.optimize import nnls
    w, _ = nnls(preds / y[:, None], np.ones(len(y)))
    return w / w.sum()
