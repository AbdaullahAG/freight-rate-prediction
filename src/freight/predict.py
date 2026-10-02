"""Final refit + prediction helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .features import build_xy
from .models import HybridModel, LGBModel, RoutedModel


def hybrid_params_split(p: dict) -> tuple[float, dict]:
    p = dict(p)
    alpha = p.pop("ridge_alpha")
    p["num_leaves"], p["n_estimators"] = p.pop("num_leaves_h"), p.pop("n_estimators_h")
    return alpha, p


def make_final_model(selection: dict, params: dict) -> RoutedModel:
    alpha, tp = hybrid_params_split(params["hybrid"]["params"])
    prim = HybridModel(selection["features_primary"], ridge_alpha=alpha, params=tp)
    fall = HybridModel(selection["features_fallback"], ridge_alpha=alpha, params=tp)
    return RoutedModel(prim, fall)


def validate_predictions(pred: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    assert list(pred.columns) == ["load_id", "predicted_rate"]
    assert len(pred) == len(template) == 12_000
    assert pred["load_id"].is_unique and set(pred["load_id"]) == set(template["load_id"]), "load_id mismatch with template"
    assert np.isfinite(pred["predicted_rate"]).all() and (pred["predicted_rate"] > 0).all()
    return template[["load_id"]].merge(pred, on="load_id", how="left", validate="one_to_one")  # template order
