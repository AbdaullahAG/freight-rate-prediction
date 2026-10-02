"""Format checks on the final deliverables (skipped when the files are absent)."""
import numpy as np
import pandas as pd
import pytest

from freight import config as C


def _december_filled() -> bool:
    """The raw assessment file has an empty predicted_rate; these tests only apply once the pipeline has filled it."""
    return C.DECEMBER_PATH.is_file() and pd.read_csv(C.DECEMBER_PATH)["predicted_rate"].notna().all()


@pytest.mark.skipif(not (C.PREDICTIONS_PATH.is_file() and C.TEMPLATE_PATH.is_file()), reason="predictions/template missing")
def test_validation_predictions_match_template():
    p, t = pd.read_csv(C.PREDICTIONS_PATH), pd.read_csv(C.TEMPLATE_PATH)
    assert list(p.columns) == ["load_id", "predicted_rate"]
    assert p["load_id"].tolist() == t["load_id"].tolist()          # same ids, same order
    assert np.isfinite(p["predicted_rate"]).all() and (p["predicted_rate"] > 0).all()


@pytest.mark.skipif(not _december_filled(), reason="december predictions not filled yet")
def test_december_columns_and_fixed_inputs():
    d = pd.read_csv(C.DECEMBER_PATH)
    assert list(d.columns) == ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    assert len(d) == 31 and (d.pickup == "Lexington").all() and (d.distance == 360).all() and d.predicted_rate.gt(0).all()


@pytest.mark.skipif(not (C.PREDICTIONS_PATH.is_file() and C.VALIDATION_PATH.is_file()), reason="data missing")
def test_prediction_shapes_and_no_bad_values():
    p, v = pd.read_csv(C.PREDICTIONS_PATH), pd.read_csv(C.VALIDATION_PATH)
    assert p.shape == (12_000, 2) and len(v) == 12_000
    assert p["load_id"].is_unique and set(p["load_id"]) == set(v["load_id"])
    assert not p["predicted_rate"].isna().any() and np.isfinite(p["predicted_rate"]).all() and (p["predicted_rate"] > 0).all()


@pytest.mark.skipif(not _december_filled(), reason="december predictions not filled yet")
def test_december_values_and_order_match_score_py_constants():
    d = pd.read_csv(C.DECEMBER_PATH)
    assert (d.delivery == "Fort Wayne").all() and (d.equipment == "Dry Van").all() and (d.weight == 32_000).all()
    assert pd.to_datetime(d.date).tolist() == list(pd.date_range("2025-12-01", "2025-12-31"))
    assert np.isfinite(d.predicted_rate).all()
