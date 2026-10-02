"""Raw-data shapes, leakage checks on the feature builder, and model sanity."""
import numpy as np
import pandas as pd
import pytest

from freight import config as C
from freight.data import load_train, load_validation
from freight.features import build_xy

needs_data = pytest.mark.skipif(not C.TRAIN_PATH.is_file(), reason="raw data not present")


@needs_data
def test_raw_shapes_and_columns():
    t, v = load_train(), load_validation()
    assert t.shape == (48_000, 14) and v.shape == (12_000, 13)
    assert C.TARGET in t.columns and C.TARGET not in v.columns
    assert t.date.max() < v.date.min()          # validation is strictly in the future


@needs_data
def test_lane_target_stats_only_use_strictly_earlier_months():
    """Changing the labels of month m must not change lane features of any training row in months <= m."""
    t = load_train()
    t = t[t.date.dt.month <= 5].reset_index(drop=True)
    clean = pd.Series(True, index=t.index)
    a = build_xy(t, t.iloc[:10], clean)["Xtr"]
    t2 = t.copy()
    # perturb only some month-3 rows (a uniform scaling would be removed by the per-month demeaning, so it would prove nothing)
    pert = (t2.date.dt.month == 3) & (np.arange(len(t2)) % 2 == 0)
    t2.loc[pert, C.TARGET] *= 5.0
    b = build_xy(t2, t2.iloc[:10], clean)["Xtr"]
    early = a["month"] <= 3
    cols = ["lane_stat", "lane_n", "rev_stat", "rev_n"]
    pd.testing.assert_frame_equal(a.loc[early, cols], b.loc[early, cols])
    assert not np.allclose(a.loc[~early, "lane_stat"], b.loc[~early, "lane_stat"])  # later months DO see it
    assert (a.loc[a["month"] == 1, "lane_n"] == 0).all()                               # nothing precedes January


@needs_data
def test_eval_rows_see_only_training_slice():
    t = load_train()
    tr, ev = t[t.date.dt.month <= 4].reset_index(drop=True), t[t.date.dt.month == 5].reset_index(drop=True)
    d = build_xy(tr, ev, pd.Series(True, index=tr.index))
    assert d["Xap"]["lane_n"].max() <= len(tr)
    ev2 = ev.copy(); ev2[C.TARGET] = 1.0                      # eval labels must never be read
    pd.testing.assert_frame_equal(d["Xap"], build_xy(tr, ev2, pd.Series(True, index=tr.index))["Xap"])


MODEL = C.OUTPUTS_DIR / "final_model_hybrid.joblib"


@pytest.mark.skipif(not (MODEL.is_file() and C.TRAIN_PATH.is_file()), reason="final model not trained yet")
@pytest.mark.parametrize("lane_n", [0, 5])
@pytest.mark.parametrize("equip", [0, 1, 2])
def test_hybrid_monotonic_in_distance_on_toy_grid(equip, lane_n):
    import joblib
    t, v = load_train().reset_index(drop=True), load_validation()
    base = build_xy(t, v.iloc[:3], pd.Series(True, index=t.index), ap_mi_fallback=0.94)["Xap"].iloc[[0]]
    grid = np.arange(100.0, 3001.0, 50.0)
    X = pd.concat([base] * len(grid), ignore_index=True)
    X = X.assign(distance=grid, log_distance=np.log(grid), haversine=grid / 1.18, dist_over_hav=1.18, mismatch=0, equip=equip, weight=32_000.0,
                 weight_missing=0, weight_capped=0, weight_bin=3.0, market_index=1.0, mi_missing=0, lane_n=lane_n, rev_n=lane_n, lane_stat=0.0, rev_stat=0.0)
    p = joblib.load(MODEL).predict(X)
    assert np.isfinite(p).all() and (p > 0).all()
    assert (np.diff(p) > 0).all(), "rate must increase with distance"
