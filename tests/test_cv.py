"""Time-ordering guarantees for every validation split."""
import numpy as np
import pandas as pd
import pytest

from freight import config as C
from freight import cv
from freight.data import load_train


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    if not C.TRAIN_PATH.is_file():
        pytest.skip("data/train_test.csv not present")
    return load_train().reset_index(drop=True)


def synthetic() -> pd.DataFrame:
    d = pd.date_range("2025-01-01", "2025-10-31", freq="D")
    return pd.DataFrame({"date": np.repeat(d, 3), "pickup": "A", "delivery": "B"})


@pytest.mark.parametrize("make", [synthetic, lambda: None])
def test_every_fold_train_precedes_eval(make, train):
    df = make() if make() is not None else train
    folds = cv.expanding_monthly_folds(df)
    assert len(folds) == C.CV_LAST_EVAL_MONTH - C.CV_FIRST_EVAL_MONTH + 1
    for f in folds:
        assert df["date"].iloc[f.train_idx].max() < df["date"].iloc[f.eval_idx].min()
        cv.assert_no_time_leak(df, f)


def test_cv_only_uses_jan_to_aug(train):
    for f in cv.expanding_monthly_folds(train):
        idx = np.concatenate([f.train_idx, f.eval_idx])
        assert train["date"].iloc[idx].dt.month.max() <= C.CV_LAST_EVAL_MONTH


def test_folds_are_expanding_and_eval_is_one_month(train):
    folds = cv.expanding_monthly_folds(train)
    sizes = [len(f.train_idx) for f in folds]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    for f in folds:
        assert train["date"].iloc[f.eval_idx].dt.month.nunique() == 1


def test_holdout_is_sep_oct_and_after_cv_window(train):
    h = cv.final_holdout(train)
    cv.assert_no_time_leak(train, h)
    assert set(train["date"].iloc[h.eval_idx].dt.month) == set(C.HOLDOUT_MONTHS)
    assert train["date"].iloc[h.train_idx].dt.month.max() == C.CV_LAST_EVAL_MONTH


def test_cold_start_holdout_is_time_and_city_disjoint(train):
    s, held = cv.cold_start_holdout(train)
    cv.assert_no_time_leak(train, s)
    tr = train.iloc[s.train_idx]
    assert not (set(tr["pickup"]) | set(tr["delivery"])) & set(held)


def test_leak_check_actually_fires():
    df = synthetic()
    bad = cv.Split("bad", np.array([len(df) - 1]), np.array([0]))
    with pytest.raises(AssertionError):
        cv.assert_no_time_leak(df, bad)
