"""Run models over the time-based splits; keep predictions so blends / residual analysis reuse them."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import config as C
from . import cv
from .features import build_xy


def make_fold_data(train: pd.DataFrame, splits: list[cv.Split]) -> dict[str, dict]:
    """Pre-compute features per split once (they do not depend on the feature subset)."""
    out = {}
    for s in splits:
        tr, ev = train.iloc[s.train_idx], train.iloc[s.eval_idx]
        d = build_xy(tr, ev, ~tr["is_label_outlier"])
        d["y_eval"] = ev[C.TARGET].to_numpy()
        d["clean_eval"] = (~ev["is_label_outlier"]).to_numpy()
        d["eval_index"] = ev.index
        out[s.name] = d
    return out


def run_splits(factory: Callable[[], object], fold_data: dict[str, dict]) -> dict[str, np.ndarray]:
    preds = {}
    for name, d in fold_data.items():
        m = factory().fit(d["Xtr"], d["ytr"])
        preds[name] = m.predict(d["Xap"])
    return preds


def fold_metrics(preds: dict[str, np.ndarray], fold_data: dict[str, dict], subset: str = "clean") -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        d = fold_data[name]
        msk = d["clean_eval"] if subset == "clean" else np.ones(len(p), bool)
        rows.append({"split": name, **cv.regression_metrics(d["y_eval"][msk], p[msk])})
    return pd.DataFrame(rows)


def pooled(preds: dict[str, np.ndarray], fold_data: dict[str, dict], names: list[str], subset: str = "clean") -> dict[str, float]:
    ys, ps = [], []
    for n in names:
        d = fold_data[n]
        msk = d["clean_eval"] if subset == "clean" else np.ones(len(preds[n]), bool)
        ys.append(d["y_eval"][msk]); ps.append(preds[n][msk])
    return cv.regression_metrics(np.concatenate(ys), np.concatenate(ps))
