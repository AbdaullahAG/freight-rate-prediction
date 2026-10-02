"""Diagnostic follow-up to Stage 5: does the hybrid's residual-tree benefit from market_index, or does it fight the linear part?
CV folds (Jan-Aug) only; Sep-Oct holdout not touched here."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from freight import config as C
from freight import cv
from freight import features as F
from freight.data import load_train
from freight.evaluate import fold_metrics, make_fold_data, pooled, run_splits
from freight.models import HybridModel, RoutedModel
from freight.predict import hybrid_params_split
from freight.quality import flag_label_outliers

train = load_train().reset_index(drop=True)
train = train.join(flag_label_outliers(train))
folds = cv.expanding_monthly_folds(train)
fd = make_fold_data(train, folds)
alpha, tp = hybrid_params_split(json.loads((C.OUTPUTS_DIR / "best_params.json").read_text())["hybrid"]["params"])
BASE = F.BASE + F.GEO
variants = {"trees see market_index (current)": (BASE + ["market_index", "mi_missing"] + F.LANE_TARGET, BASE + ["market_index", "mi_missing"]),
            "trees blind to market_index": (BASE + F.LANE_TARGET, BASE)}
rows = []
for k, (fp, ff) in variants.items():
    p = run_splits(lambda fp=fp, ff=ff: RoutedModel(HybridModel(fp, alpha, tp), HybridModel(ff, alpha, tp)), fd)
    fm, pc = fold_metrics(p, fd), pooled(p, fd, list(fd))
    # market sensitivity: mean signed error by eval-month market_index tercile
    rows.append({"variant": k, "cv_mape_mean": fm.mape_pct.mean(), "cv_mape_std": fm.mape_pct.std(), "worst": fm.mape_pct.max(), "last_fold": fm.mape_pct.iloc[-1],
                 "mae": pc["mae"], "rmse": pc["rmse"], **{f"m{n[-2:]}": v for n, v in zip(fd, fm.mape_pct)}})
out = pd.DataFrame(rows).round(4)
out.to_csv(C.OUTPUTS_DIR / "hybrid_tree_feature_check.csv", index=False)
print(out.T.to_string())
