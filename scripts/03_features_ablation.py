"""Stage 4: target choice + feature ablations. CV folds (eval Mar-Aug) ONLY; the Sep-Oct holdout is not touched."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from freight import config as C
from freight import cv
from freight import features as F
from freight.data import load_train, load_validation
from freight.evaluate import fold_metrics, make_fold_data, pooled, run_splits
from freight.models import LGBModel, RidgeModel
from freight.quality import flag_label_outliers

t0 = time.time()
train = load_train().reset_index(drop=True)
train = train.join(flag_label_outliers(train))
folds = cv.expanding_monthly_folds(train)
fd = make_fold_data(train, folds)
names = [f.name for f in folds]
print(f"fold data built in {time.time()-t0:.0f}s; min clean eval target {min(d['y_eval'][d['clean_eval']].min() for d in fd.values()):.1f}")

LGB_FAST = dict(n_estimators=300, learning_rate=0.06)


def summarize(label: str, preds, extra: dict | None = None) -> dict:
    fm = fold_metrics(preds, fd)
    pc, pa = pooled(preds, fd, names), pooled(preds, fd, names, "all")
    return {"variant": label, "mae": pc["mae"], "rmse": pc["rmse"], "mape_pct": pc["mape_pct"], "smape_pct": pc["smape_pct"],
            "fold_mape_mean": fm.mape_pct.mean(), "fold_mape_std": fm.mape_pct.std(), "fold_mape_worst": fm.mape_pct.max(),
            "last_fold_mape": fm.mape_pct.iloc[-1], "mae_allrows": pa["mae"], **(extra or {}),
            **{f"mape_{n}": m for n, m in zip(names, fm.mape_pct)}}


# ---------- 1) target: log_rate vs rpm vs log_rpm ----------
G3 = F.FEATURE_SETS["G3_+both_signals"]
rows = []
for mode in ["log_rate", "rpm", "log_rpm"]:
    for mname, fac in [("ridge", lambda m=mode: RidgeModel(features=["dow", "quote_signal", "weight_missing", "weight_capped"], alpha=1.0, mode=m)),
                       ("lightgbm", lambda m=mode: LGBModel(G3, LGB_FAST, mode=m))]:
        rows.append(summarize(f"{mname} | target={mode}", run_splits(fac, fd)))
        print(rows[-1]["variant"], round(rows[-1]["mape_pct"], 4), round(rows[-1]["mae"], 2), f"{time.time()-t0:.0f}s", flush=True)
tgt = pd.DataFrame(rows).round(4)
tgt.to_csv(C.OUTPUTS_DIR / "target_comparison.csv", index=False)

# choose target by CV: LightGBM mean fold MAPE (the model class that depends most on target form), tie -> ridge
lg = tgt[tgt.variant.str.startswith("lightgbm")].set_index("variant")
best_mode = lg.fold_mape_mean.idxmin().split("=")[1]
print("chosen target:", best_mode)

# ---------- 2) ablation with LightGBM (+ ridge for reference) ----------
def lgb_fac(feats): return lambda: LGBModel(feats, LGB_FAST, mode=best_mode)


abl, store = [], {}
sets = dict(F.FEATURE_SETS)
for k, feats in sets.items():
    store[k] = run_splits(lgb_fac(feats), fd); abl.append(summarize(k, store[k])); print(k, round(abl[-1]["mape_pct"], 4), flush=True)
# best signal choice so far (by mean fold MAPE) becomes the base of the incremental additions
best_sig = min(abl, key=lambda r: r["fold_mape_mean"])["variant"]
base = sets[best_sig]
increments = {
    "+lane_target_stats": LANE_T if (LANE_T := F.LANE_TARGET) else [],
    "+lane_distance_stats": F.LANE_DIST,
    "+geo_coords": F.GEO,
    "+month,doy (calendar)": F.CALENDAR,
}
for k, add in increments.items():
    label = f"{best_sig} {k}"
    store[label] = run_splits(lgb_fac(base + add), fd); abl.append(summarize(label, store[label])); print(label, round(abl[-1]["mape_pct"], 4), flush=True)
# without day-of-week
label = f"{best_sig} -dow"
store[label] = run_splits(lgb_fac([f for f in base if f != "dow"]), fd); abl.append(summarize(label, store[label]))
# market_index with/without for ridge as well
for lab, feats in [("ridge: market_index only", ["dow", "weight_missing", "weight_capped"]), ("ridge: +quote_signal", ["dow", "quote_signal", "weight_missing", "weight_capped"])]:
    abl.append(summarize(lab, run_splits(lambda f=feats: RidgeModel(features=f, alpha=1.0, mode=best_mode), fd)))
abl.append(summarize("ridge: NO market_index", run_splits(lambda: RidgeModel(features=["dow", "weight_missing", "weight_capped"], alpha=1.0, mode=best_mode, use_market=False), fd)))
ab = pd.DataFrame(abl).round(4)
ab.to_csv(C.OUTPUTS_DIR / "feature_ablation.csv", index=False)
print(ab[["variant", "mape_pct", "mae", "fold_mape_mean", "fold_mape_std", "fold_mape_worst", "last_fold_mape"]].to_string(index=False))

# ---------- 3) signal stability / range checks ----------
v = load_validation()
g1, g3, g0, g2 = (pd.DataFrame(x) for x in ([r for r in abl if r["variant"] == k] for k in ["G1_+market_index", "G3_+both_signals", "G0_base_no_market", "G2_+quote_signal_only"]))
stab = pd.DataFrame({"fold": names,
                     "mape_G0_none": [g0[f"mape_{n}"].iloc[0] for n in names], "mape_G1_mi": [g1[f"mape_{n}"].iloc[0] for n in names],
                     "mape_G2_qs": [g2[f"mape_{n}"].iloc[0] for n in names], "mape_G3_both": [g3[f"mape_{n}"].iloc[0] for n in names]})
stab["mi_gain_pp"] = stab.mape_G0_none - stab.mape_G1_mi
stab["qs_gain_over_G1_pp"] = stab.mape_G1_mi - stab.mape_G3_both
stab.round(4).to_csv(C.OUTPUTS_DIR / "signal_stability_by_fold.csv", index=False)
print(stab.round(3).to_string(index=False))
tr_ = train
ranges = {c: {"train_min": float(tr_[c].min()), "train_max": float(tr_[c].max()), "val_min": float(v[c].min()), "val_max": float(v[c].max()),
              "val_pct_outside": float(100 * ((v[c] < tr_[c].min()) | (v[c] > tr_[c].max())).mean())} for c in ["market_index", "quote_signal"]}
json.dump({"best_target_mode": best_mode, "signal_set_base": best_sig, "signal_ranges": ranges, "seconds": time.time() - t0}, open(C.OUTPUTS_DIR / "feature_decisions_raw.json", "w"), indent=1)
print("total seconds", round(time.time() - t0))

# ---------- 4) combinations, incl. a cold-start variant INSIDE the Jan-Aug folds (Sep-Oct stays untouched) ----------
_, held = cv.cold_start_holdout(train)
touch = (train.pickup.isin(held) | train.delivery.isin(held)).to_numpy()
cold_splits = [cv.Split(f.name, f.train_idx[~touch[f.train_idx]], f.eval_idx[touch[f.eval_idx]]) for f in folds]
for s in cold_splits:
    cv.assert_no_time_leak(train, s)
fd_cold = make_fold_data(train, cold_splits)
C1 = F.BASE + ["market_index", "mi_missing"]
combos = {
    "C1 base+mi (no dow)": C1,
    "C2 C1+lane_target": C1 + F.LANE_TARGET,
    "C3 C1+geo": C1 + F.GEO,
    "C4 C1+lane_target+geo": C1 + F.LANE_TARGET + F.GEO,
    "C5 C4+dow": C1 + F.LANE_TARGET + F.GEO + ["dow"],
    "C6 C4+quote_signal": C1 + F.LANE_TARGET + F.GEO + ["quote_signal"],
}
rows = []
for k, feats in combos.items():
    for tag, data in (("time-CV", fd), ("cold-start CV", fd_cold)):
        nm = [n for n in data]
        p = run_splits(lambda f=feats: LGBModel(f, LGB_FAST, mode=best_mode), data)
        fm = fold_metrics(p, data); pc = pooled(p, data, nm)
        rows.append({"variant": k, "eval": tag, "mape_pct": pc["mape_pct"], "mae": pc["mae"], "rmse": pc["rmse"], "fold_mape_mean": fm.mape_pct.mean(),
                     "fold_mape_std": fm.mape_pct.std(), "worst_fold": fm.mape_pct.max(), "last_fold": fm.mape_pct.iloc[-1], "n_eval": int(sum(d["clean_eval"].sum() for d in data.values()))})
        print(rows[-1]["variant"], tag, round(rows[-1]["mape_pct"], 4), flush=True)
comb = pd.DataFrame(rows).round(4)
comb.to_csv(C.OUTPUTS_DIR / "feature_combinations.csv", index=False)
print(comb.to_string(index=False))
print("cold-start held-out cities (inside CV):", held)
