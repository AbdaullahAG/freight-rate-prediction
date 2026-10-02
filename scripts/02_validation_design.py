"""Stage 3: build + verify the time-based validation design; quantify cleaning effects on CV folds ONLY.

The Sep-Oct holdout is NOT scored here (it is reserved for one evaluation of the tuned candidates in Part B).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freight import config as C
from freight import cv
from freight.data import load_train, load_validation
from freight.quality import flag_label_outliers

t = load_train().reset_index(drop=True)
t = t.join(flag_label_outliers(t))
v = load_validation()

# ---------- folds ----------
folds = cv.expanding_monthly_folds(t)
hold = cv.final_holdout(t)
cold, held_cities = cv.cold_start_holdout(t)
for s in [*folds, hold, cold]:
    cv.assert_no_time_leak(t, s)
rows = []
for s in [*folds, hold, cold]:
    tr, ev = t.iloc[s.train_idx], t.iloc[s.eval_idx]
    rows.append({"split": s.name, "train_rows": len(tr), "train_from": tr.date.min().date(), "train_to": tr.date.max().date(),
                 "eval_rows": len(ev), "eval_from": ev.date.min().date(), "eval_to": ev.date.max().date(),
                 "eval_label_outliers": int(ev.is_label_outlier.sum()), "gap_days": (ev.date.min() - tr.date.max()).days})
fold_tab = pd.DataFrame(rows)
fold_tab.to_csv(C.OUTPUTS_DIR / "cv_folds.csv", index=False)
print(fold_tab.to_string(index=False))
print("cold-start held-out cities:", held_cities)
print("final refit: train", t.date.min().date(), "->", t.date.max().date(), "| validation.csv", v.date.min().date(), "->", v.date.max().date(),
      "| gap days:", (v.date.min() - t.date.max()).days)


# ---------- baseline used ONLY to quantify cleaning effects (not the final model) ----------
def design(df: pd.DataFrame, weight_mode: str, use_mi: bool) -> np.ndarray:
    mi = df.market_index.fillna(df.groupby("date").market_index.transform("median")).fillna(df.market_index.median())
    ld = np.log(df.distance)
    X = pd.get_dummies(df.equipment, dtype=float).reindex(columns=list(C.EQUIPMENT), fill_value=0.0)
    X = X.assign(ld=ld, ld2=ld**2)
    if use_mi:
        X = X.assign(mi=mi)
    if weight_mode != "none":
        w = df.weight if weight_mode == "raw" else df.weight.abs()
        X = X.assign(w=(w.fillna(w.median()) / 1e4))
    return X.values


def run(variant: str, train_clean: bool, eval_clean: bool, weight_mode: str, use_mi: bool = True) -> dict:
    preds, ys, ws = [], [], []
    for s in folds:
        tr, ev = t.iloc[s.train_idx], t.iloc[s.eval_idx]
        if train_clean:
            tr = tr[~tr.is_label_outlier]
        if eval_clean:
            ev = ev[~ev.is_label_outlier]
        beta = np.linalg.lstsq(design(tr, weight_mode, use_mi), np.log(tr.posted_rate).values, rcond=None)[0]
        preds.append(np.exp(design(ev, weight_mode, use_mi) @ beta)); ys.append(ev.posted_rate.values)
    m = cv.regression_metrics(np.concatenate(ys), np.concatenate(preds))
    return {"variant": variant, **m}


eff = pd.DataFrame([
    run("A) train on ALL rows, eval ALL rows (naive)", False, False, "abs"),
    run("B) train CLEAN, eval ALL rows", True, False, "abs"),
    run("C) train CLEAN, eval CLEAN", True, True, "abs"),
    run("C-noMI) as C but without market_index", True, True, "abs", use_mi=False),
    run("D) as C, weight RAW (negatives kept)", True, True, "raw"),
    run("E) as C, weight ABS", True, True, "abs"),
    run("F) as C, no weight feature", True, True, "none"),
]).round(4)
eff.to_csv(C.OUTPUTS_DIR / "baseline_cleaning_effects.csv", index=False)
print(eff.to_string(index=False))

# ---------- write measured effects back into the Stage-2 issue table ----------
def _row(name: str) -> pd.Series:
    return eff.set_index("variant").loc[[i for i in eff.variant if i.startswith(name)][0]]


A, B, Cc, D, Ew, F, NM = (_row(x) for x in ["A)", "B)", "C)", "D)", "E)", "F)", "C-noMI"])
measured = {
    "Corrupted posted_rate labels": (
        f"CV folds Mar-Aug, log-linear baseline: removing them from TRAINING changes MAE {A.mae:.1f}->{B.mae:.1f}, RMSE {A.rmse:.1f}->{B.rmse:.1f} (small, the log fit is robust); "
        f"but they dominate any error metric: RMSE {B.rmse:.0f} on all eval rows vs {Cc.rmse:.0f} on clean rows (MAE {B.mae:.0f} vs {Cc.mae:.0f}). "
        "Irreducible if validation labels contain the same corruption; matters a lot for squared-error GBMs trained on raw rate."),
    "Negative weights": (
        f"baseline MAE {D.mae:.1f}->{Ew.mae:.1f} and RMSE {D.rmse:.1f}->{Ew.rmse:.1f} when weight is abs() instead of raw (clean eval); "
        f"dropping weight entirely gives MAE {F.mae:.1f}"),
    "Null market_index": (
        f"market_index is the strongest feature: baseline MAE {NM.mae:.1f}->{Cc.mae:.1f} with it; date-median fill keeps it available for all rows"),
}
tab = pd.read_csv(C.OUTPUTS_DIR / "data_quality_issues.csv")
for key, txt in measured.items():
    tab.loc[tab.issue.str.startswith(key), "effect"] = txt
tab.to_csv(C.OUTPUTS_DIR / "data_quality_issues.csv", index=False)
md = ["# Data-quality issue table (generated by scripts/01_eda.py, effects updated by scripts/02_validation_design.py)\n",
      "| # | issue | how detected | rows affected | fix | effect |", "|---|---|---|---|---|---|"]
for i, r in enumerate(tab.to_dict("records"), 1):
    md.append("| " + " | ".join([str(i)] + [str(r[k]).replace("|", "/") for k in ["issue", "how_detected", "rows_affected", "fix", "effect"]]) + " |")
(C.OUTPUTS_DIR / "data_quality_issues.md").write_text("\n".join(md) + "\n")

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(10, 3.6))
for i, s in enumerate([*folds, hold, cold]):
    tr, ev = t.iloc[s.train_idx], t.iloc[s.eval_idx]
    ax.barh(i, (tr.date.max() - tr.date.min()).days + 1, left=tr.date.min().toordinal(), color="#064A56", height=.6)
    ax.barh(i, (ev.date.max() - ev.date.min()).days + 1, left=ev.date.min().toordinal(), color="#E07A1F", height=.6)
ax.barh(len(folds) + 2, (v.date.max() - v.date.min()).days + 1, left=v.date.min().toordinal(), color="#9DAFB3", height=.6)
ax.set_yticks(list(range(len(folds) + 2)) + [len(folds) + 2]); ax.set_yticklabels([s.name for s in [*folds, hold, cold]] + ["validation.csv (no labels)"], fontsize=8)
ticks = [pd.Timestamp(f"2025-{m:02d}-01") for m in range(1, 13)]
ax.set_xticks([x.toordinal() for x in ticks]); ax.set_xticklabels([x.strftime("%b") for x in ticks]); ax.invert_yaxis()
ax.set_title("Time-based validation design (teal = train, orange = eval, grey = future to predict)", loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "07_validation_design.png", dpi=130); plt.close(fig)

(C.OUTPUTS_DIR / "validation_design.json").write_text(json.dumps({
    "cv_eval_months": [C.CV_FIRST_EVAL_MONTH, C.CV_LAST_EVAL_MONTH], "holdout_months": list(C.HOLDOUT_MONTHS),
    "cold_start_held_out_cities": held_cities, "n_folds": len(folds),
    "time_leak_assertions_passed": True, "holdout_scored_in_part_a": False}, indent=1))
