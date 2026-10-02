"""Stages 6-7: refit chosen model on ALL clean train rows (Jan-Oct), December scenario (A/B), 12,000 validation predictions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freight import config as C
from freight.data import load_december, load_template, load_train, load_validation
from freight.december import december_frame, december_signals
from freight.features import build_xy
from freight.predict import make_final_model, validate_predictions
from freight.quality import flag_label_outliers

sel = json.loads((C.OUTPUTS_DIR / "model_selection.json").read_text())
params = json.loads((C.OUTPUTS_DIR / "best_params.json").read_text())
assert sel["chosen"] == "hybrid", f"final pipeline implements the chosen model 'hybrid', got {sel['chosen']}"

train = load_train().reset_index(drop=True)
train = train.join(flag_label_outliers(train))
val, tmpl, dec_in = load_validation(), load_template(), load_december()
clean = ~train["is_label_outlier"]

# validation null market_index: constant fallback = mean market_index of the last 7 train days (no use of validation peers beyond the two permitted purposes)
mi_fallback = float(train.loc[train.date > train.date.max() - pd.Timedelta(days=7), "market_index"].mean())
d_val = build_xy(train, val, clean, ap_mi_fallback=mi_fallback)
model = make_final_model(sel, params).fit(d_val["Xtr"], d_val["ytr"])
joblib.dump(model, C.OUTPUTS_DIR / "final_model_hybrid.joblib")
print("train rows used:", len(d_val["Xtr"]), "of", len(train), "| mi_fallback", round(mi_fallback, 4), "| val null mi rows:", int(val.market_index.isna().sum()))

# ---------- Stage 7: validation predictions ----------
pred = pd.DataFrame({"load_id": val["load_id"].to_numpy(), "predicted_rate": model.predict(d_val["Xap"])})
out = validate_predictions(pred, tmpl)
out["predicted_rate"] = out["predicted_rate"].round(4)
out.to_csv(C.PREDICTIONS_PATH, index=False)
print("validation_predictions.csv:", out.shape, "min/mean/max", out.predicted_rate.min().round(1), out.predicted_rate.mean().round(1), out.predicted_rate.max().round(1))
vv = val.assign(pred=out.set_index("load_id").loc[val.load_id, "predicted_rate"].to_numpy())
vv["rpm"] = vv.pred / vv.distance
print("pred rate-per-mile by month:", vv.groupby(vv.date.dt.month).rpm.median().round(3).to_dict(), "| train clean median rpm:", round(float((train[clean].posted_rate / train[clean].distance).median()), 3))

# ---------- Stage 6: December ----------
sigs = december_signals(val, dec_in["date"])
res = {}
for k, s in sigs.items():
    fr = december_frame(dec_in, train, s)
    d = build_xy(train, fr.assign(**{}), clean)  # train slice identical; apply slice = December rows
    res[k] = model.predict(d["Xap"])
    if k.startswith("A"):
        hav, ratio = float(d["Xap"].haversine.iloc[0]), float(d["Xap"].dist_over_hav.iloc[0])
        lane_n = (d["Xap"].lane_n.iloc[0], d["Xap"].rev_n.iloc[0])
A, B = res["A_dec_validation_daily_mean"], res["B_frozen_last_nov_day"]
cmp = pd.DataFrame({"date": dec_in["date"].dt.date, "market_index_A": sigs["A_dec_validation_daily_mean"].market_index.to_numpy(), "market_index_B": sigs["B_frozen_last_nov_day"].market_index.to_numpy(),
                    "quote_signal_A": sigs["A_dec_validation_daily_mean"].quote_signal.to_numpy(), "pred_A": A, "pred_B": B, "dow": dec_in["date"].dt.day_name().to_numpy()})
cmp["abs_diff"] = (cmp.pred_A - cmp.pred_B).abs()
cmp.round(4).to_csv(C.OUTPUTS_DIR / "december_options_comparison.csv", index=False)
dec_stats = {"mean_abs_diff_A_vs_B": float(cmp.abs_diff.mean()), "max_abs_diff_A_vs_B": float(cmp.abs_diff.max()), "mean_pred_A": float(A.mean()), "mean_pred_B": float(B.mean()),
             "A_min": float(A.min()), "A_max": float(A.max()), "A_range_pct_of_mean": float(100 * (A.max() - A.min()) / A.mean()),
             "A_max_day_to_day_jump": float(np.abs(np.diff(A)).max()), "A_mean_by_weekday": cmp.groupby("dow").pred_A.mean().round(2).to_dict(),
             "haversine_lex_fw": hav, "dist_over_hav": ratio, "lane_history_rows(fwd,rev)": [float(lane_n[0]), float(lane_n[1])],
             "market_index_A_range": [float(cmp.market_index_A.min()), float(cmp.market_index_A.max())],
             "market_index_B_value": float(cmp.market_index_B.iloc[0]), "choice": "A"}
(C.OUTPUTS_DIR / "december_summary.json").write_text(json.dumps(dec_stats, indent=1, default=float))
print(json.dumps(dec_stats, indent=1, default=float))
dec_out = dec_in.copy()
dec_out["predicted_rate"] = np.round(A, 4)
assert list(dec_out.columns) == ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
dec_out["date"] = dec_out["date"].dt.strftime("%Y-%m-%d")
dec_out.to_csv(C.DECEMBER_PATH, index=False)

fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(cmp.date, cmp.pred_A, "o-", color="#064A56", label="A: December validation daily mean (chosen)"); ax.plot(cmp.date, cmp.pred_B, "s--", color="#E07A1F", ms=3, label="B: frozen at 2025-11-30")
ax.set_title("December predictions: signal-fill options", loc="left"); ax.legend(fontsize=8); ax.tick_params(axis="x", rotation=40); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "11_december_options.png", dpi=130)
