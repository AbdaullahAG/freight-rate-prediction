"""Stage 5: tune (Optuna, CV folds Jan-Aug only), compare models per fold + Sep-Oct holdout, pick by temporal stability.

Usage: python scripts/04_models.py [--skip-tune]   (re-uses outputs/best_params.json)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap

from freight import config as C
from freight import cv
from freight import features as F
from freight.data import load_train
from freight.evaluate import fold_metrics, make_fold_data, pooled
from freight.models import CatModel, HybridModel, LGBModel, RidgeModel, RoutedModel, nnls_blend_weights
from freight.quality import flag_label_outliers

optuna.logging.set_verbosity(optuna.logging.WARNING)
ap = argparse.ArgumentParser(); ap.add_argument("--skip-tune", action="store_true"); args = ap.parse_args()
T0 = time.time()
MAX_TRIALS = 20
CAPS = {"ridge": 60, "lightgbm": 480, "catboost": 780, "hybrid": 480}   # seconds per model (safety cap; trials counted in the log)

train = load_train().reset_index(drop=True)
train = train.join(flag_label_outliers(train))
folds = cv.expanding_monthly_folds(train)
hold = cv.final_holdout(train)
cold, held_cities = cv.cold_start_holdout(train)
fd = make_fold_data(train, folds)
fd_h = make_fold_data(train, [hold])
fd_c = make_fold_data(train, [cold])
names = list(fd)

BASE_MI = F.BASE + ["market_index", "mi_missing"]
F_PRIMARY = BASE_MI + F.GEO + F.LANE_TARGET
F_FALLBACK = BASE_MI + F.GEO


def mape_objective(factory) -> float:
    vals = []
    for d in fd.values():
        m = factory().fit(d["Xtr"], d["ytr"])
        p = m.predict(d["Xap"])
        msk = d["clean_eval"]
        vals.append(100 * np.mean(np.abs(p[msk] - d["y_eval"][msk]) / d["y_eval"][msk]))
    return float(np.mean(vals))


def tune(name: str, space, factory_from_params) -> dict:
    t0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=C.SEED))

    def objective(trial):
        params = space(trial)
        if time.time() - t0 > CAPS[name]:
            study.stop()
        return mape_objective(lambda: factory_from_params(params))

    study.optimize(objective, n_trials=MAX_TRIALS, show_progress_bar=False)
    df = study.trials_dataframe()
    df.to_csv(C.OUTPUTS_DIR / f"optuna_{name}.csv", index=False)
    print(f"[{name}] trials={len(study.trials)} best fold-mean MAPE={study.best_value:.4f} in {time.time()-t0:.0f}s params={study.best_params}", flush=True)
    return {"params": study.best_params, "n_trials": len(study.trials), "best_cv_mape": study.best_value, "seconds": time.time() - t0, "hit_time_cap": (time.time() - t0) > CAPS[name]}


def lgb_space(trial):
    return dict(n_estimators=trial.suggest_int("n_estimators", 200, 700), learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
                num_leaves=trial.suggest_int("num_leaves", 8, 64), min_child_samples=trial.suggest_int("min_child_samples", 20, 150),
                subsample=trial.suggest_float("subsample", 0.6, 1.0), colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 20, log=True), reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True))


def hybrid_space(trial):
    p = lgb_space(trial)
    p.update(num_leaves=trial.suggest_int("num_leaves_h", 4, 32), n_estimators=trial.suggest_int("n_estimators_h", 100, 500))
    p["ridge_alpha"] = trial.suggest_float("ridge_alpha", 1e-2, 50, log=True)
    for k in ("num_leaves", "n_estimators"):
        trial.params.pop(k, None)
    return p


def cat_space(trial):
    return dict(iterations=trial.suggest_int("iterations", 200, 500), learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
                depth=trial.suggest_int("depth", 4, 7), l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True))


def ridge_space(trial):
    return dict(alpha=trial.suggest_float("alpha", 1e-3, 100, log=True), use_dow=trial.suggest_categorical("use_dow", [False, True]))


def ridge_feats(p): return ["weight_missing", "weight_capped"] + (["dow"] if p.get("use_dow") else [])


def hybrid_split(p):
    p = dict(p); alpha = p.pop("ridge_alpha")
    if "num_leaves_h" in p: p["num_leaves"] = p.pop("num_leaves_h")
    if "n_estimators_h" in p: p["n_estimators"] = p.pop("n_estimators_h")
    return alpha, p


FACT = {
    "ridge": lambda p, feats=None: RidgeModel(features=ridge_feats(p), alpha=p["alpha"]),
    "lightgbm": lambda p, feats=F_PRIMARY: LGBModel(feats, p),
    "catboost": lambda p, feats=F_PRIMARY: CatModel(feats, p),
    "hybrid": lambda p, feats=F_PRIMARY: HybridModel(feats, ridge_alpha=hybrid_split(p)[0], params=hybrid_split(p)[1]),
}
SPACES = {"ridge": ridge_space, "lightgbm": lgb_space, "catboost": cat_space, "hybrid": hybrid_space}
PARAMS_PATH = C.OUTPUTS_DIR / "best_params.json"

if args.skip_tune and PARAMS_PATH.is_file():
    best = json.loads(PARAMS_PATH.read_text())
else:
    best = {}
    for name in ["ridge", "lightgbm", "hybrid", "catboost"]:
        best[name] = tune(name, SPACES[name], lambda p, n=name: FACT[n](p, F_PRIMARY) if n != "ridge" else FACT[n](p))
    PARAMS_PATH.write_text(json.dumps(best, indent=1))


def routed(name: str):
    p = best[name]["params"]
    if name == "ridge":
        return FACT["ridge"](p)
    return RoutedModel(FACT[name](p, F_PRIMARY), FACT[name](p, F_FALLBACK))


def run(factory, data):
    return {n: factory().fit(d["Xtr"], d["ytr"]).predict(d["Xap"]) for n, d in data.items()}


MODELS = ["ridge", "lightgbm", "catboost", "hybrid"]
oof, hold_p, cold_p = {}, {}, {}
for m in MODELS:
    oof[m] = run(lambda m=m: routed(m), fd)
    print(f"{m}: CV done {time.time()-T0:.0f}s", flush=True)

# ---------- blend weights from CV predictions (Jan-Aug folds only) ----------
Pm = np.column_stack([np.concatenate([oof[m][n][fd[n]["clean_eval"]] for n in names]) for m in MODELS])
yv = np.concatenate([fd[n]["y_eval"][fd[n]["clean_eval"]] for n in names])
w = nnls_blend_weights(Pm, yv)
blend_w = dict(zip(MODELS, map(float, w)))
print("blend weights", blend_w)
oof["blend"] = {n: sum(blend_w[m] * oof[m][n] for m in MODELS) for n in names}

# ---------- holdout (scored once) + cold-start diagnostic ----------
for m in MODELS:
    hold_p[m] = run(lambda m=m: routed(m), fd_h); cold_p[m] = run(lambda m=m: routed(m), fd_c)
hold_p["blend"] = {n: sum(blend_w[m] * hold_p[m][n] for m in MODELS) for n in fd_h}
cold_p["blend"] = {n: sum(blend_w[m] * cold_p[m][n] for m in MODELS) for n in fd_c}
ALL = MODELS + ["blend"]

rows_cv, rows_sum = [], []
for m in ALL:
    for subset in ("clean", "all"):
        fm = fold_metrics(oof[m], fd, subset).assign(model=m, subset=subset)
        rows_cv.append(fm)
    fmc = fold_metrics(oof[m], fd, "clean")
    hc = pooled(hold_p[m], fd_h, list(fd_h), "clean"); ha = pooled(hold_p[m], fd_h, list(fd_h), "all")
    cc = pooled(cold_p[m], fd_c, list(fd_c), "clean")
    pc = pooled(oof[m], fd, names, "clean")
    rows_sum.append({"model": m, "cv_mape_mean": fmc.mape_pct.mean(), "cv_mape_std": fmc.mape_pct.std(), "cv_mape_worst": fmc.mape_pct.max(), "cv_last_fold_mape": fmc.mape_pct.iloc[-1],
                     "cv_mae_pooled": pc["mae"], "cv_rmse_pooled": pc["rmse"], "cv_smape_pooled": pc["smape_pct"],
                     "holdout_mape_clean": hc["mape_pct"], "holdout_mae_clean": hc["mae"], "holdout_rmse_clean": hc["rmse"], "holdout_smape_clean": hc["smape_pct"],
                     "holdout_mae_allrows": ha["mae"], "holdout_rmse_allrows": ha["rmse"], "holdout_mape_allrows": ha["mape_pct"],
                     "coldstart_mape_clean": cc["mape_pct"], "coldstart_mae_clean": cc["mae"]})
cv_tab = pd.concat(rows_cv).round(4)
cv_tab.to_csv(C.OUTPUTS_DIR / "cv_fold_metrics.csv", index=False)
S = pd.DataFrame(rows_sum)
S["stability_score"] = S.cv_mape_mean + S.cv_mape_std
S = S.round(4)
S.to_csv(C.OUTPUTS_DIR / "model_comparison.csv", index=False)
print(S.T.to_string())

# ---------- selection rule (declared before looking at results): lowest mean+std of CV fold MAPE among candidates whose Sep-Oct holdout
# clean MAPE is within 0.10pp of the best holdout; the blend must beat the best single model's score by >= 0.03 to be preferred ----------
ok = S[S.holdout_mape_clean <= S.holdout_mape_clean.min() + 0.10]
single = ok[ok.model != "blend"].sort_values("stability_score").iloc[0]
mechanical = single.model
if "blend" in set(ok.model):
    b = ok[ok.model == "blend"].iloc[0]
    if b.stability_score <= single.stability_score - 0.03:
        mechanical = "blend"
# REVISED rule (changed after seeing results; see decision_log #17): the blend weights are fitted on the same CV predictions that score it,
# so its CV advantage is optimistic. Prefer the blend only if it is ALSO no worse than the best single model on the untouched holdout and the latest CV month.
chosen = single.model
if mechanical == "blend" and b.holdout_mape_clean <= single.holdout_mape_clean and b.cv_last_fold_mape <= single.cv_last_fold_mape:
    chosen = "blend"
selection = {"chosen": chosen, "mechanical_rule_result": mechanical,
             "rule": "min(cv_mape_mean+cv_mape_std) among models with holdout clean MAPE within 0.10pp of best; blend preferred only if >=0.03 better AND not worse on holdout and latest CV month",
             "blend_weights": blend_w, "eligible": ok.model.tolist(), "best_params": {k: v["params"] for k, v in best.items()}, "cold_start_cities": held_cities,
             "features_primary": F_PRIMARY, "features_fallback": F_FALLBACK}
(C.OUTPUTS_DIR / "model_selection.json").write_text(json.dumps(selection, indent=1))
print("CHOSEN:", chosen, "(mechanical rule said:", mechanical, ")")

# ---------- residual analysis (holdout, chosen model, + CV OOF by month) ----------
hd = fd_h[hold.name]; msk = hd["clean_eval"]
ph = hold_p[chosen][hold.name]
ev = train.iloc[hold.eval_idx].copy(); ev["pred"] = ph
ev["ape"] = 100 * (ev.pred - ev[C.TARGET]).abs() / ev[C.TARGET]; ev["sape"] = 100 * (ev.pred - ev[C.TARGET]) / ev[C.TARGET]
evc = ev[~ev.is_label_outlier]
band = pd.cut(evc.distance, [0, 300, 700, 1200, 2000, 4000], labels=["<300", "300-700", "700-1200", "1200-2000", ">2000"])
res = []
for dim, key in [("equipment", evc.equipment), ("distance_band", band), ("month", evc.date.dt.month),
                 ("lane_has_history", (hd["Xap"].loc[evc.index, "lane_n"] + hd["Xap"].loc[evc.index, "rev_n"] > 0).map({True: "yes", False: "no"}))]:
    g = evc.groupby(key, observed=True).agg(n=("ape", "size"), mape_pct=("ape", "mean"), bias_pct=("sape", "mean"), mae=("pred", lambda s: (s - evc.loc[s.index, C.TARGET]).abs().mean()))
    g.insert(0, "dimension", dim); res.append(g.reset_index(names="group"))
resid = pd.concat(res).round(3); resid.to_csv(C.OUTPUTS_DIR / "residual_analysis_holdout.csv", index=False); print(resid.to_string(index=False))
rows = []
for n in names:
    d = fd[n]; ee = train.iloc[folds[names.index(n)].eval_idx]
    e = ee[d["clean_eval"]]; pp = oof[chosen][n][d["clean_eval"]]
    for eq, g in e.assign(ape=100 * np.abs(pp - e[C.TARGET].to_numpy()) / e[C.TARGET].to_numpy()).groupby("equipment"):
        rows.append({"fold": n, "equipment": eq, "mape_pct": g.ape.mean()})
pd.DataFrame(rows).round(3).to_csv(C.OUTPUTS_DIR / "residual_cv_by_fold_equipment.csv", index=False)

# ---------- plots ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for m in ALL:
    f_ = cv_tab[(cv_tab.model == m) & (cv_tab.subset == "clean")]; ax[0].plot(range(3, 9), f_.mape_pct.values, "o-", label=m, lw=2.2 if m == chosen else 1.1)
ax[0].set_title("CV MAPE (%) by eval month, clean rows", loc="left"); ax[0].set_xlabel("eval month 2025"); ax[0].legend(fontsize=8)
S.set_index("model")[["cv_mape_mean", "holdout_mape_clean", "coldstart_mape_clean"]].plot.bar(ax=ax[1], color=["#064A56", "#E07A1F", "#9DAFB3"]); ax[1].set_title("CV mean vs Sep-Oct holdout vs cold-start (MAPE %)", loc="left"); ax[1].tick_params(axis="x", rotation=0)
fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "08_model_comparison.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, dim in zip(ax, ["equipment", "distance_band", "month"]):
    r = resid[resid.dimension == dim]; a.bar(r.group.astype(str), r.mape_pct, color="#064A56"); a2 = a.twinx(); a2.plot(r.group.astype(str), r.bias_pct, "o--", color="#E07A1F"); a2.axhline(0, color="grey", lw=.5)
    a.set_title(f"holdout MAPE (bars) / bias % (dots) by {dim}", loc="left", fontsize=9)
fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "09_residuals_holdout.png", dpi=130); plt.close(fig)

# ---------- importance: SHAP (LightGBM primary, trained Jan-Aug) + permutation for chosen model on holdout ----------
dtr = fd_h[hold.name]
lg = LGBModel(F_PRIMARY, best["lightgbm"]["params"]).fit(dtr["Xtr"], dtr["ytr"])
samp = dtr["Xap"].loc[evc.index].sample(3000, random_state=C.SEED)
sv = shap.TreeExplainer(lg._m).shap_values(samp[F_PRIMARY])
shap_imp = pd.Series(np.abs(sv).mean(0), index=F_PRIMARY).sort_values(ascending=False)
shap_imp.round(5).to_csv(C.OUTPUTS_DIR / "shap_importance_lightgbm.csv", header=["mean_abs_shap_log_rate"])
rng = np.random.default_rng(C.SEED)
cm = routed(chosen) if chosen != "blend" else None
if chosen == "blend":
    sub = {m: routed(m).fit(dtr["Xtr"], dtr["ytr"]) for m in MODELS}
    predict = lambda X: sum(blend_w[m] * sub[m].predict(X) for m in MODELS)
else:
    cm.fit(dtr["Xtr"], dtr["ytr"]); predict = cm.predict
Xh = dtr["Xap"].loc[evc.index]; yh = evc[C.TARGET].to_numpy()
base_m = 100 * np.mean(np.abs(predict(Xh) - yh) / yh)
perm = {}
for f in F_PRIMARY:
    inc = []
    for _ in range(3):
        Xp = Xh.copy(); Xp[f] = rng.permutation(Xp[f].to_numpy())
        inc.append(100 * np.mean(np.abs(predict(Xp) - yh) / yh) - base_m)
    perm[f] = float(np.mean(inc))
pi = pd.Series(perm).sort_values(ascending=False); pi.round(4).to_csv(C.OUTPUTS_DIR / "permutation_importance_chosen.csv", header=["mape_increase_pp"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
shap_imp.head(12)[::-1].plot.barh(ax=ax[0], color="#064A56"); ax[0].set_title("SHAP: mean |value| (LightGBM, log-rate)", loc="left")
pi.head(12)[::-1].plot.barh(ax=ax[1], color="#E07A1F"); ax[1].set_title(f"Permutation: MAPE increase pp ({chosen}, holdout)", loc="left")
fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "10_feature_importance.png", dpi=130); plt.close(fig)
print("top SHAP", shap_imp.head(8).round(4).to_dict()); print("top perm", pi.head(8).round(3).to_dict()); print("total sec", round(time.time() - T0))
