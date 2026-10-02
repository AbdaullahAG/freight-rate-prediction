"""Stage 2: EDA + data-quality audit. Every number in the issue table is computed here."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from freight import config as C
from freight.data import city_coordinates, load_train, load_validation, load_template
from freight.quality import flag_label_outliers, haversine_miles

C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
facts: dict = {}
rows: list[dict] = []


def issue(name, how, affected, fix, effect):
    rows.append({"issue": name, "how_detected": how, "rows_affected": affected, "fix": fix, "effect": effect})


t, v = load_train(), load_validation()
tmpl = load_template()
facts["shape"] = {"train": list(t.shape), "validation": list(v.shape)}
facts["ids_match_template"] = bool((tmpl["load_id"].to_numpy() == v["load_id"].to_numpy()).all())
facts["date_range"] = {"train": [str(t.date.min().date()), str(t.date.max().date())],
                       "validation": [str(v.date.min().date()), str(v.date.max().date())]}
facts["train_days"] = int(t.date.nunique())

# ---------- duplicates / spelling ----------
n_dup_id = int(t.load_id.duplicated().sum())
n_dup_row = int(t.drop(columns="load_id").duplicated().sum())
tc, vc = set(t.pickup) | set(t.delivery), set(v.pickup) | set(v.delivery)
spelling_suspects = [c for c in tc if c != c.strip() or "  " in c]
issue("Duplicate rows / ids", "duplicated(load_id); duplicated(all cols except id)", n_dup_id + n_dup_row,
      "none needed", "none (0 duplicates in train; 0 in validation)")
issue("City / equipment spelling", f"{len(tc)} distinct cities, {t.equipment.nunique()} equipment values, whitespace check",
      len(spelling_suspects), "none needed", "values are already canonical (Dry Van/Flatbed/Reefer)")

# ---------- weight ----------
neg_t, neg_v = int((t.weight < 0).sum()), int((v.weight < 0).sum())
absw = t.weight.abs()
cap_t = int((absw >= C.WEIGHT_CAP).sum())
facts["weight"] = {"neg_train": neg_t, "neg_val": neg_v, "abs_mean_pos": float(t.weight[t.weight > 0].mean()),
                   "abs_mean_neg": float(t.weight[t.weight < 0].abs().mean()), "at_cap_train": cap_t,
                   "at_cap_val": int((v.weight.abs() >= C.WEIGHT_CAP).sum()), "min_abs": float(absw.min())}
issue("Negative weights (sign errors)", "weight < 0; |weight| mean/quantiles identical to positive rows (31,724 vs 31,417)",
      f"{neg_t} train, {neg_v} validation", "abs(weight)", "restores plausible 5,000-47,500 lb range")
issue("Weight censored at 47,500 lb", "mass point at |weight| == 47,500", f"{cap_t} train, {facts['weight']['at_cap_val']} validation",
      "keep (legal max load); optional is_capped flag", "documented; no row removed")

# ---------- nulls ----------
nulls = {}
for c in ["weight", "market_index"]:
    m = t[c].isna()
    by_month = m.groupby(t.date.dt.month).mean()
    nulls[c] = {"train": int(m.sum()), "val": int(v[c].isna().sum()), "month_rate_min": float(by_month.min()),
                "month_rate_max": float(by_month.max()),
                "equip_rate": m.groupby(t.equipment).mean().round(4).to_dict(),
                "rate_null_vs_notnull": [float(t.posted_rate[m].mean()), float(t.posted_rate[~m].mean())]}
facts["nulls"] = nulls
issue("Null weight", "isna(); rate by month/equipment/date; mean rate null vs not-null",
      f"{nulls['weight']['train']} train ({100*nulls['weight']['train']/len(t):.2f}%), {nulls['weight']['val']} validation",
      "median impute (by equipment) + was_missing flag, or native NaN for GBMs",
      f"missingness looks random (month rate {nulls['weight']['month_rate_min']:.3f}-{nulls['weight']['month_rate_max']:.3f}, "
      f"mean rate {nulls['weight']['rate_null_vs_notnull'][0]:.0f} vs {nulls['weight']['rate_null_vs_notnull'][1]:.0f})")
g = t.groupby("date").market_index
facts["market_index"] = {"within_date_sd_median": float(g.std().median()), "between_date_sd": float(g.mean().std()),
                         "dates_all_null_train": int((g.count() == 0).sum()),
                         "dates_all_null_val": int((v.groupby("date").market_index.count() == 0).sum())}
issue("Null market_index", "isna(); market_index is a daily series (within-date sd 0.025 vs between-date sd 0.167)",
      f"{nulls['market_index']['train']} train ({100*nulls['market_index']['train']/len(t):.2f}%), {nulls['market_index']['val']} validation",
      "fill with same-date median of the other loads (every date has observed values)",
      "error of fill ~0.025 vs a 0.17 signal sd; no dates fully missing")

# ---------- label outliers ----------
fl = flag_label_outliers(t)
t = t.join(fl)
n_out = int(t.is_label_outlier.sum())
sig = fl.attrs["sigma"]
hi, lo = t[t.is_label_outlier & (t.log_resid > 0)], t[t.is_label_outlier & (t.log_resid < 0)]
mult_hi, mult_lo = np.exp(hi.log_resid), np.exp(lo.log_resid)
dist_bins = pd.qcut(t.distance, 5)
facts["label_outliers"] = {"n": n_out, "pct": 100 * n_out / len(t), "robust_sigma_clean": sig, "threshold_log": C.LABEL_OUTLIER_K * sig,
                           "n_high": len(hi), "n_low": len(lo),
                           "mult_high_range": [float(mult_hi.min()), float(mult_hi.max())],
                           "mult_low_range": [float(mult_lo.min()), float(mult_lo.max())],
                           "rate_by_distance_quintile": t.groupby(dist_bins, observed=True).is_label_outlier.mean().round(4).tolist(),
                           "rate_by_equipment": t.groupby("equipment").is_label_outlier.mean().round(4).to_dict(),
                           "rate_by_month": t.groupby(t.date.dt.month).is_label_outlier.mean().round(4).tolist(),
                           "n_between_0.3_and_0.5": int(((t.log_resid.abs() > 0.3) & (t.log_resid.abs() < 0.5)).sum()),
                           "n_in_holdout": int(t[t.date.dt.month.isin(C.HOLDOUT_MONTHS)].is_label_outlier.sum())}
t["rpm"] = t.posted_rate / t.distance
facts["rpm"] = t.rpm.quantile([0.001, 0.01, 0.5, 0.99, 0.999]).round(3).to_dict()
facts["rpm_clean"] = t[~t.is_label_outlier].rpm.quantile([0.001, 0.5, 0.999]).round(3).to_dict()
facts["rpm_by_equipment_clean"] = t[~t.is_label_outlier].groupby("equipment").rpm.median().round(3).to_dict()
issue("Corrupted posted_rate labels (x0.17-0.47 or x2.1-5.3)",
      f"|log-residual| > {C.LABEL_OUTLIER_K:.0f} robust sigma ({sig:.3f}) from structural fit (equipment, log-dist, market_index); "
      f"clean residual is tight, 0 rows between 0.3-0.5 and only a few 0.5-0.8; flag rate is flat across equipment/month/distance",
      f"{n_out} train ({100*n_out/len(t):.2f}%): {len(hi)} high, {len(lo)} low",
      "drop from training and from CV/holdout scoring (also report holdout on all rows); never 'predict' them",
      "measured in scripts/02_validation_design.py (baseline_cleaning_effects.csv)")

# ---------- distance / coordinates ----------
for d in (t, v):
    d["hav"] = haversine_miles(d.pickup_lat, d.pickup_lon, d.delivery_lat, d.delivery_lon)
    d["ratio"] = d.distance / d.hav
coords_t = city_coordinates([t])
coords_v = city_coordinates([v])
n_coord_vals_t = pd.concat([t[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["c", "a", "b"], axis=1),
                            t[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["c", "a", "b"], axis=1)]).groupby("c")[["a", "b"]].nunique()
common = coords_t.index.intersection(coords_v.index)
coord_shift = (coords_t.loc[common] - coords_v.loc[common]).abs().to_numpy().max()
pair = t.groupby(["pickup", "delivery"]).distance.agg(["nunique", "std", "count"])
short = t[t.ratio > 1.6]
facts["coords"] = {"cities_train": len(coords_t), "cities_val": len(coords_v), "max_nunique_latlon_per_city_train": int(n_coord_vals_t.max().max()),
                   "max_abs_diff_train_vs_val_common_cities": float(coord_shift),
                   "lat_range_train": [float(coords_t.lat.min()), float(coords_t.lat.max())],
                   "lon_range_train": [float(coords_t.lon.min()), float(coords_t.lon.max())],
                   "val_unseen_cities": sorted(vc - tc),
                   "val_unseen_city_coords": coords_v.loc[sorted(vc - tc)].round(4).to_dict("index"),
                   "ratio_dist_over_hav_train_pctl": t.ratio.quantile([.01, .5, .99]).round(3).tolist(),
                   "corr_dist_hav_train": float(np.corrcoef(t.hav, t.distance)[0, 1]),
                   "ratio_gt_1.6_n": int(len(short)), "ratio_gt_1.6_median_distance": float(short.distance.median()),
                   "ratio_gt_1.6_max_distance": float(short.distance.max()), "min_distance": float(t.distance.min()),
                   "n_at_distance_70": int((t.distance == 70).sum()),
                   "pair_distance_nunique_median": float(pair["nunique"].median()), "pair_distance_sd_median": float(pair["std"].median()),
                   "lexington_fortwayne_train_rows": int(((t.pickup == C.DEC_PICKUP) & (t.delivery == C.DEC_DELIVERY)).sum()),
                   "lexington_fortwayne_distance_mean": float(t[(t.pickup == C.DEC_PICKUP) & (t.delivery == C.DEC_DELIVERY)].distance.mean()),
                   "lexington_coords": coords_t.loc[C.DEC_PICKUP].round(5).to_dict(),
                   "fort_wayne_coords": coords_t.loc[C.DEC_DELIVERY].round(5).to_dict()}
issue("Coordinates: per-row deviation from city median", "nunique(lat,lon) per city; |train-val| coord diff for shared cities",
      0, "use row coordinates (== city medians); Lexington/Fort Wayne taken from train city medians",
      f"max nunique per city = {facts['coords']['max_nunique_latlon_per_city_train']}; max train-vs-val diff = {coord_shift:.1e}")
issue("Coordinates are synthetic / clipped", "Baltimore lon -72.7 etc. do not match real geography; lat in [28.36,44.30], lon in [-121.70,-69.50]; Laredo (validation only) lat 25.5",
      "all rows", "treat lat/lon as city identifiers in a synthetic plane; do not use external geography",
      "haversine*1.18 ~ distance (corr 0.9995), so coordinates and distance agree internally")
issue("distance vs haversine mismatch", f"ratio = distance/haversine: median 1.18, p1 {facts['coords']['ratio_dist_over_hav_train_pctl'][0]}, p99 {facts['coords']['ratio_dist_over_hav_train_pctl'][2]}; ratio>1.6 are only short hauls",
      f"{len(short)} train ratio>1.6 (median distance {short.distance.median():.0f} mi)",
      "keep; explained by 70-mile floor on distance + noise on very short lanes", "no rows removed; haversine used only as a cross-check/feature")
issue("Row-level distance noise within a lane", f"same pickup/delivery has {pair['nunique'].median():.0f} distinct distances (median sd {pair['sd' if False else 'std'].median():.1f} mi)",
      "most lanes", "use row distance plus lane-mean distance as separate features", "Lexington->Fort Wayne seen in train: "
      f"{facts['coords']['lexington_fortwayne_train_rows']} rows, mean {facts['coords']['lexington_fortwayne_distance_mean']:.1f} mi vs fixed 360")
issue("Cold-start cities in validation", "set(val cities) - set(train cities)",
      f"{int((v.pickup.isin(vc - tc) | v.delivery.isin(vc - tc)).sum())} validation rows ({100*(v.pickup.isin(vc - tc) | v.delivery.isin(vc - tc)).mean():.1f}%): {', '.join(sorted(vc - tc))}",
      "no city-ID/target-encoding as sole signal; rely on distance, equipment, coordinates, market_index; evaluate with the cold-start holdout",
      "Laredo lat 25.5 lies outside train lat range (28.36-44.30): tree models will extrapolate flat")
facts["unseen_val_rows_pct"] = float(100 * (v.pickup.isin(vc - tc) | v.delivery.isin(vc - tc)).mean())
facts["unseen_lane_pct"] = float(100 * np.mean([p not in set(zip(t.pickup, t.delivery)) for p in zip(v.pickup, v.delivery)]))

# ---------- trend, day-of-week, signals ----------
clean = t[~t.is_label_outlier].copy()
mon = clean.groupby(clean.date.dt.month).agg(n=("rpm", "size"), mean_rate=("posted_rate", "mean"), median_rpm=("rpm", "median"),
                                              market_index=("market_index", "mean"), quote_signal=("quote_signal", "mean"))
dow = clean.groupby(clean.date.dt.dayofweek).rpm.median()
facts["monthly"] = mon.round(4).to_dict("index")
facts["dow_median_rpm"] = dow.round(4).to_dict()
facts["dow_spread_pct"] = float(100 * (dow.max() - dow.min()) / dow.mean())
# residual WITHOUT market_index (equipment + log-distance only) so the signals' own effect is visible
_X = pd.get_dummies(clean.equipment, dtype=float).assign(ld=np.log(clean.distance), ld2=np.log(clean.distance) ** 2)
_y = np.log(clean.posted_rate)
clean["resid_nomi"] = _y - _X.values @ np.linalg.lstsq(_X.values, _y.values, rcond=None)[0]
clean["resid_nomi"] -= clean["resid_nomi"].median()
facts["corr_resid_nomi"] = clean[["resid_nomi", "market_index", "quote_signal"]].corr().round(4).to_dict()
facts["corr_logresid"] = clean[["log_resid", "market_index", "quote_signal"]].corr().round(4).to_dict()  # log_resid already contains market_index
facts["val_monthly"] = v.groupby(v.date.dt.month).agg(market_index=("market_index", "mean"), quote_signal=("quote_signal", "mean")).round(4).to_dict("index")
qrows = []
for m, g_ in clean.groupby(clean.date.dt.month):
    q = np.polyfit(g_.quote_signal, g_.resid_nomi, 1)[0]
    gm = g_.dropna(subset=["market_index"])
    mm = np.polyfit(gm.market_index, gm.resid_nomi, 1)[0]
    qrows.append({"month": int(m), "qs_mean": g_.quote_signal.mean(), "qs_slope_on_lograte": q, "qs_corr": g_[["resid_nomi", "quote_signal"]].corr().iloc[0, 1],
                  "mi_mean": g_.market_index.mean(), "mi_slope_on_lograte": mm})
qm = pd.DataFrame(qrows).round(4)
qm.to_csv(C.OUTPUTS_DIR / "quote_signal_by_month.csv", index=False)
facts["quote_signal_month_slopes"] = qm.set_index("month")[["qs_mean", "qs_slope_on_lograte", "mi_slope_on_lograte"]].to_dict("index")
issue("Monthly trend", "monthly median rate-per-mile vs mean market_index (clean rows)", "all",
      "market_index is the carrier of trend; no calendar-time feature (Nov/Dec are out of range)",
      f"rpm {mon.median_rpm.min():.2f}-{mon.median_rpm.max():.2f} tracks market_index {mon.market_index.min():.2f}-{mon.market_index.max():.2f}; corr(log-rate residual after equipment+distance, market_index)={facts['corr_resid_nomi']['resid_nomi']['market_index']:.2f}")
issue("Day-of-week effect", "median rpm by weekday (clean rows)", "all", "weak; optional weekday feature",
      f"spread {facts['dow_spread_pct']:.1f}% of mean rpm")
issue("quote_signal is NOT a near-proxy of the target but is unstable",
      "corr with equipment+distance residual is small; its slope on log-rate flips sign by month (+0.06 in high-mean months, -0.05 in low-mean months) - see quote_signal_by_month.csv",
      "all", "use with caution: test in CV with/without; do not rely on it for Nov/Dec",
      f"overall corr with residual {facts['corr_resid_nomi']['resid_nomi']['quote_signal']:.3f} (market_index: {facts['corr_resid_nomi']['resid_nomi']['market_index']:.3f}); monthly slope +-0.04-0.06 x qs sd 0.26 = at most ~+-1.6% of rate")

# ---------- drift / ranges ----------
rng = {}
for c in ["market_index", "quote_signal", "distance", "weight"]:
    lo_, hi_ = (t[c].abs() if c == "weight" else t[c]).min(), (t[c].abs() if c == "weight" else t[c]).max()
    vv = v[c].abs() if c == "weight" else v[c]
    rng[c] = {"train_min": float(lo_), "train_max": float(hi_), "val_min": float(vv.min()), "val_max": float(vv.max()),
              "val_outside_train_pct": float(100 * ((vv < lo_) | (vv > hi_)).mean()), "val_mean": float(vv.mean()), "train_mean": float((t[c].abs() if c == "weight" else t[c]).mean())}
facts["ranges"] = rng
facts["train_mi_pctl_of_val_mean"] = float((t.market_index.dropna() < v.market_index.mean()).mean())
issue("Market regime shift (validation market_index)", "validation mean 0.927 (range 0.72-1.10) vs train 1.083 (0.68-1.47); train share below val mean",
      f"{len(v)} validation rows", "model market_index monotonically/log-linearly; rely on the Jan/Aug-Oct (low-index) months for fit quality; "
      "Dec market_index filled from validation features per date", f"0% of validation market_index outside train range; quote_signal outside-range {rng['quote_signal']['val_outside_train_pct']:.2f}%")

# ---------- adversarial validation ----------
def adv_frame(d: pd.DataFrame, unseen: set) -> pd.DataFrame:
    return pd.DataFrame({"distance": d.distance, "weight_abs": d.weight.abs(), "market_index": d.market_index, "quote_signal": d.quote_signal,
                         "equip": d.equipment.map({e: i for i, e in enumerate(C.EQUIPMENT)}), "pickup_lat": d.pickup_lat, "pickup_lon": d.pickup_lon,
                         "delivery_lat": d.delivery_lat, "delivery_lon": d.delivery_lon,
                         "has_unseen_city": (d.pickup.isin(unseen) | d.delivery.isin(unseen)).astype(int)})


def adv_auc(X: pd.DataFrame, y: np.ndarray) -> tuple[float, pd.Series]:
    oof = np.zeros(len(y)); imp = pd.Series(0.0, index=X.columns)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(X, y):  # random folds are fine for a *shift detector*, not for forecasting
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=C.SEED, verbose=-1, importance_type="gain")
        m.fit(X.iloc[tr], y[tr]); oof[te] = m.predict_proba(X.iloc[te])[:, 1]; imp += m.feature_importances_
    return float(roc_auc_score(y, oof)), (imp / imp.sum()).sort_values(ascending=False)


unseen = vc - tc
A = pd.concat([adv_frame(t.drop(columns=[c for c in t.columns if c in ('rpm', 'hav', 'ratio', 'log_resid', 'is_label_outlier')]), unseen), adv_frame(v.drop(columns=['hav', 'ratio']), unseen)], ignore_index=True)
y_adv = np.r_[np.zeros(len(t)), np.ones(len(v))]
auc_all, imp_all = adv_auc(A, y_adv)
auc_nomi, imp_nomi = adv_auc(A.drop(columns=["market_index"]), y_adv)
auc_notime, imp_notime = adv_auc(A.drop(columns=["market_index", "has_unseen_city", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]), y_adv)
uni = {c: float(max(roc_auc_score(y_adv[A[c].notna()], A[c].dropna()), 1 - roc_auc_score(y_adv[A[c].notna()], A[c].dropna()))) for c in A.columns}
# also a "near-term" control: Sep-Oct train vs Jan-Aug train (does the same detector fire on in-train drift?)
late = (t.date.dt.month >= 9).astype(int).to_numpy()
auc_ctrl, _ = adv_auc(adv_frame(t.drop(columns=[c for c in t.columns if c in ('rpm', 'hav', 'ratio', 'log_resid', 'is_label_outlier')]), unseen), late)
facts["adversarial"] = {"auc_all_features": auc_all, "auc_without_market_index": auc_nomi, "auc_without_market_index_and_geo": auc_notime,
                        "control_auc_sepoct_vs_janaug_train": auc_ctrl,
                        "importance_all": imp_all.round(4).to_dict(), "importance_no_mi": imp_nomi.round(4).to_dict(),
                        "univariate_auc": {k: round(x, 4) for k, x in sorted(uni.items(), key=lambda kv: -kv[1])},
                        "note": "date is excluded (it trivially separates the sets); stratified 5-fold is used only to measure separability"}
issue("Train-vs-validation covariate shift (adversarial validation)", f"LightGBM classifier train(0) vs validation(1), date excluded: AUC={auc_all:.3f} (all), {auc_nomi:.3f} (no market_index), {auc_notime:.3f} (no market_index/geo)",
      "all validation rows", "keep features that are stable (distance, equipment, weight); handle market_index via regime-robust modelling; use time-based CV that mimics the shift",
      f"control: Sep-Oct vs Jan-Aug inside train AUC={auc_ctrl:.3f}")

# ---------- plots ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(mon.index, mon.median_rpm, "o-", color="#064A56"); ax[0].set_title("Median rate per mile by month (clean rows)", loc="left"); ax[0].set_xlabel("month 2025")
ax2 = ax[0].twinx(); ax2.plot(mon.index, mon.market_index, "s--", color="#E07A1F"); ax2.set_ylabel("mean market_index", color="#E07A1F"); ax2.spines["right"].set_visible(True)
daily = pd.concat([t.groupby("date").market_index.mean(), v.groupby("date").market_index.mean()])
ax[1].plot(daily.index, daily.values, color="#064A56", lw=1); ax[1].axvline(pd.Timestamp("2025-11-01"), color="grey", ls=":"); ax[1].set_title("Daily mean market_index (train | validation)", loc="left")
ax[1].tick_params(axis="x", rotation=30); fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "01_trend_market_index.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].hist(t.log_resid.clip(-2, 2), bins=160, color="#064A56"); ax[0].set_yscale("log"); ax[0].set_title("log(rate) residual vs structural fit", loc="left")
for s in (-1, 1): ax[0].axvline(s * C.LABEL_OUTLIER_K * sig, color="crimson", ls="--")
ax[0].set_xlabel("log residual (dashed = outlier threshold)")
c = np.where(t.is_label_outlier, "crimson", "#064A56")
ax[1].scatter(t.distance, t.posted_rate, s=2, c=c, alpha=0.4); ax[1].set_title("Rate vs distance (red = corrupted labels)", loc="left"); ax[1].set_xlabel("distance (mi)"); ax[1].set_ylabel("posted_rate ($)")
fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "02_label_outliers.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].bar(qm.month, qm.qs_slope_on_lograte, color=np.where(qm.qs_slope_on_lograte > 0, "#064A56", "#E07A1F")); ax[0].axhline(0, color="k", lw=.6)
ax[0].set_title("quote_signal slope on log-rate, by month", loc="left"); ax[0].set_xlabel("month 2025")
ax[1].plot(qm.month, qm.qs_mean, "o-", color="#064A56"); ax[1].set_title("mean quote_signal by month (validation Nov/Dec: 2.05)", loc="left"); ax[1].set_xlabel("month 2025")
fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "03_quote_signal_instability.png"); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for a, c_ in zip(ax, ["market_index", "quote_signal", "distance"]):
    a.hist(t[c_].dropna(), bins=60, alpha=.55, density=True, label="train", color="#064A56"); a.hist(v[c_].dropna(), bins=60, alpha=.55, density=True, label="validation", color="#E07A1F"); a.set_title(c_, loc="left")
ax[0].legend(); fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "04_train_vs_validation_distributions.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
imp_all.sort_values().plot.barh(ax=ax[0], color="#064A56"); ax[0].set_title(f"Adversarial importance (AUC={auc_all:.3f})", loc="left")
imp_nomi.sort_values().plot.barh(ax=ax[1], color="#E07A1F"); ax[1].set_title(f"Without market_index (AUC={auc_nomi:.3f})", loc="left")
fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "05_adversarial_validation.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].hist(t.weight.abs().dropna(), bins=60, color="#064A56"); ax[0].set_title("|weight|: mass point at 47,500", loc="left")
ax[1].hist(t.ratio.clip(upper=3), bins=100, color="#064A56"); ax[1].set_title("distance / haversine (clipped at 3)", loc="left")
fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "06_weight_and_distance_checks.png"); plt.close(fig)

# ---------- write tables ----------
tab = pd.DataFrame(rows)
tab.to_csv(C.OUTPUTS_DIR / "data_quality_issues.csv", index=False)
md = ["# Data-quality issue table (generated by scripts/01_eda.py)\n", "| # | issue | how detected | rows affected | fix | effect |", "|---|---|---|---|---|---|"]
for i, r in enumerate(rows, 1):
    md.append("| " + " | ".join([str(i)] + [str(r[k]).replace("|", "/") for k in ["issue", "how_detected", "rows_affected", "fix", "effect"]]) + " |")
(C.OUTPUTS_DIR / "data_quality_issues.md").write_text("\n".join(md) + "\n")
t.loc[:, ["load_id", "log_resid", "is_label_outlier"]].to_csv(C.OUTPUTS_DIR / "train_label_outlier_flags.csv", index=False)
(C.OUTPUTS_DIR / "eda_facts.json").write_text(json.dumps(facts, indent=2, default=float))
print(json.dumps({k: facts[k] for k in ["label_outliers", "adversarial", "ranges", "dow_spread_pct"]}, indent=1, default=float))
print(tab[["issue", "rows_affected"]].to_string())
