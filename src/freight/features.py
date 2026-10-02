"""Feature engineering. Everything that touches the target is computed from a *training slice* only and, for
training rows, from strictly earlier months of that slice (time-respecting, out-of-fold)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .quality import haversine_miles

EQUIP_CODE = {e: i for i, e in enumerate(C.EQUIPMENT)}
WEIGHT_BINS = [0, 15_000, 25_000, 35_000, 45_000, 1e9]
LANE_SHRINK_K = 10.0


def prepare(df: pd.DataFrame, mi_fallback: float | None = None) -> pd.DataFrame:
    """Label-free row features. Uses only the row itself and, for labelled data, same-date peers (market_index null fill).

    `mi_fallback`: if given, null market_index is filled with this constant instead of peers (used for validation.csv so that
    its features are not used beyond the two permitted purposes: adversarial validation and the December fill).
    """
    o = pd.DataFrame(index=df.index)
    o["date"] = df["date"]
    o["month"] = df["date"].dt.month
    o["dow"] = df["date"].dt.dayofweek
    o["doy"] = df["date"].dt.dayofyear
    o["tday"] = (df["date"] - pd.Timestamp(f"{C.YEAR}-01-01")).dt.days.astype(float) / 100.0  # linear time index (extrapolates in linear parts only)
    o["distance"] = df["distance"].astype(float)
    o["log_distance"] = np.log(o["distance"])
    o["haversine"] = haversine_miles(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    ratio = o["distance"] / o["haversine"].clip(lower=1.0)
    o["dist_over_hav"] = ratio
    o["mismatch"] = ((ratio > 1.6) | (ratio < 1.05)).astype(int)
    o["equip"] = df["equipment"].map(EQUIP_CODE).astype(int)
    w = df["weight"].abs()
    o["weight"] = w
    o["weight_missing"] = w.isna().astype(int)
    o["weight_capped"] = (w >= C.WEIGHT_CAP).astype(int)
    o["weight_bin"] = pd.cut(w, WEIGHT_BINS, labels=False, right=True).astype(float)
    mi_fill = df.groupby("date")["market_index"].transform("median") if mi_fallback is None else mi_fallback
    o["market_index"] = df["market_index"].fillna(mi_fill)
    o["mi_missing"] = df["market_index"].isna().astype(int)
    o["quote_signal"] = df["quote_signal"].astype(float)
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        o[c] = df[c].astype(float)
    o["lane"] = df["pickup"] + ">" + df["delivery"]
    o["rev_lane"] = df["delivery"] + ">" + df["pickup"]
    return o


def _month_cum(frame: pd.DataFrame, value: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per lane: sum and count by month (columns 1..12)."""
    s = frame.pivot_table(index="lane", columns="month", values=value, aggfunc="sum", fill_value=0.0)
    n = frame.pivot_table(index="lane", columns="month", values=value, aggfunc="count", fill_value=0.0)
    cols = list(range(1, 13))
    return s.reindex(columns=cols, fill_value=0.0), n.reindex(columns=cols, fill_value=0.0)


def lane_target_stats(tr: pd.DataFrame, y_log_rpm: pd.Series, ap: pd.DataFrame, k: float = LANE_SHRINK_K) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shrunk lane mean of month-demeaned log rate-per-mile, forward lane and reverse lane.

    tr: prepared TRAINING rows (clean labels only), y_log_rpm: their log(rate/distance).
    Training rows see only rows of strictly earlier months; `ap` rows (eval / validation / December) see all of tr.
    """
    res = y_log_rpm - y_log_rpm.groupby(tr["month"]).transform("mean")
    work = pd.DataFrame({"lane": tr["lane"], "month": tr["month"], "res": res})
    s, n = _month_cum(work, "res")
    s_prev = s.cumsum(axis=1).shift(1, axis=1).fillna(0.0)
    n_prev = n.cumsum(axis=1).shift(1, axis=1).fillna(0.0)
    s_all, n_all = s.sum(axis=1), n.sum(axis=1)

    def lookup_prev(lanes: pd.Series, months: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        idx = s_prev.index.get_indexer(lanes)
        ok = idx >= 0
        ss, nn = np.zeros(len(lanes)), np.zeros(len(lanes))
        mm = months.to_numpy() - 1
        ss[ok] = s_prev.to_numpy()[idx[ok], mm[ok]]
        nn[ok] = n_prev.to_numpy()[idx[ok], mm[ok]]
        return ss, nn

    def lookup_all(lanes: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        return s_all.reindex(lanes).fillna(0.0).to_numpy(), n_all.reindex(lanes).fillna(0.0).to_numpy()

    out = []
    for frame, mode in ((tr, "prev"), (ap, "all")):
        cols = {}
        for tag, lane_col in (("lane", "lane"), ("rev", "rev_lane")):
            ss, nn = lookup_prev(frame[lane_col], frame["month"]) if mode == "prev" else lookup_all(frame[lane_col])
            cols[f"{tag}_stat"] = ss / (nn + k)
            cols[f"{tag}_n"] = nn
        out.append(pd.DataFrame(cols, index=frame.index))
    return out[0], out[1]


def lane_distance_stats(tr: pd.DataFrame, ap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-free lane mean distance (leave-one-out for training rows), and the row's deviation from it."""
    g = tr.groupby("lane")["distance"].agg(["sum", "count"])
    def build(frame: pd.DataFrame, loo: bool) -> pd.DataFrame:
        s = g["sum"].reindex(frame["lane"]).fillna(0.0).to_numpy()
        n = g["count"].reindex(frame["lane"]).fillna(0.0).to_numpy()
        if loo:
            s, n = s - frame["distance"].to_numpy(), n - 1
        mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        return pd.DataFrame({"lane_mean_dist": mean, "dist_minus_lane": frame["distance"].to_numpy() - mean}, index=frame.index)
    return build(tr, True), build(ap, False)


def build_xy(tr_raw: pd.DataFrame, ap_raw: pd.DataFrame, tr_clean_mask: pd.Series, ap_mi_fallback: float | None = None) -> dict:
    """Prepare a (train slice, apply slice) pair with every candidate feature. `tr_clean_mask` marks clean labels.
    Returns Xtr (clean rows only), ytr (rate), Xap (all rows), plus raw frames for bookkeeping."""
    tr_p, ap_p = prepare(tr_raw), prepare(ap_raw, ap_mi_fallback)
    y = tr_raw[C.TARGET].astype(float)
    mask = tr_clean_mask.to_numpy()
    tr_c = tr_p[mask]
    y_c = y[mask]
    lt_tr, lt_ap = lane_target_stats(tr_c, np.log(y_c / tr_c["distance"]), ap_p)
    ld_tr, ld_ap = lane_distance_stats(tr_p, ap_p)  # label-free: use every train row
    Xtr = pd.concat([tr_c, lt_tr, ld_tr.loc[tr_c.index]], axis=1)
    Xap = pd.concat([ap_p, lt_ap, ld_ap], axis=1)
    return {"Xtr": Xtr, "ytr": y_c, "Xap": Xap}


# ---------- feature groups (ablation) ----------
BASE = ["distance", "log_distance", "haversine", "dist_over_hav", "mismatch", "equip", "weight", "weight_missing", "weight_capped", "weight_bin"]
FEATURE_SETS = {
    "G0_base_no_market": BASE + ["dow"],
    "G1_+market_index": BASE + ["dow", "market_index", "mi_missing"],
    "G2_+quote_signal_only": BASE + ["dow", "quote_signal"],
    "G3_+both_signals": BASE + ["dow", "market_index", "mi_missing", "quote_signal"],
}
LANE_TARGET = ["lane_stat", "lane_n", "rev_stat", "rev_n"]
LANE_DIST = ["lane_mean_dist", "dist_minus_lane"]
GEO = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
CALENDAR = ["month", "doy"]
