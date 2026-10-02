"""December fixed-scenario inputs. Uses validation.csv FEATURES only to fill market_index / quote_signal (permitted use b)."""
from __future__ import annotations

import pandas as pd

from . import config as C
from .data import city_coordinates


def december_signals(val: pd.DataFrame, dec_dates: pd.Series, lane_equip_min_rows: int = 5) -> dict[str, pd.DataFrame]:
    """Option A: per-date mean of December validation rows (same lane+equipment if >= lane_equip_min_rows exist for that date, else all rows of that date).
    Option B: freeze at the last validation date before December (2025-11-30) daily mean."""
    dec = val[val["date"] >= pd.Timestamp(f"{C.YEAR}-12-01")]
    same = dec[(dec.pickup == C.DEC_PICKUP) & (dec.delivery == C.DEC_DELIVERY) & (dec.equipment == "Dry Van")]
    allmean = dec.groupby("date")[["market_index", "quote_signal"]].mean()
    samemean = same.groupby("date")[["market_index", "quote_signal"]].agg(["mean", "count"])
    a = allmean.reindex(pd.DatetimeIndex(dec_dates))
    for dte in samemean.index:  # only used when enough same-lane rows exist (they do not: 3 rows in total)
        if samemean.loc[dte, ("market_index", "count")] >= lane_equip_min_rows:
            a.loc[dte] = samemean.loc[dte, [("market_index", "mean"), ("quote_signal", "mean")]].to_numpy()
    last = val[val["date"] == val.loc[val["date"] < pd.Timestamp(f"{C.YEAR}-12-01"), "date"].max()][["market_index", "quote_signal"]].mean()
    b = pd.DataFrame({"market_index": last["market_index"], "quote_signal": last["quote_signal"]}, index=pd.DatetimeIndex(dec_dates))
    return {"A_dec_validation_daily_mean": a, "B_frozen_last_nov_day": b}


def december_frame(dec_template: pd.DataFrame, train: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Raw-schema frame (same columns as train minus label) for the 31 fixed December rows."""
    coords = city_coordinates([train])
    out = dec_template.drop(columns=["predicted_rate"]).copy()
    out["pickup_lat"], out["pickup_lon"] = coords.loc[C.DEC_PICKUP, "lat"], coords.loc[C.DEC_PICKUP, "lon"]
    out["delivery_lat"], out["delivery_lon"] = coords.loc[C.DEC_DELIVERY, "lat"], coords.loc[C.DEC_DELIVERY, "lon"]
    out["market_index"] = signals["market_index"].to_numpy()
    out["quote_signal"] = signals["quote_signal"].to_numpy()
    return out.reset_index(drop=True)
