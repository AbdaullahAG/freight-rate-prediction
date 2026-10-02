# Validation design and why a random split is wrong

**The task is forecasting.** `train_test.csv` covers 2025-01-01 -> 2025-10-31 and `validation.csv` covers
2025-11-01 -> 2025-12-31, i.e. the final predictions are for *the future*, starting one day after the last labelled row.
The strongest driver of rate, `market_index`, is a daily market series that drifts month to month (train 0.68-1.47,
validation 0.72-1.10, mean 0.93 vs 1.08). A random K-fold would put rows from the same day (and the same market regime)
in both train and eval, so the model would be rewarded for interpolating inside a regime it has already seen. That
overstates accuracy and picks hyper-parameters that exploit the day-level `market_index`/`quote_signal` structure instead
of generalising forward. Time-based splits reproduce the real situation: fit on the past, score on a later month.

## Splits (implemented in `src/freight/cv.py`, asserted in `tests/test_cv.py`)

| Purpose | Train | Evaluate | Notes |
|---|---|---|---|
| Expanding-window CV (6 folds) | Jan -> month k-1 | month k, k = Mar..Aug | Optuna tunes **only** here; no Sep/Oct row is used |
| Final holdout | Jan-Aug | Sep-Oct | evaluated **once**, to choose among already-tuned candidates and to report |
| Cold-start holdout (diagnostic) | Jan-Aug, rows touching 8 seeded cities removed | Sep-Oct rows touching those cities | mimics the 8 cities that exist only in validation.csv (12.1% of its rows) |
| Final refit | Jan-Oct (all labelled rows) | validation.csv (no labels) | predictions for submission |

Every split satisfies `max(train date) < min(eval date)` (1-day gap at each boundary, same as train -> validation).
Features must not use the target of later rows (no target/lane statistics computed outside the training slice of a fold).

## Use of validation.csv *features* (not labels) - not leakage
`validation.csv` has no labels. Its features are used only for (a) adversarial validation (a drift diagnostic) and
(b) filling the December `market_index` / `quote_signal` for the fixed December scenario. No model is fit to it.
We claim no accuracy on validation.csv; Spotter computes the final score.

## Caveats stated up front
* Metric is undisclosed -> MAE, RMSE, MAPE, sMAPE, RMSLE, R2, median APE are all reported.
* 677 train rows (1.41%) have corrupted labels (x0.17-0.47 or x2.1-5.3). Models are trained on clean rows; metrics are reported on clean rows and on all rows.
* The label-outlier flagging fit used all train rows (6-parameter structural fit, labels only used for flagging). It cannot leak
  eval information into model training, but it was not fold-specific; we disclose this.
* Sep-Oct holdout is not scored anywhere in Part A.
