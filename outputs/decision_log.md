# Decision log

| # | Decision | Reason (evidence) |
|---|---|---|
| 1 | Run project in a venv with pandas 2.3.3 | provided requirements.txt caps pandas <3; sandbox default is 3.0.2 |
| 2 | Time-based expanding monthly CV on Jan-Aug (eval Mar-Aug); first fold trains on 2 months | validation is the future; need >=2 months of history |
| 3 | Sep-Oct = untouched holdout, scored once in Part B | selection among tuned candidates without touching CV tuning |
| 4 | abs(weight) | negative rows have the same |weight| distribution; baseline MAE 71.0 -> 69.0 |
| 5 | Treat 677 labels as corrupted, exclude from training and from "clean" metrics; also report all-row metrics | multiplicative x0.17-0.47 / x2.1-5.3, none in 0.3-0.5 log gap, flat across segments |
| 6 | Row lat/lon == city medians; use train medians for Lexington (36.99152,-84.99876) and Fort Wayne (41.31561,-85.36206) | 0 cities with >1 coordinate pair in train; identical in validation |
| 7 | No city-ID dependence as the only geographic signal; add a cold-start holdout | 8 validation-only cities = 12.1% of rows |
| 8 | quote_signal is optional and must earn its place in CV | slope on log-rate flips sign by month; weak overall |
| 9 | No calendar-time features (month/day-of-year/trend index) | validation months are outside the train range; market_index carries the regime |
| 10 | Report several metrics | Spotter's metric is not disclosed |
| 11 | Adversarial validation uses stratified random folds | it measures separability, not forecasting; date excluded |
| 12 | Target = log(rate) (not rate-per-mile / log-rpm) | CV LightGBM fold-mean MAPE: log_rate 3.97 vs rpm 4.45 vs log_rpm 4.53 (target_comparison.csv) |
| 13 | quote_signal EXCLUDED | hurts in every setting: +0.83pp MAPE over market_index alone (3.97 vs 3.14, G3 vs G1), fold MAPE up to 11.3-11.9% in month 4, slope on log-rate flips sign by month (signal_stability_by_fold.csv). Validation/December values are inside the train range (1.23-2.90 vs 0.69-3.61), so range is not the problem; instability is. |
| 14 | market_index INCLUDED (missing-filled by same-date peers in train; constant fallback 0.944 = mean of last 7 train days for the 249 validation nulls) | cuts CV MAPE 4.19 -> 3.14 (G0 -> G1); validation 0.72-1.10 is inside train 0.68-1.47. Gain is not uniform: fold m08 got worse without lane/geo features (-0.73pp) but +2.1pp better in the final feature set. Peer fill was NOT used for validation (rule: validation features only for adversarial validation and December fill). |
| 15 | No month/day-of-year features; day-of-week dropped from the final trees | CV MAPE 3.14 -> 3.35 with month/doy; dow: 3.04 without vs 3.14 with (G1 -dow). The weekday pattern in December comes from market_index, which has a strong weekly cycle (val means Mon 0.85 ... Thu 1.03 ... Sun 0.84; same in train) |
| 16 | Features kept: base + market_index + geo coordinates + lane target stats (time-respecting). Lane stats are used only where the lane has history, otherwise a no-lane-stat fallback model (RoutedModel) | time-CV MAPE C1 3.04 -> C4 2.28; but in a cold-start CV (8 cities removed from training) lane stats hurt (3.13 vs 2.75 with geo only), so unseen-city rows get the fallback |
| 17 | Selection rule was REVISED after seeing results (disclosed). Original: lowest mean+std CV MAPE with holdout within 0.10pp -> blend (2.844 vs hybrid 2.884). Revised: blend must also not lose on the Sep-Oct holdout and latest CV month. Result: HYBRID (holdout MAPE 2.696 vs blend 2.762; last CV fold 1.608 vs 1.653) | blend weights are fitted on the same CV predictions that score it (optimistic) and its 0.04 advantage is smaller than the fold std (0.7). The hybrid is also a single model with a linear part that extrapolates in market_index. |
| 18 | Kept trees able to see market_index in the hybrid | follow-up test (hybrid_tree_feature_check.csv): blind trees are worse in CV (2.25 vs 2.18 MAPE, last fold 1.99 vs 1.61). Known side-effect: December response to market_index (log-slope ~0.07, non-monotone) is about half the empirical 0.12-0.13 slope. |
| 19 | December: option A (December validation daily means) chosen over B (frozen 2025-11-30) | same-lane+equipment rows in December validation: 3 in total, so daily means over all December rows are used. A differs from B by mean 4.33 / max 11.10 dollars; A reproduces the weekly cycle the market series really has, B is flat by construction. |
| 20 | Final predictions come from the refit hybrid on all 47,323 clean rows (677 label outliers removed) | |
