# Loom script (about 2:00-2:15 at a calm pace, ~300 words spoken)

**[0:00 Findings]** Hi, I'm [name]. We predict freight rates for November and December using January to October, so the validation set is the future. Three findings: market_index is a daily series and the strongest driver after distance; quote_signal looks tempting but its effect flips sign month to month; and eight cities exist only in validation, twelve percent of its rows.

**[0:30 Data quality]** 677 training labels, 1.4 percent, are scaled by roughly 0.2 to 0.5 or 2 to 5 times. I removed them from training and report metrics on clean rows and on all rows. 292 weights were negative, so I use the absolute value. Weights are capped at 47,500, and about one percent of weight and market_index is null. Coordinates were constant per city.

**[1:00 Split]** I never split randomly, because rows from the same day would land on both sides. Six expanding monthly folds, March to August, are for tuning only. Then I train on January to August and score once on September-October, and finally refit on everything. A pytest checks every fold's training dates end before its evaluation dates. From validation.csv I used features only, for adversarial validation and the December market inputs, never labels.

**[1:30 Model]** I tuned ridge, LightGBM, CatBoost, a hybrid and a blend, at most twenty Optuna trials each. The hybrid wins: a linear model carries distance, equipment and market_index, which trees can't extrapolate, and trees model the residuals. CV error is 2.18 percent, holdout 2.70. I chose it for stability: the blend's weights are fitted on the same predictions that score it.

**[2:00 Code]** Open src/freight/cv.py for the time splits, features.py for lane statistics from earlier months only, models.py for the hybrid, and scripts/05_final_predictions.py for predictions and December.

**[2:20 Limits]** The model under-predicts by two to three percent on the holdout, and December seasonality isn't learned. I claim no accuracy on validation. Thanks.

---
## Likely interview questions on my code
1. **Why not a random split?** Validation is the future. Random folds put the same day and market regime in train and test, so the score is optimistic and tuning overfits.
2. **How do you prevent leakage in the lane features?** For a training row only rows from strictly earlier months of the training slice are used; eval rows see the training slice only. A pytest perturbs month-3 labels and checks months <= 3 do not change.
3. **Why a hybrid?** Trees cannot extrapolate. The linear part carries scale in distance, equipment and market_index; trees fit bounded residual structure.
4. **Why drop quote_signal?** It hurt CV (3.97 vs 3.14 MAPE with both signals vs market_index only), fold errors up to ~12%, and its slope flips sign by month.
5. **How did you treat the 677 outliers?** Flagged by a robust residual (> 10 robust sigma), excluded from training and from clean metrics, but reported on all rows too.
6. **What about the new cities?** A routed fallback model without lane statistics handles lanes with no history; cold-start diagnostic: ~4.6% MAPE.
7. **Why did you change the model-selection rule?** The blend's weights are fitted on the CV predictions that score it; its edge (0.02) is below the fold std. I disclosed the change.
8. **What is the biggest weakness?** A smooth day-level offset (holdout bias about -2.2 to -3.2%) that no feature explains, so it can't be learned for Nov/Dec.
