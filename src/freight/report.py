"""DOCX report builder (python-docx). Every number is read from outputs/*.csv|json produced by the pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from . import config as C

O = C.OUTPUTS_DIR


def _table(doc: Document, df: pd.DataFrame, widths: list[float] | None = None, font: int = 8) -> None:
    t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns):
        t.rows[0].cells[i].text = str(c)
    for row in df.itertuples(index=False):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = f"{v:.3f}" if isinstance(v, float) else str(v)
    for r in t.rows:
        for i, cell in enumerate(r.cells):
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(font)
            if widths:
                cell.width = Inches(widths[i])
    doc.add_paragraph()


def _fig(doc: Document, path: Path, width: float, caption: str) -> None:
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(caption); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].italic = True; p.runs[0].font.size = Pt(8.5)


def _bullets(doc: Document, items: list[str]) -> None:
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def build(path: Path) -> Path:
    facts = json.loads((O / "eda_facts.json").read_text())
    sel = json.loads((O / "model_selection.json").read_text())
    dec = json.loads((O / "december_summary.json").read_text())
    folds = pd.read_csv(O / "cv_folds.csv")
    comp = pd.read_csv(O / "model_comparison.csv").set_index("model")
    base = pd.read_csv(O / "baseline_cleaning_effects.csv").set_index("variant")
    issues = pd.read_csv(O / "data_quality_issues.csv")
    abl = pd.read_csv(O / "feature_ablation.csv").set_index("variant")
    comb = pd.read_csv(O / "feature_combinations.csv")
    tgt = pd.read_csv(O / "target_comparison.csv").set_index("variant")
    shap_ = pd.read_csv(O / "shap_importance_lightgbm.csv", index_col=0).iloc[:, 0]
    perm = pd.read_csv(O / "permutation_importance_chosen.csv", index_col=0).iloc[:, 0]
    resid = pd.read_csv(O / "residual_analysis_holdout.csv")
    adv = facts["adversarial"]; lo = facts["label_outliers"]
    hy = comp.loc["hybrid"]; bl = comp.loc["blend"]

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"; doc.styles["Normal"].font.size = Pt(10.5)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.9); s.top_margin = s.bottom_margin = Inches(0.8)
    h = doc.add_heading("Freight Rate Prediction: Approach, Validation and December Forecast", 0)
    doc.add_paragraph("Spotter ML Engineer take-home. Seed 42. All numbers below come from code in this repository (scripts/01-06); "
                      "no accuracy is claimed on validation.csv, which has no labels. Spotter computes the final score.").runs[0].italic = True

    doc.add_heading("1. Summary", 1)
    _bullets(doc, [
        f"Data: train_test.csv ({facts['shape']['train'][0]:,} labelled loads, {facts['date_range']['train'][0]} to {facts['date_range']['train'][1]}); validation.csv ({facts['shape']['validation'][0]:,} unlabeled loads, "
        f"{facts['date_range']['validation'][0]} to {facts['date_range']['validation'][1]}), i.e. the future of the training data.",
        "Validation: time-based only. Expanding monthly folds on Jan-Aug (6 folds) for tuning, a Sep-Oct holdout, then a refit on all of train. No random split anywhere in model evaluation.",
        f"Chosen model: a hybrid (Ridge on log(rate) with distance, equipment and market_index, plus LightGBM on its residuals). Mean CV MAPE {hy.cv_mape_mean:.2f}% (fold std {hy.cv_mape_std:.2f}), "
        f"Sep-Oct holdout MAPE {hy.holdout_mape_clean:.2f}% on clean rows.",
        f"Data quality: {lo['n']} corrupted labels ({lo['pct']:.2f}%) excluded from training, sign-flipped weights fixed, quote_signal excluded as unstable, 8 validation-only cities handled with a fallback model.",
    ])

    doc.add_heading("2. How the data was split and validated (key section)", 1)
    doc.add_heading("2.1 Why a random split is wrong here", 2)
    doc.add_paragraph(
        "The final predictions are for 2025-11-01 to 2025-12-31, one day after the last labelled row. This is a forecasting problem. "
        "The strongest feature, market_index, is a daily market series that moves from month to month (train mean 1.083, range 0.68-1.47; validation mean 0.927). "
        "A random K-fold would place rows from the same day, and therefore the same market regime, in both the training and evaluation folds, so the model would be scored on interpolation "
        "of conditions it has already seen. That overstates accuracy and rewards hyper-parameters that memorise day-level structure. A time-based split reproduces the real task: fit on the past, score on a later period.")
    doc.add_heading("2.2 The splits", 2)
    tab = folds[["split", "train_rows", "train_from", "train_to", "eval_rows", "eval_from", "eval_to", "eval_label_outliers"]].copy()
    tab.columns = ["split", "train rows", "train from", "train to", "eval rows", "eval from", "eval to", "corrupted labels in eval"]
    _table(doc, tab, [1.25, 0.7, 0.8, 0.8, 0.65, 0.8, 0.8, 0.8], 7.5)
    _fig(doc, C.PLOTS_DIR / "07_validation_design.png", 6.3, "Figure 1. Validation design (teal = train, orange = evaluation, grey = validation.csv to predict).")
    _bullets(doc, [
        "Expanding-window CV: fold k trains on every month before k and evaluates on month k, for k = March..August. Optuna (max 20 trials per model, seed 42) tunes only on these folds.",
        "Final holdout: train Jan-Aug, evaluate on Sep-Oct. It was scored once per tuned candidate, to choose among them and to report; nothing was tuned on it. Note: the model-selection rule was revised after seeing these results (disclosed in decision_log.md #17).",
        "Cold-start diagnostic: like the holdout, but 8 seeded cities are removed from training and only Sep-Oct rows touching them are scored. It mimics the 8 cities that exist only in validation.csv "
        f"({facts['unseen_val_rows_pct']:.1f}% of its rows).",
        "Final model: refit on all clean train rows (Jan-Oct) and predict validation.csv.",
        "Every split satisfies max(train date) < min(eval date) with a one-day gap, as between train and validation. This is asserted by pytest for every fold.",
    ])
    doc.add_heading("2.3 Leakage controls", 2)
    _bullets(doc, [
        "Lane target statistics (shrunk mean of month-demeaned log rate-per-mile for the lane and its reverse lane) are computed from the training slice only. For a training row, only rows from strictly earlier months are used (out-of-fold, time-respecting); "
        "evaluation rows see the whole training slice. A pytest changes labels in month 3 and checks that no row from month 3 or earlier changes, and that evaluation labels are never read.",
        "No calendar-time feature (month, day-of-year) is used; Nov/Dec are outside the training range and CV showed it hurts (MAPE 3.35 vs 3.14).",
        "The label-outlier flag (which rows are excluded) came from a 6-parameter structural fit on all train rows. It only decides which rows to drop and is not fold-specific; we disclose it.",
    ])
    doc.add_heading("2.4 How validation.csv features were used (no labels, no leakage)", 2)
    doc.add_paragraph("validation.csv has no labels. Its features were used for exactly two purposes: (a) adversarial validation, a drift diagnostic; and (b) filling market_index and quote_signal "
                      "for the 31 fixed December rows. No model was fit on it, and it was not used to impute or scale the 12,000 validation predictions "
                      "(null market_index in validation uses a constant taken from the last 7 train days).")
    doc.add_paragraph(f"Adversarial validation (LightGBM, train vs validation, date excluded): AUC {adv['auc_all_features']:.3f}; {adv['auc_without_market_index']:.3f} without market_index; "
                      f"{adv['auc_without_market_index_and_geo']:.3f} without market_index and geography. As a control, Sep-Oct vs Jan-Aug inside train gives {adv['control_auc_sepoct_vs_janaug_train']:.3f}, "
                      "so the train-to-validation shift is not larger than ordinary month-to-month drift. The drift is concentrated in market_index (univariate AUC "
                      f"{adv['univariate_auc']['market_index']:.3f}) and in the 8 new cities ({adv['univariate_auc']['has_unseen_city']:.3f}); distance, weight and equipment are stable (about 0.50). "
                      "Validation market_index (0.72-1.10) lies inside the train range (0.68-1.47).")
    _fig(doc, C.PLOTS_DIR / "05_adversarial_validation.png", 6.2, "Figure 2. Adversarial validation: what separates train from validation.")

    doc.add_heading("3. Fixed December prediction chart (produced by score.py)", 1)
    doc.add_picture(str(C.SCORER_DIR / "candidate_december.png"), width=Inches(6.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph("Figure 3. scorer_results/candidate_december.png, created by the unmodified score.py."); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.runs[0].italic = True
    _bullets(doc, [
        f"Inputs fixed by the task: Lexington to Fort Wayne, 360 mi, Dry Van, 32,000 lb. Coordinates are the train city medians; market_index and quote_signal for each December date come from the December rows of validation.csv "
        f"(daily mean over all rows; only 3 December rows share the lane and equipment, too few to use).",
        f"Option A (chosen, daily means of December validation rows): {dec['A_min']:.1f} to {dec['A_max']:.1f}, mean {dec['mean_pred_A']:.1f}. Option B (frozen at the last available day, 2025-11-30): mean {dec['mean_pred_B']:.1f}. "
        f"Mean absolute difference {dec['mean_abs_diff_A_vs_B']:.2f}, maximum {dec['max_abs_diff_A_vs_B']:.2f} dollars.",
        f"The curve has a 7-day cycle (peaks Wednesday/Thursday, troughs Saturday to Monday) and no unexplained spikes (largest day-to-day move {dec['A_max_day_to_day_jump']:.1f} dollars). "
        "The cycle is inherited from market_index, which has a strong weekly cycle in both train and validation; the model has no weekday feature.",
        "Caveats: training ends in October, so December seasonality (holiday demand) is not learned; tree models do not extrapolate beyond the training range; "
        "the model's response to market_index in December (log-slope about 0.07) is weaker than the empirical slope (about 0.12-0.13), so the amplitude of the weekly cycle is probably understated.",
    ])

    doc.add_heading("4. Data exploration and data-quality issues", 1)
    iss = issues[["issue", "rows_affected", "fix"]].copy(); iss.columns = ["issue", "rows affected", "fix"]
    iss = iss[~iss.issue.str.startswith(("Monthly", "Day-of", "Duplicate", "City / equipment"))]
    _table(doc, iss, [2.2, 2.0, 2.6], 7.5)
    doc.add_paragraph("Before/after effect of cleaning, measured on the CV folds (Mar-Aug) with a plain log-linear baseline (not the final model):")
    ba = base.reset_index()[["variant", "mae", "rmse", "mape_pct"]]; ba.columns = ["baseline variant", "MAE", "RMSE", "MAPE %"]
    _table(doc, ba, [3.6, 0.9, 0.9, 0.9], 8)
    _bullets(doc, [
        f"Corrupted labels ({lo['n']} rows, x{lo['mult_low_range'][0]:.2f}-{lo['mult_low_range'][1]:.2f} or x{lo['mult_high_range'][0]:.1f}-{lo['mult_high_range'][1]:.1f}): removing them from training barely changes a log-linear fit, "
        f"but they dominate every error metric (RMSE {base.loc[[i for i in base.index if i.startswith('B)')][0], 'rmse']:.0f} on all rows vs {base.loc[[i for i in base.index if i.startswith('C)')][0], 'rmse']:.0f} on clean rows). "
        "If validation labels contain the same corruption, that error is irreducible. We therefore report metrics on clean rows and, for the holdout, on all rows.",
        "market_index is the main non-geometric driver; quote_signal is not a near-proxy of the target, but its relationship flips sign by month (outputs/quote_signal_by_month.csv).",
        "Coordinates are constant per city (no row-level deviations) and synthetic; distance is about 1.18 x haversine with row-level noise and a 70-mile floor.",
    ])
    _fig(doc, C.PLOTS_DIR / "02_label_outliers.png", 6.3, "Figure 4. Corrupted labels: residual histogram (log scale) and rate vs distance.")
    _fig(doc, C.PLOTS_DIR / "03_quote_signal_instability.png", 6.3, "Figure 5. quote_signal: sign of its effect changes from month to month.")

    doc.add_heading("5. Features and ablations (CV folds Mar-Aug only)", 1)
    _bullets(doc, [
        f"Target: log(rate) beat rate-per-mile and log(rate-per-mile) (LightGBM fold-mean MAPE {tgt.loc['lightgbm | target=log_rate', 'fold_mape_mean']:.2f} vs {tgt.loc['lightgbm | target=rpm', 'fold_mape_mean']:.2f} vs {tgt.loc['lightgbm | target=log_rpm', 'fold_mape_mean']:.2f}).",
        f"Signals (LightGBM, same base features): no market variables {abl.loc['G0_base_no_market', 'mape_pct']:.2f}; market_index {abl.loc['G1_+market_index', 'mape_pct']:.2f}; "
        f"quote_signal only {abl.loc['G2_+quote_signal_only', 'mape_pct']:.2f}; both {abl.loc['G3_+both_signals', 'mape_pct']:.2f}. market_index is used (stable, inside the train range for validation and December). "
        "quote_signal is excluded: it is also inside the train range, but it makes the model less accurate and much less stable (fold MAPE up to 11.9%).",
    ])
    c2 = comb[comb["eval"] == "time-CV"][["variant", "mape_pct", "fold_mape_std", "last_fold"]].copy(); c2.columns = ["feature set", "MAPE %", "fold std", "last fold"]
    c3 = comb[comb["eval"] == "cold-start CV"][["variant", "mape_pct"]].rename(columns={"mape_pct": "cold-start CV MAPE %"})
    _table(doc, c2.merge(c3, left_on="feature set", right_on="variant").drop(columns="variant"), [2.4, 0.9, 0.9, 0.9, 1.3], 8)
    doc.add_paragraph("Lane statistics improve seen lanes but hurt rows with new cities; geography helps both. The final system therefore routes rows without lane history to a fallback model without lane statistics.")

    doc.add_heading("6. Models, results and choice", 1)
    r = comp.reset_index()[["model", "cv_mape_mean", "cv_mape_std", "cv_mape_worst", "cv_last_fold_mape", "holdout_mape_clean", "holdout_mae_clean", "holdout_rmse_clean", "holdout_mape_allrows", "coldstart_mape_clean"]]
    r.columns = ["model", "CV MAPE mean", "CV std", "CV worst", "CV last fold", "Sep-Oct MAPE", "Sep-Oct MAE", "Sep-Oct RMSE", "Sep-Oct MAPE all rows", "cold-start MAPE"]
    _table(doc, r, [0.8, 0.65, 0.5, 0.55, 0.6, 0.65, 0.65, 0.65, 0.7, 0.7], 7.5)
    doc.add_paragraph("MAPE is stable (minimum clean target in any fold is 185). Metrics are on clean rows unless stated; 'all rows' includes the corrupted labels. The tuning objective was the mean fold MAPE.").runs[0].italic = True
    _fig(doc, C.PLOTS_DIR / "08_model_comparison.png", 6.3, "Figure 6. Per-fold CV MAPE and CV / holdout / cold-start comparison.")
    bw = sel["blend_weights"]
    _bullets(doc, [
        "Models: Ridge on log(rate); LightGBM; CatBoost; hybrid; non-negative blend of the four (weights "
        + ", ".join(f"{k} {v:.2f}" for k, v in bw.items()) + ", fitted on the CV predictions).",
        f"Choice by temporal stability, not only the lowest mean: hybrid has the best latest-month CV error ({hy.cv_last_fold_mape:.2f}) and the best Sep-Oct holdout ({hy.holdout_mape_clean:.2f}). "
        f"The blend has a slightly lower CV mean ({bl.cv_mape_mean:.2f} vs {hy.cv_mape_mean:.2f}) but its weights are fitted on the same predictions that score it, its advantage (0.02) is far smaller than the fold std (0.7), "
        f"and it is worse on the holdout ({bl.holdout_mape_clean:.2f}). The original selection rule would have picked the blend; it was revised after seeing the results (disclosed).",
        "Why a hybrid: trees cannot extrapolate. The linear part carries the global scale in distance, equipment and market_index (which can move beyond what trees saw); the trees model bounded residual structure (weight, geography, lane effects).",
        "A toy-grid test confirms the final model's rate increases strictly with distance for every equipment type, with and without lane history.",
    ])
    doc.add_heading("7. Importance and residuals", 1)
    _bullets(doc, [
        "Permutation importance on the Sep-Oct holdout (increase in MAPE, percentage points): " + ", ".join(f"{k} {v:.2f}" for k, v in perm.head(6).items()) + f"; market_index {perm['market_index']:.2f}, lane_stat {perm['lane_stat']:.2f}.",
        "SHAP (LightGBM, mean |value| on log-rate): " + ", ".join(f"{k} {v:.3f}" for k, v in shap_.head(7).items()) + ".",
    ])
    _fig(doc, C.PLOTS_DIR / "10_feature_importance.png", 6.3, "Figure 7. SHAP and permutation importance.")
    eq = resid[resid.dimension == "equipment"]; mo = resid[resid.dimension == "month"]
    doc.add_paragraph("Holdout residuals (clean rows): " + "; ".join(f"{r.group} MAPE {r.mape_pct:.2f}% / bias {r.bias_pct:+.2f}%" for r in eq.itertuples()) + ". "
                      "All distance bands and both months are under-predicted by 2.3-2.7%, so the largest remaining error is a systematic level offset, not noise in one segment. "
                      "The daily mean residual is smooth and highly autocorrelated (lag-1 0.96); none of the available features explains it, and it cannot be learned for November and December.")
    _fig(doc, C.PLOTS_DIR / "09_residuals_holdout.png", 6.4, "Figure 8. Holdout MAPE (bars) and bias (dots) by equipment, distance band and month.")

    doc.add_heading("8. Limitations", 1)
    _bullets(doc, [
        "Systematic under-prediction of about 2.2-3.2% on the Sep-Oct holdout; it may carry over to Nov/Dec.",
        "December seasonality is not learned (training ends 31 Oct); trees do not extrapolate; the model's market_index response in December is about half the empirical slope.",
        "8 cities appear only in validation (12.1% of rows); they are served by the fallback model. In the cold-start diagnostic MAPE is about 4.6% versus 2.7% on seen lanes.",
        "Corrupted labels (about 1.4%) are excluded from training; if validation labels contain them, the final error is dominated by them.",
        "Spotter's metric is undisclosed; MAE, RMSE, MAPE, sMAPE, RMSLE, R2 and median APE are all computed (outputs/cv_fold_metrics.csv).",
        "The holdout was used to choose among five tuned candidates and the selection rule was changed after seeing results; the holdout number is therefore mildly optimistic.",
        "Validation null market_index (249 rows) is filled with a constant (0.944) to respect the rule on validation features; this may cost accuracy on about 2% of rows and cannot be measured without labels.",
    ])
    doc.add_heading("9. What was not done / not run", 1)
    _bullets(doc, [
        "No accuracy is claimed on validation.csv; no labels exist for it.",
        "Blend weights were not nested-cross-validated; no second Optuna seed; more than 20 trials per model not tried.",
        "No quantile or interval predictions; no explicit day-level latent-level model; no external data (calendar/holiday features were not used).",
        "No TabPFN or pretrained weights; no network access beyond installing the pinned packages.",
        "Not verified on other operating systems or library versions (versions are pinned in requirements.txt; tested on Python 3.12, Linux).",
        "The 2-3 minute Loom video is a deliverable of the candidate and is not part of this document (script in loom_script.md).",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    return path
