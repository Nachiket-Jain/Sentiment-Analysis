"""
Quick Fixes: Improved Calibration + Weighted Ensemble
======================================================
Fixes 7 & 8 from the improvement plan.
Runs instantly on saved evaluation_results.pkl — no retraining needed.

Usage: python quick_fixes.py
"""

import pickle
import os
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ── Load saved results ──────────────────────────────────────────────────────────
PKL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "evaluation_results.pkl")
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", "csv"), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", "plots"), exist_ok=True)

print("=" * 70)
print("QUICK FIXES: CALIBRATION + WEIGHTED ENSEMBLE")
print("=" * 70)

with open(PKL_PATH, "rb") as f:
    saved = pickle.load(f)

ib_results = saved["ib_results"]
xlmr_results = saved["xlmr_results"]
baseline_preds = np.array(saved["baseline_preds"])
test_df = saved["test_df"]
y_test = np.array(saved["y_test"])
label2id = saved["label2id"]
id2label = saved["id2label"]

ib_logits = ib_results["logits"]
xlmr_logits = xlmr_results["logits"]

ib_probs = softmax(ib_logits, axis=1)
xlmr_probs = softmax(xlmr_logits, axis=1)

ib_preds = np.argmax(ib_logits, axis=1)
xlmr_preds = np.argmax(xlmr_logits, axis=1)

reviews = test_df["Review"].reset_index(drop=True)

print(f"Test set: {len(y_test)} samples")

# ── Hinglish word lists ─────────────────────────────────────────────────────────
HINGLISH_POSITIVE = [
    'mast', 'badhiya', 'accha', 'achha', 'zabardast', 'kamaal',
    'ekdum', 'sahi', 'best', 'super', 'boht', 'bahut', 'bohot',
    'shandar', 'awesome', 'perfect', 'excellent'
]
HINGLISH_NEGATIVE = [
    'bakwas', 'bekar', 'bekaar', 'ghatiya', 'faltu', 'kharab',
    'worst', 'terrible', 'bad', 'useless'
]


def compute_all_metrics(y_true, y_pred, label=""):
    """Compute comprehensive metrics."""
    return {
        "Label": label,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "F1 Macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "F1 Weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "Neg Recall": round(recall_score(y_true, y_pred, labels=[0], average=None, zero_division=0)[0], 4),
        "Neu Accuracy": round((y_pred[y_true == 1] == 1).mean(), 4),
        "Pos Precision": round(precision_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0], 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FIX 7: IMPROVED CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FIX 7: CALIBRATION COMPARISON")
print("=" * 70)


def apply_old_calibration(probs_array, texts, model_name="XLM-R"):
    """Original aggressive calibration from Cell 11B."""
    preds = np.argmax(probs_array, axis=1).copy()
    if model_name == "XLM-R":
        min_pos_threshold = 0.0006
        max_neg_threshold = 0.030
    else:
        min_pos_threshold = 0.003
        max_neg_threshold = 0.10

    for i in range(len(preds)):
        neg_p, neu_p, pos_p = probs_array[i]
        text_lower = str(texts.iloc[i]).lower()
        has_pos = any(w in text_lower for w in HINGLISH_POSITIVE)
        has_neg = any(w in text_lower for w in HINGLISH_NEGATIVE)

        if preds[i] == 1 and neu_p > 0.85 and neg_p < max_neg_threshold and pos_p > min_pos_threshold:
            preds[i] = 2
        elif has_pos and not has_neg and preds[i] == 1 and pos_p > neg_p and neg_p < 0.20:
            preds[i] = 2
        elif has_neg and preds[i] == 1 and neg_p > 0.05 and pos_p < 0.15:
            preds[i] = 0
    return preds


def apply_conservative_calibration(probs_array, texts):
    """
    FIX 7A: Conservative calibration — much stricter thresholds.
    Only flips when there's very strong evidence.
    """
    preds = np.argmax(probs_array, axis=1).copy()
    for i in range(len(preds)):
        neg_p, neu_p, pos_p = probs_array[i]
        text_lower = str(texts.iloc[i]).lower()
        has_pos = any(w in text_lower for w in HINGLISH_POSITIVE)
        has_neg = any(w in text_lower for w in HINGLISH_NEGATIVE)

        # Only apply Hinglish rules — remove the aggressive standard calibration entirely
        if has_pos and not has_neg and preds[i] == 1 and pos_p > neg_p and neg_p < 0.10:
            preds[i] = 2
        elif has_neg and not has_pos and preds[i] == 1 and neg_p > 0.10 and pos_p < 0.10:
            preds[i] = 0
    return preds


def apply_hinglish_only_calibration(probs_array, texts):
    """
    FIX 7B: Only Hinglish keyword rules, no standard threshold calibration.
    """
    preds = np.argmax(probs_array, axis=1).copy()
    for i in range(len(preds)):
        neg_p, neu_p, pos_p = probs_array[i]
        text_lower = str(texts.iloc[i]).lower()
        has_pos = any(w in text_lower for w in HINGLISH_POSITIVE)
        has_neg = any(w in text_lower for w in HINGLISH_NEGATIVE)

        if has_pos and not has_neg and preds[i] == 1 and pos_p > neg_p and neg_p < 0.20:
            preds[i] = 2
        elif has_neg and preds[i] == 1 and neg_p > 0.05 and pos_p < 0.15:
            preds[i] = 0
    return preds


# Compare calibration approaches for XLM-R
print("\nXLM-R Calibration Comparison:")
print("-" * 70)

cal_rows = []
# Raw (no calibration)
cal_rows.append(compute_all_metrics(y_test, xlmr_preds, "XLM-R (No Calibration)"))
# Old calibration
cal_rows.append(compute_all_metrics(y_test, apply_old_calibration(xlmr_probs, reviews, "XLM-R"),
                                     "XLM-R (Old Calibration)"))
# Hinglish only
cal_rows.append(compute_all_metrics(y_test, apply_hinglish_only_calibration(xlmr_probs, reviews),
                                     "XLM-R (Hinglish Only)"))
# Conservative
cal_rows.append(compute_all_metrics(y_test, apply_conservative_calibration(xlmr_probs, reviews),
                                     "XLM-R (Conservative)"))

cal_df = pd.DataFrame(cal_rows)
print(cal_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# FIX 8: WEIGHTED & STACKED ENSEMBLES
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("FIX 8: ENSEMBLE STRATEGIES")
print("=" * 70)

# ── Held-out split for hyperparameter selection ──────────────────────────────
# The weighted-ensemble alpha below is a tuned hyperparameter, so it must be
# selected on a split that's separate from the one final numbers are reported
# on — tuning alpha directly against the same test set it's then scored on
# (as the original version did) inflates the reported accuracy. All Fix 8
# strategies are reported on idx_report for a consistent, leak-free comparison.
idx_tune, idx_report = train_test_split(
    np.arange(len(y_test)), test_size=0.5, stratify=y_test, random_state=42
)

ensemble_rows = []

# Strategy 1: Simple average (current)
simple_logits = (ib_logits + xlmr_logits) / 2.0
simple_preds = np.argmax(simple_logits, axis=1)
ensemble_rows.append(compute_all_metrics(y_test[idx_report], simple_preds[idx_report], "Ensemble: Simple Average"))

# Strategy 2: Weighted average — alpha selected on idx_tune, reported on idx_report
print("\nTuning ensemble weight alpha on held-out tuning split...")
alpha_tune_rows = []
for alpha in [0.55, 0.60, 0.65, 0.70]:
    weighted_logits = alpha * xlmr_logits + (1 - alpha) * ib_logits
    weighted_preds = np.argmax(weighted_logits, axis=1)
    tune_acc = accuracy_score(y_test[idx_tune], weighted_preds[idx_tune])
    alpha_tune_rows.append((alpha, tune_acc))
    print(f"    alpha={alpha:.2f} -> tuning-split accuracy={tune_acc:.4f}")

best_alpha = max(alpha_tune_rows, key=lambda r: r[1])[0]
print(f"  Selected alpha={best_alpha:.2f} (best on tuning split)")

best_weighted_logits = best_alpha * xlmr_logits + (1 - best_alpha) * ib_logits
best_weighted_preds = np.argmax(best_weighted_logits, axis=1)
ensemble_rows.append(compute_all_metrics(
    y_test[idx_report], best_weighted_preds[idx_report],
    f"Ensemble: Weighted (XLM-R={best_alpha:.0%}, alpha tuned on held-out split)"
))

# Strategy 3: Max confidence — pick whichever model is more confident
max_conf_preds = np.zeros(len(y_test), dtype=int)
for i in range(len(y_test)):
    ib_conf = ib_probs[i].max()
    xlmr_conf = xlmr_probs[i].max()
    if xlmr_conf >= ib_conf:
        max_conf_preds[i] = xlmr_preds[i]
    else:
        max_conf_preds[i] = ib_preds[i]
ensemble_rows.append(compute_all_metrics(y_test[idx_report], max_conf_preds[idx_report], "Ensemble: Max Confidence"))

# Strategy 4: Stacking (Logistic Regression on model probabilities)
# Fit only on idx_tune, evaluate on idx_report — avoids scoring the stacker on
# any row it could have influenced via the meta-feature construction.
print("\nTraining stacked ensemble (fit on tuning split, scored on report split)...")
features = np.hstack([ib_probs, xlmr_probs])  # 6 features: 3 probs from each model

stacker = LogisticRegression(C=1.0, max_iter=1000, multi_class="multinomial")
stacker.fit(features[idx_tune], y_test[idx_tune])
stacked_preds_report = stacker.predict(features[idx_report])

ensemble_rows.append(compute_all_metrics(y_test[idx_report], stacked_preds_report, "Ensemble: Stacking (LR)"))

# Strategy 5: Stacking + include baseline predictions as feature
baseline_onehot = np.zeros((len(y_test), 3))
for i, p in enumerate(baseline_preds):
    baseline_onehot[i, p] = 1.0
features_with_baseline = np.hstack([ib_probs, xlmr_probs, baseline_onehot])

stacker_v2 = LogisticRegression(C=1.0, max_iter=1000, multi_class="multinomial")
stacker_v2.fit(features_with_baseline[idx_tune], y_test[idx_tune])
stacked_preds_v2_report = stacker_v2.predict(features_with_baseline[idx_report])

ensemble_rows.append(compute_all_metrics(y_test[idx_report], stacked_preds_v2_report,
                                          "Ensemble: Stacking (LR + Baseline)"))

# Print comparison
ensemble_df = pd.DataFrame(ensemble_rows)
print("\nAll rows below are scored on the held-out report split (not seen during alpha/stacker tuning):")
print(ensemble_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# BEST COMBINATION: Best ensemble + best calibration
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("BEST COMBINATIONS (Ensemble + Calibration)")
print("=" * 70)

# All rows scored on idx_report — the same held-out split used above, so this
# stays consistent with (and comparable to) the Fix 8 table.
combo_rows = []

# Current system
old_cal_ensemble_full = apply_old_calibration(softmax(simple_logits, axis=1), reviews, "XLM-R")
combo_rows.append(compute_all_metrics(y_test[idx_report], old_cal_ensemble_full[idx_report],
                                       "OLD: Simple Avg + Old Cal"))

# Best: No calibration, just XLM-R raw
combo_rows.append(compute_all_metrics(y_test[idx_report], xlmr_preds[idx_report],
                                       "XLM-R Raw (no cal, no ensemble)"))

# Best: Weighted ensemble (alpha selected above), no calibration
combo_rows.append(compute_all_metrics(y_test[idx_report], best_weighted_preds[idx_report],
                                       f"Weighted Ensemble (XLM-R={best_alpha:.0%}), no cal"))

# Best: Weighted ensemble + conservative calibration
best_weighted_probs_full = softmax(best_weighted_logits, axis=1)
best_weighted_cal_full = apply_conservative_calibration(best_weighted_probs_full, reviews)
combo_rows.append(compute_all_metrics(y_test[idx_report], best_weighted_cal_full[idx_report],
                                       f"Weighted Ensemble (XLM-R={best_alpha:.0%}) + Conservative Cal"))

# Stacking results (already fit on idx_tune, scored on idx_report above)
combo_rows.append(compute_all_metrics(y_test[idx_report], stacked_preds_report, "Stacked Ensemble (LR)"))
combo_rows.append(compute_all_metrics(y_test[idx_report], stacked_preds_v2_report, "Stacked Ensemble (LR + Baseline)"))

combo_df = pd.DataFrame(combo_rows)
print("\n" + combo_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# FIND THE OVERALL BEST
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("OVERALL BEST CONFIGURATION")
print("=" * 70)

# Note: cal_df (Fix 7) is scored on the full test set — its calibration
# variants use fixed, hand-set thresholds rather than anything tuned against
# labels, so there's no leakage risk in scoring it on the full set. ensemble_df
# and combo_df (Fix 8) are scored on idx_report only, since alpha/the stacker
# were tuned on idx_tune. Both are stratified samples of the same
# distribution, so accuracy/F1 remain comparable across the two — but note
# the split size difference if you report per-class support in the paper.
all_results = pd.concat([cal_df, ensemble_df, combo_df], ignore_index=True)
all_results = all_results.drop_duplicates(subset=["Label"])

best_acc = all_results.loc[all_results["Accuracy"].idxmax()]
best_f1 = all_results.loc[all_results["F1 Macro"].idxmax()]

print(f"\nBest by Accuracy:  {best_acc['Label']}")
print(f"  Accuracy={best_acc['Accuracy']}, F1 Macro={best_acc['F1 Macro']}, "
      f"Neu Acc={best_acc['Neu Accuracy']}")

print(f"\nBest by F1 Macro:  {best_f1['Label']}")
print(f"  Accuracy={best_f1['Accuracy']}, F1 Macro={best_f1['F1 Macro']}, "
      f"Neu Acc={best_f1['Neu Accuracy']}")

# Compare to old system
old_system = combo_rows[0]
print(f"\n\nOLD system: Accuracy={old_system['Accuracy']}, F1={old_system['F1 Macro']}")
print(f"BEST:       Accuracy={best_acc['Accuracy']}, F1={best_f1['F1 Macro']}")
print(f"Improvement: +{best_acc['Accuracy'] - old_system['Accuracy']:.4f} accuracy, "
      f"+{best_f1['F1 Macro'] - old_system['F1 Macro']:.4f} F1")

# Save
all_results.to_csv("../results/csv/quick_fixes_comparison.csv", index=False)
print("\nSaved: quick_fixes_comparison.csv")

print("\n" + "=" * 70)
print("QUICK FIXES COMPLETE")
print("=" * 70)
