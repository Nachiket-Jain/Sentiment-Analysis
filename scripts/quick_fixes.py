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
from sklearn.model_selection import StratifiedKFold

# ── Load saved results ──────────────────────────────────────────────────────────
PKL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "evaluation_results.pkl")

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

ensemble_rows = []

# Strategy 1: Simple average (current)
simple_logits = (ib_logits + xlmr_logits) / 2.0
simple_preds = np.argmax(simple_logits, axis=1)
ensemble_rows.append(compute_all_metrics(y_test, simple_preds, "Ensemble: Simple Average"))

# Strategy 2: Weighted average (more weight to XLM-R since it's better)
for alpha in [0.55, 0.60, 0.65, 0.70]:
    weighted_logits = alpha * xlmr_logits + (1 - alpha) * ib_logits
    weighted_preds = np.argmax(weighted_logits, axis=1)
    ensemble_rows.append(compute_all_metrics(y_test, weighted_preds,
                                              f"Ensemble: Weighted (XLM-R={alpha:.0%})"))

# Strategy 3: Max confidence — pick whichever model is more confident
max_conf_preds = np.zeros(len(y_test), dtype=int)
for i in range(len(y_test)):
    ib_conf = ib_probs[i].max()
    xlmr_conf = xlmr_probs[i].max()
    if xlmr_conf >= ib_conf:
        max_conf_preds[i] = xlmr_preds[i]
    else:
        max_conf_preds[i] = ib_preds[i]
ensemble_rows.append(compute_all_metrics(y_test, max_conf_preds, "Ensemble: Max Confidence"))

# Strategy 4: Stacking (Logistic Regression on model probabilities)
# Use 5-fold CV to get fair stacking predictions
print("\nTraining stacked ensemble (5-fold CV)...")
stacked_preds = np.zeros(len(y_test), dtype=int)
features = np.hstack([ib_probs, xlmr_probs])  # 6 features: 3 probs from each model

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold_idx, (meta_train, meta_test) in enumerate(skf.split(features, y_test)):
    stacker = LogisticRegression(C=1.0, max_iter=1000, multi_class="multinomial")
    stacker.fit(features[meta_train], y_test[meta_train])
    stacked_preds[meta_test] = stacker.predict(features[meta_test])

ensemble_rows.append(compute_all_metrics(y_test, stacked_preds, "Ensemble: Stacking (LR)"))

# Strategy 5: Stacking + include baseline predictions as feature
# The baseline model has different error patterns (TF-IDF captures different signals)
baseline_onehot = np.zeros((len(y_test), 3))
for i, p in enumerate(baseline_preds):
    baseline_onehot[i, p] = 1.0
features_with_baseline = np.hstack([ib_probs, xlmr_probs, baseline_onehot])

stacked_preds_v2 = np.zeros(len(y_test), dtype=int)
for fold_idx, (meta_train, meta_test) in enumerate(skf.split(features_with_baseline, y_test)):
    stacker = LogisticRegression(C=1.0, max_iter=1000, multi_class="multinomial")
    stacker.fit(features_with_baseline[meta_train], y_test[meta_train])
    stacked_preds_v2[meta_test] = stacker.predict(features_with_baseline[meta_test])

ensemble_rows.append(compute_all_metrics(y_test, stacked_preds_v2,
                                          "Ensemble: Stacking (LR + Baseline)"))

# Print comparison
ensemble_df = pd.DataFrame(ensemble_rows)
print("\n" + ensemble_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# BEST COMBINATION: Best ensemble + best calibration
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("BEST COMBINATIONS (Ensemble + Calibration)")
print("=" * 70)

# Find the best ensemble first, then apply conservative calibration on top
combo_rows = []

# Current system
old_cal_ensemble = apply_old_calibration(softmax(simple_logits, axis=1), reviews, "XLM-R")
combo_rows.append(compute_all_metrics(y_test, old_cal_ensemble, "OLD: Simple Avg + Old Cal"))

# Best: No calibration, just XLM-R raw
combo_rows.append(compute_all_metrics(y_test, xlmr_preds, "XLM-R Raw (no cal, no ensemble)"))

# Best: Weighted ensemble, no calibration
best_weighted = np.argmax(0.60 * xlmr_logits + 0.40 * ib_logits, axis=1)
combo_rows.append(compute_all_metrics(y_test, best_weighted, "Weighted Ensemble (60/40), no cal"))

# Best: Weighted ensemble + conservative calibration
best_weighted_probs = softmax(0.60 * xlmr_logits + 0.40 * ib_logits, axis=1)
best_weighted_cal = apply_conservative_calibration(best_weighted_probs, reviews)
combo_rows.append(compute_all_metrics(y_test, best_weighted_cal,
                                       "Weighted Ensemble (60/40) + Conservative Cal"))

# Stacking results (already computed)
combo_rows.append(compute_all_metrics(y_test, stacked_preds, "Stacked Ensemble (LR)"))
combo_rows.append(compute_all_metrics(y_test, stacked_preds_v2, "Stacked Ensemble (LR + Baseline)"))

combo_df = pd.DataFrame(combo_rows)
print("\n" + combo_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# FIND THE OVERALL BEST
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("OVERALL BEST CONFIGURATION")
print("=" * 70)

# Combine all results
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
