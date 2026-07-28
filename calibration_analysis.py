"""
Task 2: Calibration Quantitative Evaluation
=============================================
- Before vs After calibration comparison table
- Calibration metrics: ECE, MCE, Brier Score
- Confidence histograms before/after calibration

Loads from evaluation_results.pkl — no retraining needed.

Usage: python calibration_analysis.py
"""

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, brier_score_loss
)
from scipy.special import softmax

# ── Load saved results ──────────────────────────────────────────────────────────
PKL_PATH = os.path.join(os.path.dirname(__file__), "evaluation_results.pkl")

print("=" * 70)
print("TASK 2: CALIBRATION QUANTITATIVE EVALUATION")
print("=" * 70)

print("\nLoading evaluation results...")
with open(PKL_PATH, "rb") as f:
    saved = pickle.load(f)

ib_results = saved["ib_results"]
xlmr_results = saved["xlmr_results"]
test_df = saved["test_df"]
y_test = np.array(saved["y_test"])
label2id = saved["label2id"]
id2label = saved["id2label"]

# Get logits and convert to probabilities
ib_logits = ib_results["logits"]
xlmr_logits = xlmr_results["logits"]
ensemble_logits = (ib_logits + xlmr_logits) / 2.0

ib_probs = softmax(ib_logits, axis=1)
xlmr_probs = softmax(xlmr_logits, axis=1)
ensemble_probs = softmax(ensemble_logits, axis=1)

# Raw predictions (before calibration)
ib_preds_raw = np.argmax(ib_logits, axis=1)
xlmr_preds_raw = np.argmax(xlmr_logits, axis=1)
ensemble_preds_raw = np.argmax(ensemble_logits, axis=1)

print(f"  Test set size: {len(y_test)}")
print(f"  Classes: {id2label}")

# ── Calibration Rules (from Cell 11B in finetune+shap.ipynb) ─────────────────
HINGLISH_POSITIVE = [
    'mast', 'badhiya', 'accha', 'achha', 'zabardast', 'kamaal',
    'ekdum', 'sahi', 'best', 'super', 'boht', 'bahut', 'bohot',
    'shandar', 'awesome', 'perfect', 'excellent'
]

HINGLISH_NEGATIVE = [
    'bakwas', 'bekar', 'bekaar', 'ghatiya', 'faltu', 'kharab',
    'worst', 'terrible', 'bad', 'useless'
]


def apply_calibration(probs_array, texts, model_name="IndicBERT"):
    """
    Apply the calibration rules from Cell 11B to the full test set.
    Returns calibrated predictions and calibrated confidence scores.
    """
    n = len(probs_array)
    calibrated_preds = np.argmax(probs_array, axis=1).copy()
    calibration_flags = np.zeros(n, dtype=int)  # 0=no change, 1=standard, 2=hinglish+, 3=hinglish-

    # Model-specific thresholds
    if model_name == "XLM-R":
        min_pos_threshold = 0.0006
        max_neg_threshold = 0.030
    else:
        min_pos_threshold = 0.003
        max_neg_threshold = 0.10

    neutral_threshold = 0.85

    for i in range(n):
        neg_p, neu_p, pos_p = probs_array[i]
        pred = calibrated_preds[i]
        text_lower = str(texts.iloc[i]).lower() if hasattr(texts, 'iloc') else str(texts[i]).lower()

        has_hinglish_pos = any(w in text_lower for w in HINGLISH_POSITIVE)
        has_hinglish_neg = any(w in text_lower for w in HINGLISH_NEGATIVE)

        # Rule 1: Standard English calibration (neutral → positive)
        if (pred == 1 and  # neutral
                neu_p > neutral_threshold and
                neg_p < max_neg_threshold and
                pos_p > min_pos_threshold):
            calibrated_preds[i] = 2  # positive
            calibration_flags[i] = 1

        # Rule 2: Hinglish positive override
        elif (has_hinglish_pos and not has_hinglish_neg and
              pred == 1 and pos_p > neg_p and neg_p < 0.20):
            calibrated_preds[i] = 2  # positive
            calibration_flags[i] = 2

        # Rule 3: Hinglish negative override
        elif (has_hinglish_neg and pred == 1 and
              neg_p > 0.05 and pos_p < 0.15):
            calibrated_preds[i] = 0  # negative
            calibration_flags[i] = 3

    return calibrated_preds, calibration_flags


# Apply calibration to both models and ensemble
print("\nApplying calibration rules to full test set...")
reviews = test_df["Review"].reset_index(drop=True)

ib_preds_cal, ib_flags = apply_calibration(ib_probs, reviews, "IndicBERT")
xlmr_preds_cal, xlmr_flags = apply_calibration(xlmr_probs, reviews, "XLM-R")
ensemble_preds_cal, ens_flags = apply_calibration(ensemble_probs, reviews, "Ensemble")

print(f"  IndicBERT: {np.sum(ib_flags > 0)} samples calibrated "
      f"({np.sum(ib_flags == 1)} standard, {np.sum(ib_flags == 2)} hinglish+, {np.sum(ib_flags == 3)} hinglish-)")
print(f"  XLM-R:     {np.sum(xlmr_flags > 0)} samples calibrated "
      f"({np.sum(xlmr_flags == 1)} standard, {np.sum(xlmr_flags == 2)} hinglish+, {np.sum(xlmr_flags == 3)} hinglish-)")
print(f"  Ensemble:  {np.sum(ens_flags > 0)} samples calibrated "
      f"({np.sum(ens_flags == 1)} standard, {np.sum(ens_flags == 2)} hinglish+, {np.sum(ens_flags == 3)} hinglish-)")


# ── Section A: Before vs After Calibration Table ────────────────────────────────
print("\n\n" + "=" * 70)
print("A. BEFORE vs AFTER CALIBRATION — PERFORMANCE TABLE")
print("=" * 70)


def compute_metrics_dict(y_true, y_pred, label=""):
    """Compute the requested metrics: Neg Recall, Pos Precision, Accuracy"""
    return {
        "Label": label,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Neg Recall": round(recall_score(y_true, y_pred, labels=[0], average=None, zero_division=0)[0], 4),
        "Pos Precision": round(precision_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0], 4),
        "F1 Macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "F1 Weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


# Before calibration
rows = []
rows.append(compute_metrics_dict(y_test, ib_preds_raw, "IndicBERT (Before)"))
rows.append(compute_metrics_dict(y_test, ib_preds_cal, "IndicBERT (After)"))
rows.append(compute_metrics_dict(y_test, xlmr_preds_raw, "XLM-R (Before)"))
rows.append(compute_metrics_dict(y_test, xlmr_preds_cal, "XLM-R (After)"))
rows.append(compute_metrics_dict(y_test, ensemble_preds_raw, "Ensemble (Before)"))
rows.append(compute_metrics_dict(y_test, ensemble_preds_cal, "Ensemble (After)"))

calibration_table = pd.DataFrame(rows)
calibration_table = calibration_table[["Label", "Accuracy", "Neg Recall", "Pos Precision", "F1 Macro", "F1 Weighted"]]

print("\n" + calibration_table.to_string(index=False))

# ── Section B: Calibration Metrics (ECE, MCE, Brier) ────────────────────────────
print("\n\n" + "=" * 70)
print("B. CALIBRATION METRICS: ECE, MCE, BRIER SCORE")
print("=" * 70)


def compute_ece(probs, y_true, preds, n_bins=10):
    """
    Expected Calibration Error (ECE).
    Bins predictions by confidence, computes |accuracy - confidence| per bin.
    ECE = weighted average of per-bin calibration error.
    """
    confidences = np.max(probs, axis=1)
    accuracies = (preds == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    per_bin = []

    for b in range(n_bins):
        mask = (confidences > bin_boundaries[b]) & (confidences <= bin_boundaries[b + 1])
        if mask.sum() == 0:
            per_bin.append({"bin": f"({bin_boundaries[b]:.1f}, {bin_boundaries[b+1]:.1f}]",
                            "count": 0, "avg_conf": 0, "avg_acc": 0, "gap": 0})
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        gap = abs(avg_acc - avg_conf)
        weight = mask.sum() / len(y_true)
        ece += gap * weight

        per_bin.append({
            "bin": f"({bin_boundaries[b]:.1f}, {bin_boundaries[b+1]:.1f}]",
            "count": int(mask.sum()),
            "avg_conf": round(avg_conf, 4),
            "avg_acc": round(avg_acc, 4),
            "gap": round(gap, 4)
        })

    return round(ece, 4), per_bin


def compute_mce(probs, y_true, preds, n_bins=10):
    """Maximum Calibration Error — max gap across bins."""
    confidences = np.max(probs, axis=1)
    accuracies = (preds == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_gap = 0.0

    for b in range(n_bins):
        mask = (confidences > bin_boundaries[b]) & (confidences <= bin_boundaries[b + 1])
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        gap = abs(avg_acc - avg_conf)
        max_gap = max(max_gap, gap)

    return round(max_gap, 4)


def compute_brier_multiclass(probs, y_true, n_classes=3):
    """
    Multi-class Brier Score = (1/N) * sum_i sum_k (p_ik - y_ik)^2
    where y_ik is 1 if sample i belongs to class k, else 0.
    Lower is better. Range: [0, 2].
    """
    n = len(y_true)
    # One-hot encode true labels
    y_onehot = np.zeros((n, n_classes))
    y_onehot[np.arange(n), y_true] = 1

    brier = np.mean(np.sum((probs - y_onehot) ** 2, axis=1))
    return round(brier, 4)


# Compute calibration metrics for all models, before and after
cal_metrics_rows = []

for name, probs, preds_raw, preds_cal in [
    ("IndicBERT", ib_probs, ib_preds_raw, ib_preds_cal),
    ("XLM-R", xlmr_probs, xlmr_preds_raw, xlmr_preds_cal),
    ("Ensemble", ensemble_probs, ensemble_preds_raw, ensemble_preds_cal),
]:
    # Before calibration
    ece_before, _ = compute_ece(probs, y_test, preds_raw)
    mce_before = compute_mce(probs, y_test, preds_raw)
    brier_before = compute_brier_multiclass(probs, y_test)

    # After calibration — probabilities same, predictions change
    ece_after, _ = compute_ece(probs, y_test, preds_cal)
    mce_after = compute_mce(probs, y_test, preds_cal)
    # Brier score uses probabilities, which don't change with rule-based calibration
    brier_after = brier_before  # Same probabilities

    cal_metrics_rows.append({
        "Model": f"{name} (Before)",
        "ECE": ece_before,
        "MCE": mce_before,
        "Brier Score": brier_before,
    })
    cal_metrics_rows.append({
        "Model": f"{name} (After)",
        "ECE": ece_after,
        "MCE": mce_after,
        "Brier Score": brier_after,
    })

cal_metrics_df = pd.DataFrame(cal_metrics_rows)
print("\n" + cal_metrics_df.to_string(index=False))

print("\nNote: Brier Score remains the same before/after because rule-based")
print("calibration changes predicted labels but not probability distributions.")
print("For probability-level calibration, temperature scaling would be needed.")

# ── Section C: Confidence Histograms ────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("C. GENERATING CONFIDENCE HISTOGRAMS")
print("=" * 70)


def plot_confidence_histograms(probs, y_true, preds_before, preds_after,
                               model_name, save_path):
    """
    Plot confidence histograms before and after calibration.
    Shows distribution of max prediction probabilities, colored by correctness.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    conf = np.max(probs, axis=1)

    for idx, (preds, title_suffix) in enumerate([
        (preds_before, "Before Calibration"),
        (preds_after, "After Calibration")
    ]):
        ax = axes[idx]
        correct = (preds == y_true)

        ax.hist(conf[correct], bins=20, range=(0, 1), alpha=0.7,
                label=f"Correct ({correct.sum()})", color="#2196F3", edgecolor="white")
        ax.hist(conf[~correct], bins=20, range=(0, 1), alpha=0.7,
                label=f"Incorrect ({(~correct).sum()})", color="#F44336", edgecolor="white")

        acc = accuracy_score(y_true, preds)
        ax.set_title(f"{model_name} — {title_suffix}\nAccuracy: {acc:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Confidence (max probability)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {save_path}")


def plot_reliability_diagram(probs, y_true, preds, model_name, save_path, n_bins=10):
    """
    Reliability diagram (calibration curve).
    Shows perfect calibration (diagonal) vs actual calibration per bin.
    """
    confidences = np.max(probs, axis=1)
    accuracies = (preds == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_counts = []

    for b in range(n_bins):
        mask = (confidences > bin_boundaries[b]) & (confidences <= bin_boundaries[b + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_boundaries[b] + bin_boundaries[b + 1]) / 2)
            bin_accs.append(accuracies[mask].mean())
            bin_counts.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Reliability curve
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, color="#4CAF50",
            edgecolor="white", label="Model accuracy")
    ax1.scatter(bin_centers, bin_accs, color="#2196F3", s=50, zorder=5)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title(f"{model_name} — Reliability Diagram", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    # Confidence histogram
    ax2.bar(bin_centers, bin_counts, width=0.08, alpha=0.7, color="#FF9800", edgecolor="white")
    ax2.set_xlabel("Confidence", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {save_path}")


# Generate all plots
os.makedirs("calibration_plots", exist_ok=True)

print("\nGenerating confidence histograms...")
plot_confidence_histograms(ib_probs, y_test, ib_preds_raw, ib_preds_cal,
                           "IndicBERT", "calibration_plots/confidence_hist_indicbert.png")
plot_confidence_histograms(xlmr_probs, y_test, xlmr_preds_raw, xlmr_preds_cal,
                           "XLM-RoBERTa", "calibration_plots/confidence_hist_xlmr.png")
plot_confidence_histograms(ensemble_probs, y_test, ensemble_preds_raw, ensemble_preds_cal,
                           "Ensemble", "calibration_plots/confidence_hist_ensemble.png")

print("\nGenerating reliability diagrams...")
plot_reliability_diagram(ib_probs, y_test, ib_preds_raw,
                         "IndicBERT (Before)", "calibration_plots/reliability_indicbert_before.png")
plot_reliability_diagram(ib_probs, y_test, ib_preds_cal,
                         "IndicBERT (After)", "calibration_plots/reliability_indicbert_after.png")
plot_reliability_diagram(xlmr_probs, y_test, xlmr_preds_raw,
                         "XLM-R (Before)", "calibration_plots/reliability_xlmr_before.png")
plot_reliability_diagram(xlmr_probs, y_test, xlmr_preds_cal,
                         "XLM-R (After)", "calibration_plots/reliability_xlmr_after.png")
plot_reliability_diagram(ensemble_probs, y_test, ensemble_preds_raw,
                         "Ensemble (Before)", "calibration_plots/reliability_ensemble_before.png")
plot_reliability_diagram(ensemble_probs, y_test, ensemble_preds_cal,
                         "Ensemble (After)", "calibration_plots/reliability_ensemble_after.png")

# ── Save all tables ─────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

calibration_table.to_csv("calibration_before_after_table.csv", index=False)
cal_metrics_df.to_csv("calibration_metrics_ece_mce_brier.csv", index=False)

print("✓ Saved calibration_before_after_table.csv")
print("✓ Saved calibration_metrics_ece_mce_brier.csv")
print("✓ Saved calibration_plots/ (6 histograms + 6 reliability diagrams)")

print("\n" + "-" * 70)
print("FOR YOUR PAPER:")
print("-" * 70)
print("  • Table: Use calibration_before_after_table.csv for Before/After comparison")
print("  • Table: Use calibration_metrics_ece_mce_brier.csv for ECE/MCE/Brier")
print("  • Figures: Use confidence histograms to show prediction distribution")
print("  • Figures: Use reliability diagrams to show calibration quality")

print("\n" + "=" * 70)
print("✅ CALIBRATION ANALYSIS COMPLETE")
print("=" * 70)
