"""
Task 3: Ablation Study
========================
Systematically removes components to measure their individual contribution.
Loads from evaluation_results.pkl — no retraining needed.

Usage: python ablation_study.py
"""

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.special import softmax

# ── Load saved results ──────────────────────────────────────────────────────────
PKL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "evaluation_results.pkl")
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", "csv"), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", "plots"), exist_ok=True)

print("=" * 70)
print("TASK 3: ABLATION STUDY")
print("=" * 70)

print("\nLoading evaluation results...")
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
ib_preds = np.array(ib_results["predictions"])
xlmr_preds = np.array(xlmr_results["predictions"])

ib_probs = softmax(ib_logits, axis=1)
xlmr_probs = softmax(xlmr_logits, axis=1)

ensemble_logits = (ib_logits + xlmr_logits) / 2.0
ensemble_probs = softmax(ensemble_logits, axis=1)
ensemble_preds = np.argmax(ensemble_logits, axis=1)

reviews = test_df["Review"].reset_index(drop=True)

print(f"  Test set size: {len(y_test)}")

# ── Calibration logic (reused from Cell 11B) ────────────────────────────────────
HINGLISH_POSITIVE = [
    'mast', 'badhiya', 'accha', 'achha', 'zabardast', 'kamaal',
    'ekdum', 'sahi', 'best', 'super', 'boht', 'bahut', 'bohot',
    'shandar', 'awesome', 'perfect', 'excellent'
]

HINGLISH_NEGATIVE = [
    'bakwas', 'bekar', 'bekaar', 'ghatiya', 'faltu', 'kharab',
    'worst', 'terrible', 'bad', 'useless'
]


def apply_full_calibration(probs_array, texts, model_name="IndicBERT"):
    """Apply all calibration rules (standard + Hinglish)."""
    preds = np.argmax(probs_array, axis=1).copy()

    if model_name == "XLM-R":
        min_pos_threshold = 0.0006
        max_neg_threshold = 0.030
    else:
        min_pos_threshold = 0.003
        max_neg_threshold = 0.10

    for i in range(len(preds)):
        neg_p, neu_p, pos_p = probs_array[i]
        pred = preds[i]
        text_lower = str(texts.iloc[i]).lower() if hasattr(texts, 'iloc') else str(texts[i]).lower()
        has_hinglish_pos = any(w in text_lower for w in HINGLISH_POSITIVE)
        has_hinglish_neg = any(w in text_lower for w in HINGLISH_NEGATIVE)

        if pred == 1 and neu_p > 0.85 and neg_p < max_neg_threshold and pos_p > min_pos_threshold:
            preds[i] = 2
        elif has_hinglish_pos and not has_hinglish_neg and pred == 1 and pos_p > neg_p and neg_p < 0.20:
            preds[i] = 2
        elif has_hinglish_neg and pred == 1 and neg_p > 0.05 and pos_p < 0.15:
            preds[i] = 0

    return preds


def apply_standard_calibration_only(probs_array, model_name="IndicBERT"):
    """Apply only the standard threshold calibration (no Hinglish rules)."""
    preds = np.argmax(probs_array, axis=1).copy()

    if model_name == "XLM-R":
        min_pos_threshold = 0.0006
        max_neg_threshold = 0.030
    else:
        min_pos_threshold = 0.003
        max_neg_threshold = 0.10

    for i in range(len(preds)):
        neg_p, neu_p, pos_p = probs_array[i]
        if preds[i] == 1 and neu_p > 0.85 and neg_p < max_neg_threshold and pos_p > min_pos_threshold:
            preds[i] = 2

    return preds


def apply_hinglish_only(probs_array, texts):
    """Apply only Hinglish keyword rules (no standard calibration)."""
    preds = np.argmax(probs_array, axis=1).copy()

    for i in range(len(preds)):
        neg_p, neu_p, pos_p = probs_array[i]
        pred = preds[i]
        text_lower = str(texts.iloc[i]).lower() if hasattr(texts, 'iloc') else str(texts[i]).lower()
        has_hinglish_pos = any(w in text_lower for w in HINGLISH_POSITIVE)
        has_hinglish_neg = any(w in text_lower for w in HINGLISH_NEGATIVE)

        if has_hinglish_pos and not has_hinglish_neg and pred == 1 and pos_p > neg_p and neg_p < 0.20:
            preds[i] = 2
        elif has_hinglish_neg and pred == 1 and neg_p > 0.05 and pos_p < 0.15:
            preds[i] = 0

    return preds


# ── Compute metrics for all ablation variants ───────────────────────────────────
def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "F1 Macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "F1 Weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "Neg Recall": round(recall_score(y_true, y_pred, labels=[0], average=None, zero_division=0)[0], 4),
        "Pos Precision": round(precision_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0], 4),
    }


print("\nComputing ablation variants...")

# Variant 1: Full System (Ensemble + All Calibration)
full_system_preds = apply_full_calibration(ensemble_probs, reviews, "Ensemble")

# Variant 2: Without ANY Calibration (raw ensemble)
no_calibration_preds = ensemble_preds.copy()

# Variant 3: Without Hinglish Rules (only standard calibration)
no_hinglish_preds = apply_standard_calibration_only(ensemble_probs, "Ensemble")

# Variant 4: Without Standard Calibration (only Hinglish rules)
no_standard_preds = apply_hinglish_only(ensemble_probs, reviews)

# Variant 5: Without Ensemble — XLM-R only + full calibration
xlmr_only_cal = apply_full_calibration(xlmr_probs, reviews, "XLM-R")

# Variant 6: Without Ensemble — IndicBERT only + full calibration
ib_only_cal = apply_full_calibration(ib_probs, reviews, "IndicBERT")

# Variant 7: XLM-R only, no calibration
xlmr_only_raw = xlmr_preds.copy()

# Variant 8: IndicBERT only, no calibration
ib_only_raw = ib_preds.copy()

# Variant 9: Baseline (TF-IDF + SVM)
baseline_raw = baseline_preds.copy()

# ── Build ablation table ────────────────────────────────────────────────────────
ablation_rows = [
    ("Full System (Ensemble + All Calibration)", full_system_preds),
    ("− Remove All Calibration", no_calibration_preds),
    ("− Remove Hinglish Rules Only", no_hinglish_preds),
    ("− Remove Standard Calibration Only", no_standard_preds),
    ("− Remove Ensemble (XLM-R + Calibration)", xlmr_only_cal),
    ("− Remove Ensemble (IndicBERT + Calibration)", ib_only_cal),
    ("− Remove Ensemble + Calibration (XLM-R raw)", xlmr_only_raw),
    ("− Remove Ensemble + Calibration (IndicBERT raw)", ib_only_raw),
    ("− Use Baseline Only (TF-IDF + SVM)", baseline_raw),
]

ablation_data = []
for variant_name, preds in ablation_rows:
    metrics = compute_metrics(y_test, preds)
    metrics["Variant"] = variant_name
    ablation_data.append(metrics)

ablation_df = pd.DataFrame(ablation_data)
ablation_df = ablation_df[["Variant", "Accuracy", "F1 Macro", "F1 Weighted", "Neg Recall", "Pos Precision"]]

# Compute deltas from Full System
full_acc = ablation_data[0]["Accuracy"]
full_f1 = ablation_data[0]["F1 Macro"]
ablation_df["Δ Accuracy"] = ablation_df["Accuracy"].apply(lambda x: round(x - full_acc, 4))
ablation_df["Δ F1 Macro"] = ablation_df["F1 Macro"].apply(lambda x: round(x - full_f1, 4))

print("\n" + "=" * 70)
print("ABLATION STUDY RESULTS")
print("=" * 70)
print("\n" + ablation_df.to_string(index=False))

# ── Component Contribution Summary ──────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("COMPONENT CONTRIBUTION ANALYSIS")
print("=" * 70)

contributions = [
    ("Ensemble (vs XLM-R alone)",
     ablation_data[0]["Accuracy"] - ablation_data[4]["Accuracy"],
     ablation_data[0]["F1 Macro"] - ablation_data[4]["F1 Macro"]),
    ("Ensemble (vs IndicBERT alone)",
     ablation_data[0]["Accuracy"] - ablation_data[5]["Accuracy"],
     ablation_data[0]["F1 Macro"] - ablation_data[5]["F1 Macro"]),
    ("All Calibration",
     ablation_data[0]["Accuracy"] - ablation_data[1]["Accuracy"],
     ablation_data[0]["F1 Macro"] - ablation_data[1]["F1 Macro"]),
    ("Hinglish Rules",
     ablation_data[0]["Accuracy"] - ablation_data[2]["Accuracy"],
     ablation_data[0]["F1 Macro"] - ablation_data[2]["F1 Macro"]),
    ("Standard Calibration",
     ablation_data[0]["Accuracy"] - ablation_data[3]["Accuracy"],
     ablation_data[0]["F1 Macro"] - ablation_data[3]["F1 Macro"]),
    ("Transformers (vs Baseline)",
     ablation_data[0]["Accuracy"] - ablation_data[8]["Accuracy"],
     ablation_data[0]["F1 Macro"] - ablation_data[8]["F1 Macro"]),
]

contrib_df = pd.DataFrame(contributions, columns=["Component", "Δ Accuracy", "Δ F1 Macro"])
contrib_df["Δ Accuracy"] = contrib_df["Δ Accuracy"].round(4)
contrib_df["Δ F1 Macro"] = contrib_df["Δ F1 Macro"].round(4)

print("\n" + contrib_df.to_string(index=False))

# ── Ablation Bar Chart ──────────────────────────────────────────────────────────
print("\n\nGenerating ablation bar chart...")

fig, ax = plt.subplots(figsize=(12, 7))

variants = [r["Variant"] for r in ablation_data]
accuracies = [r["Accuracy"] for r in ablation_data]
f1_macros = [r["F1 Macro"] for r in ablation_data]

x = np.arange(len(variants))
width = 0.35

bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy",
               color="#2196F3", alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + width/2, f1_macros, width, label="F1 Macro",
               color="#FF9800", alpha=0.85, edgecolor="white")

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7, rotation=45)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

ax.set_ylabel("Score", fontsize=12)
ax.set_title("Ablation Study — Component Contribution", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(variants, rotation=35, ha="right", fontsize=8)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Highlight the full system bar
bars1[0].set_edgecolor("#1565C0")
bars1[0].set_linewidth(2)
bars2[0].set_edgecolor("#E65100")
bars2[0].set_linewidth(2)

# Set y-axis to start near the minimum to show differences better
min_val = min(min(accuracies), min(f1_macros))
ax.set_ylim(max(0, min_val - 0.05), 1.0)

plt.tight_layout()
plt.savefig("../results/plots/ablation_study_chart.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ Saved ablation_study_chart.png")

# ── Save results ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

ablation_df.to_csv("../results/csv/ablation_study_results.csv", index=False)
contrib_df.to_csv("../results/csv/ablation_component_contributions.csv", index=False)

print("✓ Saved ablation_study_results.csv")
print("✓ Saved ablation_component_contributions.csv")
print("✓ Saved ablation_study_chart.png")

print("\n" + "-" * 70)
print("FOR YOUR PAPER:")
print("-" * 70)
print("  • Table: Use ablation_study_results.csv — main ablation table")
print("  • Table: Use ablation_component_contributions.csv — per-component impact")
print("  • Figure: Use ablation_study_chart.png — visual comparison")
print("  • The Full System row is your reference; all others show degradation")

print("\n" + "=" * 70)
print("✅ ABLATION STUDY COMPLETE")
print("=" * 70)
