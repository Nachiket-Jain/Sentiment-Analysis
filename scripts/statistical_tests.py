"""
Task 1: Statistical Significance Tests
========================================
McNemar's Test + Bootstrap Paired t-test for comparing model performance.
Loads from evaluation_results.pkl — no retraining needed.

Usage: python statistical_tests.py
"""

import pickle
import os
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

# ── Load saved results ──────────────────────────────────────────────────────────
PKL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "evaluation_results.pkl")

print("=" * 70)
print("TASK 1: STATISTICAL SIGNIFICANCE TESTS")
print("=" * 70)

print("\nLoading evaluation results...")
with open(PKL_PATH, "rb") as f:
    saved = pickle.load(f)

ib_results = saved["ib_results"]
xlmr_results = saved["xlmr_results"]
baseline_preds = saved["baseline_preds"]
y_test = saved["y_test"]
label2id = saved["label2id"]
id2label = saved["id2label"]

# Get predictions as arrays
ib_preds = ib_results["predictions"]
xlmr_preds = xlmr_results["predictions"]

# Ensure all are numpy arrays
ib_preds = np.array(ib_preds)
xlmr_preds = np.array(xlmr_preds)
baseline_preds = np.array(baseline_preds)
y_test = np.array(y_test)

# Ensemble predictions
ensemble_logits = (ib_results["logits"] + xlmr_results["logits"]) / 2.0
ensemble_preds = np.argmax(ensemble_logits, axis=-1)

models = {
    "IndicBERT": ib_preds,
    "XLM-R": xlmr_preds,
    "Baseline (TF-IDF+SVM)": baseline_preds,
    "Ensemble (IB+XLM-R)": ensemble_preds,
}

print(f"  Test set size: {len(y_test)}")
print(f"  Models: {list(models.keys())}")

# ── 1. McNemar's Test ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("1. McNEMAR'S TEST (Pairwise Model Comparison)")
print("=" * 70)
print("H0: Both models have the same error rate")
print("H1: Models have significantly different error rates\n")


def mcnemar_test(y_true, preds_a, preds_b):
    """
    McNemar's test for comparing two classifiers on the same test set.
    
    Builds a 2x2 contingency table:
        |             | B correct | B wrong |
        |-------------|-----------|---------|
        | A correct   |    n00    |   n01   |
        | A wrong     |    n10    |   n11   |
    
    Test statistic: chi2 = (|n01 - n10| - 1)^2 / (n01 + n10)
    (with continuity correction)
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    # Contingency table
    n00 = np.sum(correct_a & correct_b)       # Both correct
    n01 = np.sum(correct_a & ~correct_b)      # A correct, B wrong
    n10 = np.sum(~correct_a & correct_b)      # A wrong, B correct
    n11 = np.sum(~correct_a & ~correct_b)     # Both wrong

    # McNemar's test with continuity correction
    if (n01 + n10) == 0:
        return {
            "n00": int(n00), "n01": int(n01),
            "n10": int(n10), "n11": int(n11),
            "chi2": 0.0, "p_value": 1.0,
            "significant": False
        }

    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "n00": int(n00), "n01": int(n01),
        "n10": int(n10), "n11": int(n11),
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05
    }


# Run McNemar test for all pairs
mcnemar_results = []
model_names = list(models.keys())

for name_a, name_b in combinations(model_names, 2):
    result = mcnemar_test(y_test, models[name_a], models[name_b])
    result["Model A"] = name_a
    result["Model B"] = name_b
    mcnemar_results.append(result)

    print(f"  {name_a} vs {name_b}:")
    print(f"    Contingency: A+B+={result['n00']}, A+B-={result['n01']}, "
          f"A-B+={result['n10']}, A-B-={result['n11']}")
    print(f"    chi2 = {result['chi2']:.4f}, p = {result['p_value']:.6f}"
          f"  {'*** SIGNIFICANT ***' if result['significant'] else '(not significant)'}")
    print()

# Create McNemar summary table
mcnemar_df = pd.DataFrame(mcnemar_results)
mcnemar_df = mcnemar_df[["Model A", "Model B", "chi2", "p_value", "significant",
                          "n01", "n10", "n00", "n11"]]
mcnemar_df.columns = ["Model A", "Model B", "chi2", "p-value", "Significant (p<0.05)",
                       "A+ B-", "A- B+", "Both+", "Both-"]

print("\n" + "-" * 70)
print("McNEMAR'S TEST SUMMARY TABLE")
print("-" * 70)
print(mcnemar_df[["Model A", "Model B", "chi2", "p-value", "Significant (p<0.05)"]].to_string(index=False))

# ── 2. Bootstrap Paired t-test ──────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("2. BOOTSTRAP PAIRED t-TEST")
print("=" * 70)
print("Resamples test set 1000 times, computes accuracy per resample,")
print("then runs paired t-test on the paired accuracy differences.\n")


def bootstrap_paired_ttest(y_true, preds_a, preds_b, n_bootstrap=1000, seed=42):
    """
    Bootstrap paired t-test for comparing two classifiers.
    
    1. Resample test set with replacement (1000 times)
    2. Compute accuracy for both models on each resample
    3. Run paired t-test on the accuracy differences
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    acc_a_samples = []
    acc_b_samples = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        y_boot = y_true[indices]
        pa_boot = preds_a[indices]
        pb_boot = preds_b[indices]

        acc_a_samples.append(np.mean(pa_boot == y_boot))
        acc_b_samples.append(np.mean(pb_boot == y_boot))

    acc_a_samples = np.array(acc_a_samples)
    acc_b_samples = np.array(acc_b_samples)

    # Paired t-test on bootstrap accuracy differences
    t_stat, p_value = stats.ttest_rel(acc_a_samples, acc_b_samples)

    # Effect size (Cohen's d for paired samples)
    diff = acc_a_samples - acc_b_samples
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)

    return {
        "mean_acc_a": round(np.mean(acc_a_samples), 4),
        "mean_acc_b": round(np.mean(acc_b_samples), 4),
        "mean_diff": round(np.mean(diff), 4),
        "std_diff": round(np.std(diff, ddof=1), 4),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "cohens_d": round(cohens_d, 4),
        "significant": p_value < 0.05,
        "ci_95_lower": round(np.percentile(diff, 2.5), 4),
        "ci_95_upper": round(np.percentile(diff, 97.5), 4),
    }


# Run bootstrap paired t-test for all pairs
bootstrap_results = []

for name_a, name_b in combinations(model_names, 2):
    result = bootstrap_paired_ttest(y_test, models[name_a], models[name_b])
    result["Model A"] = name_a
    result["Model B"] = name_b
    bootstrap_results.append(result)

    print(f"  {name_a} vs {name_b}:")
    print(f"    Mean Acc A: {result['mean_acc_a']:.4f}, Mean Acc B: {result['mean_acc_b']:.4f}")
    print(f"    Mean Diff: {result['mean_diff']:+.4f} (95% CI: [{result['ci_95_lower']:+.4f}, {result['ci_95_upper']:+.4f}])")
    print(f"    t = {result['t_stat']:.4f}, p = {result['p_value']:.6f}, Cohen's d = {result['cohens_d']:.4f}")
    print(f"    {'*** SIGNIFICANT ***' if result['significant'] else '(not significant)'}")
    print()

# Create bootstrap summary table
bootstrap_df = pd.DataFrame(bootstrap_results)
bootstrap_df = bootstrap_df[["Model A", "Model B", "mean_acc_a", "mean_acc_b",
                              "mean_diff", "t_stat", "p_value", "cohens_d", "significant"]]
bootstrap_df.columns = ["Model A", "Model B", "Acc A", "Acc B",
                         "Δ Acc", "t-stat", "p-value", "Cohen's d", "Significant"]

print("\n" + "-" * 70)
print("BOOTSTRAP PAIRED t-TEST SUMMARY TABLE")
print("-" * 70)
print(bootstrap_df.to_string(index=False))

# ── Save all results ────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

mcnemar_df.to_csv("../results/csv/mcnemar_test_results.csv", index=False)
bootstrap_df.to_csv("../results/csv/bootstrap_ttest_results.csv", index=False)

print("✓ Saved mcnemar_test_results.csv")
print("✓ Saved bootstrap_ttest_results.csv")

# Combined summary for the paper
print("\n" + "-" * 70)
print("FOR YOUR PAPER — KEY FINDINGS:")
print("-" * 70)
for r in mcnemar_results:
    if r["significant"]:
        print(f"  - {r['Model A']} vs {r['Model B']}: "
              f"Significantly different (chi2={r['chi2']:.2f}, p={r['p_value']:.4f})")
    else:
        print(f"  - {r['Model A']} vs {r['Model B']}: "
              f"NOT significantly different (chi2={r['chi2']:.2f}, p={r['p_value']:.4f})")

print("\n" + "=" * 70)
print("STATISTICAL TESTS COMPLETE")
print("=" * 70)
