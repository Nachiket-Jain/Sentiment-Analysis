"""
Task 4: Attribution Score Analysis
====================================
- Average attribution score per model
- English vs Hinglish token attribution comparison
- Percentage of important Hinglish tokens in top-k

Requires loading model checkpoints for gradient-based attribution (inference only).

Usage: python attribution_analysis.py
"""

import pickle
import os
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict

# ── Configuration ───────────────────────────────────────────────────────────────
INDICBERT_DIR = os.path.join(os.path.dirname(__file__), "..", "indicbert-finetuned")
XLMR_DIR = os.path.join(os.path.dirname(__file__), "..", "xlmr-finetuned")
PKL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "evaluation_results.pkl")
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", "csv"), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", "plots"), exist_ok=True)

# Number of test samples to analyze (full test set takes long on CPU)
N_SAMPLES = 200  # Increase if you have time; 200 takes ~15-30 min on CPU

# Known Hinglish/Hindi-origin tokens (expanded list for detection)
HINGLISH_TOKENS = {
    # Positive sentiment words
    'mast', 'badhiya', 'accha', 'achha', 'acha', 'zabardast', 'kamaal',
    'kamal', 'ekdum', 'sahi', 'boht', 'bahut', 'bohot', 'bhut',
    'shandar', 'jabardast', 'jhakas', 'dhansu', 'fatafat', 'tagda',
    'lajawab', 'umda', 'shaandar', 'behtareen', 'behtarin',
    # Negative sentiment words
    'bakwas', 'bekar', 'bekaar', 'ghatiya', 'faltu', 'kharab',
    'wahiyat', 'gandaa', 'ganda', 'tatti', 'bura', 'nikamma',
    # Intensifiers/modifiers
    'bilkul', 'ekdum', 'bahot', 'zyada', 'thoda', 'kuch',
    'nahi', 'nai', 'nahin', 'mat', 'karo',
    # Common Hindi words in reviews
    'hai', 'hain', 'tha', 'thi', 'ka', 'ki', 'ke', 'ko', 'se',
    'me', 'mein', 'par', 'pe', 'aur', 'ya', 'lekin', 'magar',
    'wala', 'wali', 'wale', 'abhi', 'yeh', 'ye', 'woh', 'wo',
    'kya', 'kaise', 'kaisa', 'kaisi', 'kitna', 'kitni', 'kitne',
    'paisa', 'paise', 'rupee', 'rupaye', 'daam', 'kimat',
    'cheez', 'chiz', 'saman', 'samaan', 'product',
    'dekho', 'dekha', 'dekhi', 'liya', 'lena', 'kharida',
    'pasand', 'pasandida', 'acchi', 'buri', 'theek', 'thik',
    'chalega', 'chalta', 'chal', 'milta', 'mila', 'mile',
    'pehle', 'baad', 'uske', 'iske', 'jaise', 'waise',
}

# ── Load models ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("TASK 4: ATTRIBUTION SCORE ANALYSIS")
print("=" * 70)

device = torch.device("cpu")

print("\nLoading models for gradient attribution (inference only)...")
print("  Loading IndicBERT...")
tok_ib = AutoTokenizer.from_pretrained(INDICBERT_DIR)
model_ib = AutoModelForSequenceClassification.from_pretrained(
    INDICBERT_DIR, torch_dtype=torch.float32
)
model_ib.to(device)
model_ib.eval()
print("  ✓ IndicBERT loaded")

print("  Loading XLM-RoBERTa...")
tok_xlmr = AutoTokenizer.from_pretrained(XLMR_DIR)
model_xlmr = AutoModelForSequenceClassification.from_pretrained(
    XLMR_DIR, torch_dtype=torch.float32
)
model_xlmr.to(device)
model_xlmr.eval()
print("  ✓ XLM-R loaded")

# Load test data
print("  Loading evaluation data...")
with open(PKL_PATH, "rb") as f:
    saved = pickle.load(f)

test_df = saved["test_df"]
y_test = np.array(saved["y_test"])
id2label = saved["id2label"]
label2id = saved["label2id"]

# Sample reviews for analysis
np.random.seed(42)
sample_indices = np.random.choice(len(test_df), size=min(N_SAMPLES, len(test_df)), replace=False)
sample_df = test_df.iloc[sample_indices].reset_index(drop=True)
sample_labels = y_test[sample_indices]

print(f"  Analyzing {len(sample_df)} samples from test set")

# ── Gradient-based Attribution ──────────────────────────────────────────────────

def compute_attribution(text, model, tokenizer, id2label, remove_token_type_ids=False):
    """
    Compute gradient-based token attribution for a single text.
    Returns: list of (token, importance_score, is_special) tuples
    """
    model.eval()

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128, padding=True
    )

    if remove_token_type_ids and "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Enable gradient computation
    embeddings = model.get_input_embeddings()
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    embedded = embeddings(input_ids)
    embedded.retain_grad()

    if attention_mask is not None:
        outputs = model(inputs_embeds=embedded, attention_mask=attention_mask)
    else:
        outputs = model(inputs_embeds=embedded)

    logits = outputs.logits
    pred_idx = torch.argmax(logits, dim=-1).item()

    # Backward pass
    logits[0, pred_idx].backward()

    if embedded.grad is not None:
        token_importance = embedded.grad.abs().sum(dim=-1)[0].detach().cpu().numpy()
    else:
        token_importance = np.zeros(len(tokens))

    # Normalize
    if token_importance.max() > 0:
        token_importance = token_importance / token_importance.max()

    # Identify special tokens
    special_tokens = {'[CLS]', '[SEP]', '<s>', '</s>', '<pad>', '▁', '[PAD]', '[UNK]'}

    result = []
    for tok, score in zip(tokens, token_importance):
        is_special = tok in special_tokens
        clean_tok = tok.replace('▁', '').replace('##', '').strip()
        if clean_tok and not is_special:
            result.append((clean_tok, float(score)))

    return result, id2label[pred_idx]


def is_hinglish_token(token):
    """
    Check if a token is Hinglish/Hindi-origin.

    Only exact whole-word matches against HINGLISH_TOKENS, plus exact matches
    on WordPiece/SentencePiece continuation fragments (tokens with no
    remaining subword marker after cleanup are treated as whole words).
    The previous version did loose substring containment (`clean in hw or hw
    in clean`), which misclassified many ordinary English words as Hinglish
    whenever they happened to contain a short Hindi function word as a
    substring (e.g. "hair" contains "hai", "seat" contains "se") — that
    inflated the reported Hinglish/English attribution gap.
    """
    clean = token.lower().strip()
    return clean in HINGLISH_TOKENS


# ── Run attribution analysis ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RUNNING GRADIENT ATTRIBUTION ANALYSIS")
print("=" * 70)

results = {
    "IndicBERT": {"all_scores": [], "english_scores": [], "hinglish_scores": [],
                   "top10_hinglish_pcts": [], "token_counts": defaultdict(int)},
    "XLM-R": {"all_scores": [], "english_scores": [], "hinglish_scores": [],
               "top10_hinglish_pcts": [], "token_counts": defaultdict(int)},
}

model_configs = [
    ("IndicBERT", model_ib, tok_ib, True),
    ("XLM-R", model_xlmr, tok_xlmr, False),
]

for model_name, model, tokenizer, remove_ttids in model_configs:
    print(f"\n  Processing {model_name}...")
    r = results[model_name]

    for idx in range(len(sample_df)):
        text = sample_df.iloc[idx]["Review"]

        try:
            token_attrs, pred_label = compute_attribution(
                text, model, tokenizer, id2label, remove_token_type_ids=remove_ttids
            )
        except Exception as e:
            continue

        if not token_attrs:
            continue

        # Collect scores
        for tok, score in token_attrs:
            r["all_scores"].append(score)
            if is_hinglish_token(tok):
                r["hinglish_scores"].append(score)
                r["token_counts"]["hinglish"] += 1
            else:
                r["english_scores"].append(score)
                r["token_counts"]["english"] += 1

        # Top-10 analysis
        sorted_attrs = sorted(token_attrs, key=lambda x: x[1], reverse=True)
        top_10 = sorted_attrs[:min(10, len(sorted_attrs))]
        hinglish_in_top10 = sum(1 for tok, _ in top_10 if is_hinglish_token(tok))
        r["top10_hinglish_pcts"].append(hinglish_in_top10 / len(top_10) * 100)

        if (idx + 1) % 50 == 0:
            print(f"    Processed {idx + 1}/{len(sample_df)} samples...")

    print(f"    ✓ {model_name} complete — "
          f"{len(r['all_scores'])} tokens analyzed, "
          f"{r['token_counts']['hinglish']} Hinglish, "
          f"{r['token_counts']['english']} English")

# ── Compute Summary Statistics ──────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("ATTRIBUTION ANALYSIS RESULTS")
print("=" * 70)

summary_rows = []
for model_name in ["IndicBERT", "XLM-R"]:
    r = results[model_name]

    avg_all = np.mean(r["all_scores"]) if r["all_scores"] else 0
    avg_eng = np.mean(r["english_scores"]) if r["english_scores"] else 0
    avg_hing = np.mean(r["hinglish_scores"]) if r["hinglish_scores"] else 0
    avg_top10_pct = np.mean(r["top10_hinglish_pcts"]) if r["top10_hinglish_pcts"] else 0

    total_tokens = r["token_counts"]["hinglish"] + r["token_counts"]["english"]
    hinglish_pct = (r["token_counts"]["hinglish"] / total_tokens * 100) if total_tokens > 0 else 0

    summary_rows.append({
        "Model": model_name,
        "Avg Attribution (All)": round(avg_all, 4),
        "Avg Attribution (English)": round(avg_eng, 4),
        "Avg Attribution (Hinglish)": round(avg_hing, 4),
        "Hinglish/English Ratio": round(avg_hing / avg_eng, 4) if avg_eng > 0 else 0,
        "% Hinglish in Top-10": round(avg_top10_pct, 2),
        "Total Hinglish Tokens": r["token_counts"]["hinglish"],
        "Total English Tokens": r["token_counts"]["english"],
        "% Hinglish Tokens Overall": round(hinglish_pct, 2),
    })

summary_df = pd.DataFrame(summary_rows)

print("\n1. AVERAGE ATTRIBUTION SCORES:")
print("-" * 70)
print(summary_df[["Model", "Avg Attribution (All)", "Avg Attribution (English)",
                   "Avg Attribution (Hinglish)", "Hinglish/English Ratio"]].to_string(index=False))

print("\n\n2. HINGLISH TOKEN IMPORTANCE:")
print("-" * 70)
print(summary_df[["Model", "% Hinglish in Top-10", "% Hinglish Tokens Overall",
                   "Total Hinglish Tokens", "Total English Tokens"]].to_string(index=False))

# ── Interpretation ──────────────────────────────────────────────────────────────
print("\n\n3. INTERPRETATION:")
print("-" * 70)
for row in summary_rows:
    ratio = row["Hinglish/English Ratio"]
    model = row["Model"]
    if ratio > 1.0:
        print(f"  {model}: Hinglish tokens receive {ratio:.2f}x MORE attention than English tokens")
        print(f"         → Model recognizes Hinglish sentiment markers effectively")
    elif ratio > 0.8:
        print(f"  {model}: Hinglish tokens receive comparable attention to English ({ratio:.2f}x)")
        print(f"         → Model handles both languages fairly equally")
    else:
        print(f"  {model}: Hinglish tokens receive {ratio:.2f}x LESS attention than English tokens")
        print(f"         → Model may underweight Hinglish sentiment cues")

# ── Visualization ───────────────────────────────────────────────────────────────
print("\n\nGenerating attribution comparison plots...")

os.makedirs("../results/plots/attribution", exist_ok=True)

# Plot 1: English vs Hinglish attribution distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, model_name in enumerate(["IndicBERT", "XLM-R"]):
    ax = axes[idx]
    r = results[model_name]

    eng_scores = r["english_scores"]
    hing_scores = r["hinglish_scores"]

    ax.hist(eng_scores, bins=30, alpha=0.7, label=f"English (n={len(eng_scores)})",
            color="#2196F3", edgecolor="white", density=True)
    ax.hist(hing_scores, bins=30, alpha=0.7, label=f"Hinglish (n={len(hing_scores)})",
            color="#FF9800", edgecolor="white", density=True)

    avg_eng = np.mean(eng_scores) if eng_scores else 0
    avg_hing = np.mean(hing_scores) if hing_scores else 0
    ax.axvline(avg_eng, color="#1565C0", linestyle="--", linewidth=2, label=f"Eng mean: {avg_eng:.3f}")
    ax.axvline(avg_hing, color="#E65100", linestyle="--", linewidth=2, label=f"Hing mean: {avg_hing:.3f}")

    ax.set_title(f"{model_name}\nAttribution Score Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Attribution Score (normalized)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../results/plots/attribution/english_vs_hinglish_attribution.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✓ Saved attribution_plots/english_vs_hinglish_attribution.png")

# Plot 2: Summary bar chart
fig, ax = plt.subplots(figsize=(10, 6))

models = ["IndicBERT", "XLM-R"]
eng_means = [np.mean(results[m]["english_scores"]) for m in models]
hing_means = [np.mean(results[m]["hinglish_scores"]) for m in models]

x = np.arange(len(models))
width = 0.3

bars1 = ax.bar(x - width/2, eng_means, width, label="English Tokens",
               color="#2196F3", alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + width/2, hing_means, width, label="Hinglish Tokens",
               color="#FF9800", alpha=0.85, edgecolor="white")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.4f}", ha="center", fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.4f}", ha="center", fontsize=10)

ax.set_ylabel("Average Attribution Score", fontsize=12)
ax.set_title("English vs Hinglish Token Attribution Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("../results/plots/attribution/attribution_comparison_bar.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✓ Saved attribution_plots/attribution_comparison_bar.png")

# Plot 3: % Hinglish in Top-10 per model
fig, ax = plt.subplots(figsize=(8, 5))

top10_data = [results[m]["top10_hinglish_pcts"] for m in models]
# pyrefly: ignore [unexpected-keyword]
bp = ax.boxplot(top10_data, labels=models, patch_artist=True,
                boxprops=dict(facecolor="#E3F2FD", edgecolor="#1565C0"),
                medianprops=dict(color="#F44336", linewidth=2))

for i, data in enumerate(top10_data):
    ax.scatter([i + 1] * len(data), data, alpha=0.1, color="#2196F3", s=10)

ax.set_ylabel("% Hinglish Tokens in Top-10 Important", fontsize=12)
ax.set_title("Distribution of Hinglish Token Importance", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("../results/plots/attribution/hinglish_top10_boxplot.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✓ Saved attribution_plots/hinglish_top10_boxplot.png")

# ── Save results ────────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

summary_df.to_csv("../results/csv/attribution_analysis_results.csv", index=False)
print("✓ Saved attribution_analysis_results.csv")
print("✓ Saved attribution_plots/ (3 visualization files)")

print("\n" + "-" * 70)
print("FOR YOUR PAPER:")
print("-" * 70)
print("  • Table: Use attribution_analysis_results.csv for the analysis table")
print("  • Key metrics: Avg Attribution Score, Hinglish/English Ratio, % Hinglish in Top-10")
print("  • Figure: Use english_vs_hinglish_attribution.png for distribution comparison")
print("  • Figure: Use attribution_comparison_bar.png for summary comparison")
print("  • Discuss whether the models attend more to Hinglish or English tokens")

print("\n" + "=" * 70)
print("✅ ATTRIBUTION ANALYSIS COMPLETE")
print("=" * 70)
