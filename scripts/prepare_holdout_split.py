"""
Step 0: Carve Out a Leakage-Free Held-Out Test Set
=====================================================
Splits the cleaned dataset into a training pool and a held-out test set
BEFORE augmentation ever runs. This matters because augment.ipynb creates
paraphrased / back-translated / Hinglish-transformed duplicates of existing
reviews — if the test set is carved out *after* augmentation (or from the
augmented file itself), near-duplicate copies of the same underlying review
can end up on both sides of the split, inflating every downstream accuracy
number. Any reviewer will flag this if noticed, so this step must run before
augment.ipynb.

Usage: python prepare_holdout_split.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SRC_PATH = os.path.join(DATA_DIR, "Dataset-SA.csv")
TRAIN_POOL_PATH = os.path.join(DATA_DIR, "Dataset-SA-TrainPool.csv")
TEST_CLEAN_PATH = os.path.join(DATA_DIR, "Dataset-SA-Test-Clean.csv")

TEST_FRACTION = 0.15
SEED = 42

print("=" * 70)
print("STEP 0: CARVING OUT A LEAKAGE-FREE HELD-OUT TEST SET")
print("=" * 70)

df = pd.read_csv(SRC_PATH)
df = df.dropna(subset=["Sentiment"]).reset_index(drop=True)

print(f"  Loaded {len(df)} rows from {SRC_PATH}")
print(f"  Class distribution:\n{df['Sentiment'].value_counts()}")

train_pool_df, test_clean_df = train_test_split(
    df, test_size=TEST_FRACTION, stratify=df["Sentiment"], random_state=SEED
)

train_pool_df = train_pool_df.reset_index(drop=True)
test_clean_df = test_clean_df.reset_index(drop=True)

train_pool_df.to_csv(TRAIN_POOL_PATH, index=False, encoding="utf-8-sig")
test_clean_df.to_csv(TEST_CLEAN_PATH, index=False, encoding="utf-8-sig")

print(f"\n  Train pool: {len(train_pool_df)} rows -> {TRAIN_POOL_PATH}")
print(f"    {train_pool_df['Sentiment'].value_counts().to_dict()}")
print(f"  Held-out test (never augmented): {len(test_clean_df)} rows -> {TEST_CLEAN_PATH}")
print(f"    {test_clean_df['Sentiment'].value_counts().to_dict()}")

print("\n" + "-" * 70)
print("NEXT STEPS:")
print("-" * 70)
print("  1. Run notebooks/augment.ipynb — it now reads Dataset-SA-TrainPool.csv")
print("     (not Dataset-SA.csv), so augmentation never touches the test rows.")
print("  2. Run notebooks/tfidf_logistic.ipynb to train the baseline on the")
print("     augmented train pool.")
print("  3. Run scripts/retrain_improved.py to train IndicBERT/XLM-R — it")
print("     evaluates on Dataset-SA-Test-Clean.csv, not a re-split of the")
print("     augmented data.")
print("  4. Run notebooks/finetune+shap.ipynb Cells 1-6 to produce")
print("     evaluation_results.pkl from the same held-out test set.")
print("=" * 70)
