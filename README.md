# Hinglish Sentiment Analysis

Sentiment analysis system for code-mixed Hinglish (Hindi-English) product reviews using transformer models (IndicBERT, XLM-RoBERTa) with an ensemble approach.

## Project Structure

```
Sentiment Analysis/
├── notebooks/                    # Jupyter notebooks (training & exploration)
│   ├── cleandata.ipynb           # Data cleaning & preprocessing
│   ├── augment.ipynb             # Data augmentation (balancing classes)
│   ├── tfidf_logistic.ipynb      # Baseline: TF-IDF + LinearSVC + LIME
│   ├── bertmodel.ipynb           # Legacy training notebook — superseded by
│   │                                scripts/retrain_improved.py, kept for reference
│   └── finetune+shap.ipynb       # Evaluation, calibration & explainability
│
├── scripts/                      # Analysis & improvement scripts
│   ├── prepare_holdout_split.py  # Step 0: carves out the leakage-free test set
│   ├── statistical_tests.py      # McNemar test + bootstrap paired t-test
│   ├── calibration_analysis.py   # ECE/MCE/Brier + confidence histograms + temperature scaling
│   ├── ablation_study.py         # Component contribution analysis
│   ├── attribution_analysis.py   # English vs Hinglish token attribution
│   ├── quick_fixes.py            # Ensemble tuning (stacking, weighted avg)
│   └── retrain_improved.py       # Primary training script (class weights, dropout fix,
│                                    label smoothing, F1-based checkpointing)
│
├── results/                      # Generated outputs (regeneratable)
│   ├── csv/                      # Result tables
│   └── plots/                    # Visualizations
│       ├── calibration/          # Confidence histograms & reliability diagrams
│       ├── attribution/          # English vs Hinglish attribution plots
│       └── explainability/       # Gradient-based token attribution plots
│
├── data/                          # Data files
│   ├── Dataset-SA.csv             # Original (raw, then cleaned-in-place) dataset
│   ├── Dataset-SA-TrainPool.csv   # 85% split, source for augmentation (no test rows)
│   ├── Dataset-SA-Test-Clean.csv  # 15% held-out, never-augmented test set
│   ├── Dataset-SA-Augmented.csv   # Augmented (class-balanced) TRAINING data only
│   ├── evaluation_results.pkl     # Cached predictions & logits
│   └── tfidf_logreg_model.pkl     # Baseline TF-IDF + SVM model
│
├── indicbert-finetuned/          # Fine-tuned IndicBERT model checkpoint
└── xlmr-finetuned/               # Fine-tuned XLM-RoBERTa model checkpoint
```

## Full Pipeline — Run In This Order

Every script/notebook below uses paths relative to `scripts/` or `notebooks/`,
so run scripts from inside `scripts/` and notebooks from inside `notebooks/`
(e.g. `cd scripts` before `python statistical_tests.py`).

**Step 0 — data must exist first.** Put the raw dataset at `data/Dataset-SA.csv`
if it isn't there already.

```
1. notebooks/cleandata.ipynb          # Cleans Dataset-SA.csv in place
2. scripts/prepare_holdout_split.py   # Carves out data/Dataset-SA-Test-Clean.csv
                                       # BEFORE augmentation touches anything —
                                       # this is what keeps the final numbers
                                       # leakage-free and defensible in a paper.
                                       # Also writes data/Dataset-SA-TrainPool.csv.
3. notebooks/augment.ipynb            # Reads Dataset-SA-TrainPool.csv (never
                                       # the full dataset or the test set) and
                                       # writes data/Dataset-SA-Augmented.csv
4. notebooks/tfidf_logistic.ipynb     # Trains baseline, saves
                                       # data/tfidf_logreg_model.pkl
5. scripts/retrain_improved.py --model all
                                       # Trains IndicBERT + XLM-R with class
                                       # weighting, dropout fix, label
                                       # smoothing, F1-based checkpointing.
                                       # Evaluates on the real held-out test
                                       # set (Dataset-SA-Test-Clean.csv), not
                                       # a re-split of the augmented data.
                                       # ~2-4 hrs per model on a single GPU.
6. notebooks/finetune+shap.ipynb      # Run Cells 1-6 (or 6B if
                                       # evaluation_results.pkl already
                                       # exists) to produce
                                       # data/evaluation_results.pkl — the
                                       # input every script below needs.
```

**Step 7 — analysis scripts (run from inside `scripts/`):**

```bash
cd scripts
python statistical_tests.py     # McNemar (exact test for small samples) + bootstrap paired t-test
python calibration_analysis.py  # Before/after calibration table, ECE/MCE/Brier,
                                 # temperature-scaling baseline comparison, reliability diagrams
python ablation_study.py        # Component contribution analysis
python attribution_analysis.py  # English vs Hinglish token attribution (loads models)
python quick_fixes.py           # Ensemble tuning — alpha and stacker are tuned on a
                                 # held-out split, scored on a separate held-out split
```

Why this order matters: augmentation (step 3) creates paraphrased,
back-translated, and Hinglish-transformed duplicates of existing reviews. If
the test set were carved out *after* augmentation — or by re-splitting the
augmented file, as an earlier version of this pipeline did — near-duplicate
copies of the same underlying review could land on both sides of the split,
inflating every accuracy number downstream. Step 2 exists specifically to
prevent that.
