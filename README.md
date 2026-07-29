# Hinglish Sentiment Analysis

Sentiment analysis system for code-mixed Hinglish (Hindi-English) product reviews using transformer models (IndicBERT, XLM-RoBERTa) with an ensemble approach.

## Project Structure

```
Sentiment Analysis/
├── notebooks/                    # Jupyter notebooks (training & exploration)
│   ├── cleandata.ipynb           # Data cleaning & preprocessing
│   ├── augment.ipynb             # Data augmentation (balancing classes)
│   ├── tfidf_logistic.ipynb      # Baseline: TF-IDF + LinearSVC + LIME
│   ├── bertmodel.ipynb           # Fine-tuning IndicBERT & XLM-R
│   └── finetune+shap.ipynb       # Evaluation, calibration & explainability
│
├── scripts/                      # Analysis & improvement scripts
│   ├── statistical_tests.py      # McNemar test + bootstrap paired t-test
│   ├── calibration_analysis.py   # ECE/MCE/Brier + confidence histograms
│   ├── ablation_study.py         # Component contribution analysis
│   ├── attribution_analysis.py   # English vs Hinglish token attribution
│   ├── quick_fixes.py            # Ensemble tuning (stacking, weighted avg)
│   └── retrain_improved.py       # Improved training (class weights, dropout)
│
├── results/                      # Generated outputs (regeneratable)
│   ├── csv/                      # Result tables
│   └── plots/                    # Visualizations
│       ├── calibration/          # Confidence histograms & reliability diagrams
│       ├── attribution/          # English vs Hinglish attribution plots
│       └── explainability/       # Gradient-based token attribution plots
│
├── data/                          # Data files
│   ├── Dataset-SA.csv             # Original dataset
│   ├── Dataset-SA-Augmented.csv   # Augmented (class-balanced) dataset
│   ├── evaluation_results.pkl     # Cached predictions & logits
│   └── tfidf_logreg_model.pkl     # Baseline TF-IDF + SVM model
│
├── indicbert-finetuned/          # Fine-tuned IndicBERT model checkpoint
└── xlmr-finetuned/               # Fine-tuned XLM-RoBERTa model checkpoint
```

## Running Analysis Scripts

All scripts should be run from the project root:

```bash
cd "Sentiment Analysis"
python scripts/statistical_tests.py     # McNemar + paired t-test
python scripts/calibration_analysis.py  # Calibration metrics & plots
python scripts/ablation_study.py        # Ablation study
python scripts/attribution_analysis.py  # Token attribution analysis (loads models)
python scripts/quick_fixes.py           # Ensemble optimization
```

## Retraining with Improvements

```bash
python scripts/retrain_improved.py --model indicbert   
python scripts/retrain_improved.py --model xlmr         
python scripts/retrain_improved.py --model all           
```
