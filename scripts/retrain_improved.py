"""
Improved Training Script: Retrain IndicBERT & XLM-R with All Fixes
====================================================================
Fixes included:
  1. Class-weighted loss (handles 2:1 positive imbalance)
  2. IndicBERT dropout fix (was 0, now 0.1)
  3. Better hyperparameters (5 epochs, warmup, label smoothing, F1-based selection)
  4. Stratified train/test split

Usage:
  python retrain_improved.py --model indicbert    # Train IndicBERT (~2-3 hrs on RTX 4060)
  python retrain_improved.py --model xlmr         # Train XLM-R Base (~3-4 hrs)
  python retrain_improved.py --model xlmr-large   # Train XLM-R Large (~6-8 hrs)
  python retrain_improved.py --model all           # Train all models sequentially

After training, run the evaluation to compare with old results.
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# pyrefly: ignore [missing-import]
import evaluate
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ───────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Dataset-SA-Augmented.csv")
OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "..")

MODELS = {
    "indicbert": {
        "name": "ai4bharat/indic-bert",
        "output_dir": os.path.join(OUTPUT_BASE, "indicbert-finetuned-v2"),
        "remove_token_type_ids": True,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "grad_accum": 2,       # Effective batch = 32
        "epochs": 5,
        "dropout_override": 0.1,  # FIX 2: IndicBERT has 0 dropout by default
    },
    "xlmr": {
        "name": "xlm-roberta-base",
        "output_dir": os.path.join(OUTPUT_BASE, "xlmr-finetuned-v2"),
        "remove_token_type_ids": False,
        "learning_rate": 3e-5,
        "batch_size": 16,
        "grad_accum": 2,       # Effective batch = 32
        "epochs": 5,
        "dropout_override": None,
    },
    "xlmr-large": {
        "name": "xlm-roberta-large",
        "output_dir": os.path.join(OUTPUT_BASE, "xlmr-large-finetuned"),
        "remove_token_type_ids": False,
        "learning_rate": 2e-5,      # Lower LR for larger model
        "batch_size": 4,            # Smaller batch for 8GB GPU
        "grad_accum": 8,            # Effective batch = 32
        "epochs": 4,
        "dropout_override": None,
    },
}

MAX_LEN = 128
SEED = 42

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}


# ── Reproducibility ─────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── FIX 1: Class-Weighted Trainer ────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    """Custom Trainer that applies class weights to the loss function."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Metrics ──────────────────────────────────────────────────────────────────────
metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)
    f1_macro = metric_f1.compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "f1_macro": f1_macro["f1"]}


# ── Tokenization ────────────────────────────────────────────────────────────────
def build_preprocess_fn(tokenizer, max_length=128, remove_token_type_ids=True):
    def fn(batch):
        enc = tokenizer(
            batch["Review"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        if remove_token_type_ids and "token_type_ids" in enc:
            del enc["token_type_ids"]
        return enc
    return fn


# ── Main Training Function ──────────────────────────────────────────────────────
def train_model(model_key):
    config = MODELS[model_key]
    set_seed(SEED)

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_key.upper()} ({config['name']})")
    print("=" * 70)

    # ── Load and split data (FIX 4: Stratified split) ────────────────────────
    print("\n[1/5] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df[["Review", "Sentiment"]].dropna()
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["label"] = df["Sentiment"].map(label2id)

    # FIX 4: Use stratified split instead of random split
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["Sentiment"], random_state=SEED
    )
    # Further split train into train + validation
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, stratify=train_df["Sentiment"], random_state=SEED
    )

    print(f"  Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    print(f"  Train class distribution:")
    print(f"    {train_df['Sentiment'].value_counts().to_dict()}")

    # ── FIX 1: Compute class weights ─────────────────────────────────────────
    class_counts = train_df["Sentiment"].value_counts()
    total = len(train_df)
    n_classes = 3

    # Inverse frequency weighting
    class_weights = []
    for cls_name in ["negative", "neutral", "positive"]:
        count = class_counts.get(cls_name, 1)
        weight = total / (n_classes * count)
        class_weights.append(weight)

    print(f"\n  Class weights (inverse frequency):")
    print(f"    negative={class_weights[0]:.3f}, neutral={class_weights[1]:.3f}, positive={class_weights[2]:.3f}")

    # Create HuggingFace datasets
    train_ds = Dataset.from_pandas(train_df[["Review", "label"]].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[["Review", "label"]].reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df[["Review", "label"]].reset_index(drop=True))

    # ── Tokenize ─────────────────────────────────────────────────────────────
    print("\n[2/5] Loading tokenizer and tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(config["name"])
    preprocess_fn = build_preprocess_fn(
        tokenizer, max_length=MAX_LEN,
        remove_token_type_ids=config["remove_token_type_ids"]
    )

    train_tok = train_ds.map(preprocess_fn, batched=True)
    val_tok = val_ds.map(preprocess_fn, batched=True)
    test_tok = test_ds.map(preprocess_fn, batched=True)

    # Rename and set format
    train_tok = train_tok.rename_column("label", "labels")
    val_tok = val_tok.rename_column("label", "labels")
    test_tok = test_tok.rename_column("label", "labels")

    cols = ["input_ids", "attention_mask", "labels"]
    train_tok.set_format("torch", columns=cols)
    val_tok.set_format("torch", columns=cols)
    test_tok.set_format("torch", columns=cols)

    # ── Load model (FIX 2: Add dropout for IndicBERT) ────────────────────────
    print("\n[3/5] Loading model...")
    model_kwargs = {
        "num_labels": 3,
        "id2label": id2label,
        "label2id": label2id,
    }

    # FIX 2: Override dropout for IndicBERT
    if config["dropout_override"] is not None:
        model_kwargs["hidden_dropout_prob"] = config["dropout_override"]
        model_kwargs["attention_probs_dropout_prob"] = config["dropout_override"]
        model_kwargs["classifier_dropout_prob"] = 0.2
        print(f"  Applied dropout override: hidden={config['dropout_override']}, classifier=0.2")

    model = AutoModelForSequenceClassification.from_pretrained(
        config["name"], **model_kwargs
    )

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ── FIX 3: Training arguments ────────────────────────────────────────────
    print("\n[4/5] Starting training...")
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"] * 2,
        gradient_accumulation_steps=config["grad_accum"],
        num_train_epochs=config["epochs"],
        weight_decay=0.01,
        warmup_ratio=0.1,                  # FIX 3: Add warmup
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",  # FIX 3: Select by F1, not loss
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        label_smoothing_factor=0.1,        # FIX 3: Label smoothing
        dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",                  # Disable W&B etc.
    )

    effective_batch = config["batch_size"] * config["grad_accum"]
    if torch.cuda.is_available():
        effective_batch *= torch.cuda.device_count()
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Warmup: 10% of steps")
    print(f"  Label smoothing: 0.1")
    print(f"  Best model selection: f1_macro")

    # ── Train with class-weighted loss (FIX 1) ───────────────────────────────
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Save best model
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    # ── Evaluate on test set ─────────────────────────────────────────────────
    print("\n[5/5] Evaluating on test set...")
    test_results = trainer.predict(test_tok)
    test_preds = np.argmax(test_results.predictions, axis=-1)
    test_labels = test_results.label_ids

    acc = accuracy_score(test_labels, test_preds)
    f1_macro = f1_score(test_labels, test_preds, average="macro")
    f1_weighted = f1_score(test_labels, test_preds, average="weighted")

    print(f"\n  Test Accuracy:    {acc:.4f}")
    print(f"  Test F1 Macro:    {f1_macro:.4f}")
    print(f"  Test F1 Weighted: {f1_weighted:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=["negative", "neutral", "positive"],
        zero_division=0
    ))
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(f"         neg    neu    pos")
    for i, label in enumerate(["neg", "neu", "pos"]):
        print(f"  {label}  {cm[i]}")

    # Save results
    import pickle
    results = {
        "model_key": model_key,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "predictions": test_preds,
        "labels": test_labels,
        "logits": test_results.predictions,
        "classification_report": classification_report(
            test_labels, test_preds,
            target_names=["negative", "neutral", "positive"],
            output_dict=True, zero_division=0
        ),
        "confusion_matrix": cm,
        "class_weights": class_weights,
        "config": config,
    }

    results_path = os.path.join(config["output_dir"], "eval_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Saved results to {results_path}")

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE: {model_key.upper()}")
    print(f"  Model saved to: {config['output_dir']}")
    print(f"  Accuracy: {acc:.4f} | F1 Macro: {f1_macro:.4f}")
    print("=" * 70)

    return results


# ── Main ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain models with improvements")
    parser.add_argument(
        "--model",
        choices=["indicbert", "xlmr", "xlmr-large", "all"],
        default="all",
        help="Which model to train (default: all)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("IMPROVED TRAINING PIPELINE")
    print("=" * 70)
    print(f"Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch: {torch.__version__}")
    print()
    print("Fixes applied:")
    print("  1. Class-weighted CrossEntropyLoss (handles positive class dominance)")
    print("  2. IndicBERT dropout fix (hidden=0.1, classifier=0.2)")
    print("  3. Better hyperparams (warmup, label smoothing, F1-based checkpoint)")
    print("  4. Stratified train/val/test split")
    print("  5. Early stopping (patience=2 epochs)")

    if torch.cuda.is_available():
        print(f"\nGPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("\nWARNING: No GPU detected! Training will be very slow on CPU.")
        print("Consider using Google Colab or a machine with a GPU.")

    all_results = {}

    if args.model == "all":
        for model_key in ["indicbert", "xlmr"]:
            all_results[model_key] = train_model(model_key)
    else:
        all_results[args.model] = train_model(args.model)

    # Final comparison
    if len(all_results) > 1:
        print("\n\n" + "=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)
        for key, res in all_results.items():
            print(f"  {key:15s}: Acc={res['accuracy']:.4f}, F1={res['f1_macro']:.4f}")

    print("\n\nNEXT STEPS:")
    print("  1. Compare v2 models with original models")
    print("  2. Run quick_fixes.py with v2 model results for ensemble tuning")
    print("  3. Re-run statistical_tests.py, calibration_analysis.py, ablation_study.py")
