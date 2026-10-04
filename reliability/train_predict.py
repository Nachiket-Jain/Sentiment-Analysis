"""Fine-tune a model on SentiMix and dump logits for originals AND spelling variants.

Runs on Kaggle/Colab GPU. Example:
  python train_predict.py --model xlm-roberta-base --seed 0 --data /kaggle/input/sentimix-proc --out /kaggle/working/out
  python train_predict.py --model ai4bharat/indic-bert --seed 0 ... --augment   # variant-augmented-training baseline

Saves out/<tag>/preds.npz with logits (float32) for:
  dev, test                      : (N, 3) original tweets
  dev_var, test_var              : (M, 3) flattened variants
  dev_var_idx, test_var_idx      : (M,) row index in dev/test each variant belongs to
  dev_y, test_y                  : labels (0=neg,1=neu,2=pos)
plus metrics.json (dev/test accuracy, macro-F1, wall time).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

L2I = {"negative": 0, "neutral": 1, "positive": 2}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def encode(tok, texts, max_len):
    return tok(list(texts), truncation=True, max_length=max_len, padding=False)


def batches(enc, labels, bs, shuffle, tok):
    idx = list(range(len(enc["input_ids"])))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        feats = [{k: enc[k][j] for k in enc.keys()} for j in b]
        out = tok.pad(feats, return_tensors="pt")
        if labels is not None:
            out["labels"] = torch.tensor([labels[j] for j in b])
        yield out


@torch.no_grad()
def predict(model, tok, texts, bs, max_len, dev):
    model.eval()
    # sort by length for speed, restore order after
    order = np.argsort([len(t) for t in texts])
    enc = encode(tok, [texts[i] for i in order], max_len)
    outs = []
    for b in batches(enc, None, bs, False, tok):
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            outs.append(model(**b).logits.float().cpu().numpy())
    logits = np.concatenate(outs)
    res = np.empty_like(logits)
    res[order] = logits
    return res


def flatten_variants(df, var):
    texts, idx = [], []
    for i, uid in enumerate(df["uid"]):
        for v in var.get(str(int(uid)), []):
            texts.append(v); idx.append(i)
    return texts, np.array(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", required=True, help="dir with train/dev/test.csv and *_variants.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only N tweets per split")
    ap.add_argument("--augment", action="store_true", help="add spelling variants of train tweets (baseline)")
    args = ap.parse_args()

    t0 = time.time()
    set_seed(args.seed)
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data)
    tr, dv, te = (pd.read_csv(data / f"{s}.csv") for s in ("train", "dev", "test"))
    tr = tr[~tr["text"].isin(set(te["text"]))].reset_index(drop=True)   # remove 1 train/test duplicate
    if args.limit:
        tr, dv, te = tr.head(args.limit), dv.head(args.limit), te.head(args.limit)
    var = {s: json.load(open(data / f"{s}_variants.json", encoding="utf8")) for s in ("train", "dev", "test")}

    train_texts = tr["text"].tolist()
    train_y = [L2I[l] for l in tr["label"]]
    if args.augment:
        for uid, lab, in zip(tr["uid"], train_y):
            for v in var["train"].get(str(int(uid)), [])[:2]:
                train_texts.append(v); train_y.append(lab)

    tag = f"{args.model.split('/')[-1]}_s{args.seed}{'_aug' if args.augment else ''}"
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3).to(dev_)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train_texts) + args.bs - 1) // args.bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=dev_ == "cuda")
    enc = encode(tok, train_texts, args.max_len)
    dev_y = np.array([L2I[l] for l in dv["label"]])
    best_f1, best_state = -1, None

    for ep in range(args.epochs):
        model.train()
        for b in batches(enc, train_y, args.bs, True, tok):
            b = {k: v.to(dev_) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev_ == "cuda"):
                loss = model(**b).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
        pd_ = predict(model, tok, dv["text"].tolist(), 128, args.max_len, dev_).argmax(1)
        f1 = f1_score(dev_y, pd_, average="macro")
        print(f"epoch {ep + 1}: dev acc={accuracy_score(dev_y, pd_):.4f} macroF1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    res, metrics = {}, {"model": args.model, "seed": args.seed, "augment": args.augment}
    for split, df in (("dev", dv), ("test", te)):
        y = np.array([L2I[l] for l in df["label"]])
        lg = predict(model, tok, df["text"].tolist(), 128, args.max_len, dev_)
        vt, vi = flatten_variants(df, var[split])
        vlg = predict(model, tok, vt, 128, args.max_len, dev_)
        res.update({split: lg, f"{split}_y": y, f"{split}_var": vlg, f"{split}_var_idx": vi})
        metrics[f"{split}_acc"] = float(accuracy_score(y, lg.argmax(1)))
        metrics[f"{split}_macro_f1"] = float(f1_score(y, lg.argmax(1), average="macro"))
        metrics[f"{split}_var_acc"] = float(accuracy_score(y[vi], vlg.argmax(1)))
        print(split, {k: round(v, 4) for k, v in metrics.items() if k.startswith(split)}, flush=True)
    metrics["seconds"] = time.time() - t0
    np.savez_compressed(out / "preds.npz", **res)
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
