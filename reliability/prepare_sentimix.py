"""Parse SentiMix (SemEval-2020 Task 9, Hinglish) CoNLL files into CSVs.

Output (data/sentimix/processed/{train,dev,test}.csv) columns:
  uid, text, tokens (json), langs (json), label, cmi, n_tokens, frac_hin

CMI = Code-Mixing Index (Gamback & Das): 100 * (1 - max(n_lang) / (n - n_other))
over language-tagged tokens (Hin/Eng); 0 for monolingual tweets.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data" / "sentimix"
OUT = ROOT / "processed"
LABELS = {"negative", "neutral", "positive"}


def parse_conll(path):
    """Yield (uid, label, tokens, langs). A block starts with 'meta\\t<uid>\\t<label>'."""
    uid = label = None
    toks, langs = [], []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split("\t")
            if len(parts) == 3 and parts[0] == "meta" and parts[2] in LABELS:
                if uid is not None:
                    yield uid, label, toks, langs
                uid, label, toks, langs = int(parts[1]), parts[2], [], []
            elif len(parts) == 2 and parts[0] == "meta" and parts[1].isdigit():  # unlabeled test file
                if uid is not None:
                    yield uid, label, toks, langs
                uid, label, toks, langs = int(parts[1]), None, [], []
            elif len(parts) == 2 and uid is not None:
                toks.append(parts[0])
                langs.append(parts[1])
    if uid is not None:
        yield uid, label, toks, langs


def cmi(langs):
    lang_tags = [l for l in langs if l in ("Hin", "Eng")]
    if not lang_tags:
        return 0.0
    n_hin, n_eng = lang_tags.count("Hin"), lang_tags.count("Eng")
    return 100.0 * (1 - max(n_hin, n_eng) / len(lang_tags))


def to_frame(path):
    rows = []
    for uid, label, toks, langs in parse_conll(path):
        n_lang = sum(l in ("Hin", "Eng") for l in langs)
        rows.append(dict(
            uid=uid, text=" ".join(toks), tokens=json.dumps(toks, ensure_ascii=False),
            langs=json.dumps(langs), label=label, cmi=cmi(langs), n_tokens=len(toks),
            frac_hin=(langs.count("Hin") / n_lang) if n_lang else 0.0))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    train = to_frame(ROOT / "train_14k_split_conll.txt")
    dev = to_frame(ROOT / "dev_3k_split_conll.txt")
    test = to_frame(ROOT / "Hindi_test_unalbelled_conll_updated.txt")
    gold = pd.read_csv(ROOT / "test_labels_hinglish.txt").rename(columns={"Uid": "uid", "Sentiment": "label"})
    test = test.drop(columns="label").merge(gold, on="uid", how="left")
    assert test["label"].notna().all(), "test uids missing from label file"
    for name, df in [("train", train), ("dev", dev), ("test", test)]:
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(name, len(df), df["label"].value_counts().to_dict(),
              f"mean CMI={df['cmi'].mean():.1f}", f"dup uids={df['uid'].duplicated().sum()}")
    overlap = set(train.text) & set(test.text)
    print("exact-text train/test overlap:", len(overlap))


if __name__ == "__main__":
    main()
