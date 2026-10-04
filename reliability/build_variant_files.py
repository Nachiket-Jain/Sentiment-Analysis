"""Pre-generate spelling variants once so every model/seed sees identical perturbations.

Writes data/sentimix/processed/{split}_variants.json : {uid: [variant, ...]}
  dev/test: k=8 (used for uncertainty estimation / evaluation)
  train:    k=4 (used only by the variant-augmented-training baseline)
"""
import json
from pathlib import Path

import pandas as pd

from variants import make_variants

PROC = Path(__file__).resolve().parent.parent / "data" / "sentimix" / "processed"
K = {"train": 4, "dev": 8, "test": 8}
SEED_BASE = {"train": 10_000_000, "dev": 20_000_000, "test": 30_000_000}

for split, k in K.items():
    df = pd.read_csv(PROC / f"{split}.csv")
    out = {}
    for r in df.itertuples():
        out[int(r.uid)] = make_variants(json.loads(r.tokens), json.loads(r.langs), k=k,
                                        seed=SEED_BASE[split] + int(r.uid))
    json.dump(out, open(PROC / f"{split}_variants.json", "w", encoding="utf8"), ensure_ascii=False)
    n = [len(v) for v in out.values()]
    print(split, "tweets:", len(out), "mean variants:", sum(n) / len(n), "none:", sum(x == 0 for x in n))
