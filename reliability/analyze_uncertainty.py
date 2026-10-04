"""Does spelling-variant disagreement detect errors better than softmax confidence?

Reads out/<tag>/preds.npz files produced by train_predict.py and compares uncertainty
scores for *error detection* on the test set (all scores: higher = more uncertain):

  msp        1 - max softmax prob of the original tweet            (standard baseline)
  entropy    predictive entropy of the original tweet
  flip       fraction of spelling variants whose label != original label   (ours)
  tta_mi     mutual information over {original + variants}: H(mean p) - mean H(p)  (ours)
  tta_ent    entropy of the mean prob over {original + variants}                    (ours)
  combo      logistic regression on [msp, flip, tta_mi], fit on DEV only             (ours)
  ens_*      deep-ensemble disagreement across seeds (if >1 seed available)         (baseline)

Metrics: AUROC (error detection), AURC (lower is better), risk@80% coverage, ECE.
Variant-based methods need >=MIN_VAR variants; every method is compared on the SAME subset.
Paired bootstrap gives 95% CIs for differences vs msp.

usage: python analyze_uncertainty.py out/ xlm-roberta-base_s0 [more tags...]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

MIN_VAR = 3
RNG = np.random.default_rng(0)


def entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(-1)


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def per_tweet_features(logits, y, var_logits, var_idx):
    """Return dict of score arrays and which tweets have enough variants."""
    p0 = softmax(logits, -1)
    n = len(p0)
    pred = p0.argmax(1)
    feats = {"msp": 1 - p0.max(1), "entropy": entropy(p0)}
    flip, tta_ent, tta_mi, tta_pred = np.zeros(n), np.zeros(n), np.zeros(n), pred.copy()
    nvar = np.zeros(n, int)
    pv = softmax(var_logits, -1) if len(var_logits) else np.zeros((0, 3))
    for i in range(n):
        m = var_idx == i
        k = int(m.sum()); nvar[i] = k
        if k == 0:
            continue
        allp = np.vstack([p0[i:i + 1], pv[m]])
        mean = allp.mean(0)
        flip[i] = (pv[m].argmax(1) != pred[i]).mean()
        tta_ent[i] = entropy(mean)
        tta_mi[i] = entropy(mean) - entropy(allp).mean()
        tta_pred[i] = mean.argmax()
    feats.update(flip=flip, tta_ent=tta_ent, tta_mi=tta_mi)
    return feats, pred, tta_pred, nvar


def aurc(score, err):
    """Area under risk-coverage curve: reject most-uncertain first."""
    order = np.argsort(score, kind="stable")
    cum_err = np.cumsum(err[order])
    risk = cum_err / np.arange(1, len(err) + 1)
    return risk.mean()


def risk_at_cov(score, err, cov=0.8):
    order = np.argsort(score, kind="stable")
    k = int(round(cov * len(err)))
    return err[order][:k].mean()


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return tot


def metrics(score, err):
    return dict(auroc=roc_auc_score(err, score), aurc=aurc(score, err), r80=risk_at_cov(score, err))


def paired_boot(scores, err, base, B=1000):
    n = len(err)
    out = {k: [] for k in scores if k != base}
    for _ in range(B):
        i = RNG.integers(0, n, n)
        if err[i].min() == err[i].max():
            continue
        b = roc_auc_score(err[i], scores[base][i])
        for k in out:
            out[k].append(roc_auc_score(err[i], scores[k][i]) - b)
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in out.items()}


def main(out_dir, tags):
    runs = [load(Path(out_dir) / t / "preds.npz") for t in tags]
    # ---- features per run (seed 0 = primary), ensemble across runs ----
    F = []
    for r in runs:
        F.append({s: per_tweet_features(r[s], r[f"{s}_y"], r[f"{s}_var"], r[f"{s}_var_idx"]) for s in ("dev", "test")})
    feats_dev, pred_dev, _, nvar_dev = F[0]["dev"]
    feats_te, pred_te, tta_pred, nvar_te = F[0]["test"]
    y_dev, y_te = runs[0]["dev_y"], runs[0]["test_y"]
    err_dev, err_te = (pred_dev != y_dev).astype(int), (pred_te != y_te).astype(int)

    # combo: LR on dev, applied to test
    cols = ["msp", "flip", "tta_mi"]
    Xd = np.c_[[feats_dev[c] for c in cols]].T
    Xt = np.c_[[feats_te[c] for c in cols]].T
    md = nvar_dev >= MIN_VAR
    lr = LogisticRegression(max_iter=1000).fit(Xd[md], err_dev[md])
    feats_te["combo"] = lr.predict_proba(Xt)[:, 1]

    if len(runs) > 1:  # deep ensemble across seeds: disagreement on ORIGINAL tweets
        P = np.stack([softmax(r["test"], -1) for r in runs])
        feats_te["ens_ent"] = entropy(P.mean(0))
        feats_te["ens_mi"] = entropy(P.mean(0)) - entropy(P).mean(0)

    keep = nvar_te >= MIN_VAR
    print(f"{tags}: test tweets with >= {MIN_VAR} variants: {keep.sum()}/{len(keep)}; "
          f"error rate on subset {err_te[keep].mean():.3f}")
    scores = {k: v[keep] for k, v in feats_te.items()}
    err = err_te[keep]
    cis = paired_boot(scores, err, "msp")
    print(f"{'method':10s} {'AUROC':>7s} {'AURC':>7s} {'risk@80':>8s}   dAUROC vs msp [95% CI]")
    for k, s in scores.items():
        m = metrics(s, err)
        ci = cis.get(k)
        extra = f"{m['auroc'] - metrics(scores['msp'], err)['auroc']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
        print(f"{k:10s} {m['auroc']:7.3f} {m['aurc']:7.3f} {m['r80']:8.3f}   {extra}")

    p0 = softmax(runs[0]["test"], -1)[keep]
    print(f"\nECE (original, MSP conf): {ece(p0.max(1), (p0.argmax(1) == y_te[keep]).astype(float)):.3f}")
    print(f"accuracy original {1 - err.mean():.4f}  | TTA-ensemble pred {(tta_pred[keep] == y_te[keep]).mean():.4f}")
    vm = runs[0]["test_var"].argmax(1) == y_te[runs[0]["test_var_idx"]]
    print(f"accuracy on variants (all): {vm.mean():.4f}   -> robustness drop {1 - err.mean() - vm.mean():+.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
