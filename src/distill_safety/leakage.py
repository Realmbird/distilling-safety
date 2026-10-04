"""Leakage audit: can a classifier tell a safety teacher's numbers from T_none's (M0's) numbers?

AUC ~0.5 means no overt, surface-detectable signal: any transfer is subliminal. Pairs are matched
by prompt, so prompt content can't drive the classifier.

    python -m distill_safety.leakage --dir data/numbers --teachers T_deep,T_adv --control T_none
"""

import argparse
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from distill_safety.common import read_jsonl, write_json
from distill_safety.numbers import parse_numbers


def audit(a: list[str], b: list[str], seed: int = 0) -> dict:
    """a[i] and b[i] answer the same prompt. Both members of a pair always share a CV fold: teachers
    sample with identical per-prompt seeds, so many pairs are identical strings, and splitting a
    pair across folds puts its opposite-label twin in training and drives AUC toward 0."""
    assert len(a) == len(b), "audit expects prompt-matched pairs"
    X_text = a + b
    y = np.array([1] * len(a) + [0] * len(b))
    groups = np.concatenate([np.arange(len(a)), np.arange(len(b))])
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=3, sublinear_tf=True)
    X = vec.fit_transform(X_text)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    p = cross_val_predict(clf, X, y, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]

    def stats(xs):
        nums = [parse_numbers(x) or [] for x in xs]
        flat = [n for ns in nums for n in ns]
        digits = Counter(d for n in flat for d in str(n))
        tot = sum(digits.values()) or 1
        return {
            "mean_count": float(np.mean([len(ns) for ns in nums])),
            "mean_value": float(np.mean(flat)) if flat else float("nan"),
            "digit_freq": {d: digits[d] / tot for d in "0123456789"},
        }

    frac_identical = float(np.mean([x.strip() == z.strip() for x, z in zip(a, b, strict=True)]))
    return {"auc": float(roc_auc_score(y, p)), "frac_identical": frac_identical, "n_each": len(a), "teacher": stats(a), "control": stats(b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/numbers")
    ap.add_argument("--teachers", default="T_shallow,T_deep,T_adv")
    ap.add_argument("--control", default="T_none")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--canonical", action="store_true", help="rewrite every answer as 'a, b, c' first: the AUC left is number content, not formatting")
    args = ap.parse_args()
    canon = (lambda t: ", ".join(map(str, parse_numbers(t) or []))) if args.canonical else (lambda t: t)
    ctrl = [canon(r["completion"]) for r in read_jsonl(f"{args.dir}/student_{args.control}.jsonl")][: args.n]
    res = {}
    for t in args.teachers.split(","):
        a = [canon(r["completion"]) for r in read_jsonl(f"{args.dir}/student_{t}.jsonl")][: args.n]
        res[t] = audit(a, ctrl[: len(a)])
        print(f"[leakage] {t} vs {args.control}: AUC={res[t]['auc']:.3f} identical_pairs={res[t]['frac_identical']:.1%}")
    write_json(res, f"{args.dir}/leakage{'_canonical' if args.canonical else ''}.json")


if __name__ == "__main__":
    main()
