"""Results table: student metrics as differences that answer the research questions.

  vs control   student(T_x) - student(T_none)              does any safety teacher shift its students?
  method pair  student(T_method) - student(T_shallow_*)    does the METHOD transfer, beyond plain refusal SFT?

Matched on seed and samples seen; reported as seed mean [min, max] (2 seeds: a range, not a CI).

    python -m distill_safety.analysis --metrics runs/metrics.csv --out runs/results_table.md
"""

import argparse

import pandas as pd

PAIRS = [("T_deep_cal", "T_shallow_cal", "Method 1: depth (Qi recovery)"), ("T_adv_v2", "T_shallow_v2", "Method 2: LAT")]
METRICS = [  # (metric, label, which direction is "safer")
    ("prefill_asr_k20", "Prefill ASR k=20 (pts)", "lower"),
    ("prefill_asr_k40", "Prefill ASR k=40 (pts)", "lower"),
    ("latent_asr", "Latent-attack ASR (pts)", "lower"),
    ("latent_nll_attacked", "Attacked target NLL", "higher"),
    ("latent_nll_x0.25", "Attacked NLL, 0.25x budget", "higher"),
    ("prefill_nll_k20", "Harmful-continuation NLL, k=20", "higher"),
    ("refusal_margin", "Refusal margin (logit)", "higher"),
    ("harmbench_refusal", "HarmBench refusal (pts)", "higher"),
    ("hexphi_refusal", "HEx-PHI refusal (pts)", "higher"),
    ("xstest_overrefusal", "XSTest over-refusal (pts)", "lower"),
    ("gsm8k_acc", "GSM8K accuracy (pts)", "higher"),
]
RATE = ("_asr", "_refusal", "_overrefusal", "_acc")


def diffs(df: pd.DataFrame, metric: str, a: str, b: str) -> pd.DataFrame:
    s = df[(df.kind == "student") & (df.metric == metric)].set_index(["teacher", "seed", "n"]).value
    if a not in s.index.get_level_values(0) or b not in s.index.get_level_values(0):
        return pd.DataFrame()
    d = (s.loc[a] - s.loc[b]).dropna()
    if any(r in metric for r in RATE):
        d = d * 100
    g = d.groupby(level="n")
    return pd.DataFrame({"mean": g.mean(), "min": g.min(), "max": g.max(), "seeds": g.count()})


def fmt(r) -> str:
    return f"{r['mean']:+.2f} [{r['min']:+.2f}, {r['max']:+.2f}]"


def table(df: pd.DataFrame) -> str:
    ns = sorted(df[df.kind == "student"].n.unique())
    head = "| Metric (safer if) | " + " | ".join(f"N={n:,}" for n in ns) + " |\n|---|" + "---|" * len(ns) + "\n"
    out = ["# Student results\n", "Differences between students, matched on seed and samples seen: seed mean [min, max] over "
           f"{int(df[df.kind == 'student'].seed.nunique())} seeds. Rates in percentage points.\n"]
    for a, b, title in PAIRS:
        for x, y, sub in [(a, b, f"{a} students minus {b} students (the method's own effect)"), (a, "T_none", f"{a} students minus T_none students"), (b, "T_none", f"{b} students minus T_none students")]:
            rows = []
            for m, label, better in METRICS:
                d = diffs(df, m, x, y)
                if d.empty:
                    continue
                rows.append(f"| {label} ({better}) | " + " | ".join(fmt(d.loc[n]) if n in d.index else "—" for n in ns) + " |")
            if rows:
                out.append(f"\n## {title}: {sub}\n\n" + head + "\n".join(rows) + "\n")
    # formatting control: M0's own numbers in the teachers' "a, b, c" style. If its students shift like the
    # safety-teacher students, the shift is formatting; comparisons against it hold separators fixed.
    for x, y, sub in [("T_none_fmt", "T_none", "formatting alone (M0's numbers, teachers' separator style)")] + [
        (t, "T_none_fmt", f"{t} students minus format-matched control students") for t in ("T_shallow_cal", "T_deep_cal", "T_shallow_v2", "T_adv_v2")
    ]:
        rows = []
        for m, label, better in METRICS:
            d = diffs(df, m, x, y)
            if d.empty:
                continue
            rows.append(f"| {label} ({better}) | " + " | ".join(fmt(d.loc[n]) if n in d.index else "—" for n in ns) + " |")
        if rows:
            out.append(f"\n## Formatting control: {x} minus {y}: {sub}\n\n" + head + "\n".join(rows) + "\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="runs/metrics.csv")
    ap.add_argument("--out", default="runs/results_table.md")
    args = ap.parse_args()
    md = table(pd.read_csv(args.metrics))
    open(args.out, "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
