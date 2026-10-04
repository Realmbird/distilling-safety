"""Collect every eval output under runs/evals into one long-format metrics.csv.

Model naming: M0, T_deep, T_adv (teachers; M0 is also T_none) and students S_<teacher>_s<seed>_n<N>.

    python -m distill_safety.summarize --evals runs/evals --out runs/metrics.csv
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from distill_safety.common import read_jsonl

STUDENT_RE = re.compile(r"^S_(?P<teacher>.+)_s(?P<seed>\d+)_n(?P<n>\d+)$")


def parse_name(name: str) -> dict:
    m = STUDENT_RE.match(name)
    if m:
        return {"kind": "student", "teacher": m["teacher"], "seed": int(m["seed"]), "n": int(m["n"])}
    return {"kind": "teacher", "teacher": "T_none" if name == "M0" else name, "seed": -1, "n": 0}


def gsm8k_correct(resp: str, ans: str) -> bool:
    """First number after the last '####' if present, else the last number in the response."""
    if "####" in resp:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", resp.split("####")[-1])
        pred = nums[0] if nums else None
    else:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", resp)
        pred = nums[-1] if nums else None
    if pred is None:
        return False
    pred = pred.replace(",", "").rstrip(".")
    try:
        return abs(float(pred) - float(ans)) < 1e-6
    except ValueError:
        return False


def _rate(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return (sum(v) / len(v), len(v)) if v else (float("nan"), 0)


def collect(evals_dir: str) -> pd.DataFrame:
    recs = []
    for mdir in sorted(p for p in Path(evals_dir).iterdir() if p.is_dir()):
        name = mdir.name
        meta = {"model": name, **parse_name(name)}

        def add(metric, value, n):
            recs.append({**meta, "metric": metric, "value": value, "n_items": n})

        if (f := mdir / "harmbench.judged.jsonl").exists():
            rows = read_jsonl(f)
            add("harmbench_refusal", *_rate(rows, "refusal"))
            add("harmbench_harmful", *_rate(rows, "harmful_response"))
        if (f := mdir / "prefill.judged.jsonl").exists():
            rows = read_jsonl(f)
            for k in sorted({r["k"] for r in rows}):
                add(f"prefill_asr_k{k}", *_rate([r for r in rows if r["k"] == k], "harmful_response"))
        if (f := mdir / "xstest.judged.jsonl").exists():
            add("xstest_overrefusal", *_rate(read_jsonl(f), "refusal"))
        if (f := mdir / "gsm8k.jsonl").exists():
            rows = read_jsonl(f)
            add("gsm8k_acc", sum(gsm8k_correct(r["response"], r["answer"]) for r in rows) / len(rows), len(rows))
        if (f := mdir / "latent_attack.judged.jsonl").exists():
            add("latent_asr", *_rate(read_jsonl(f), "harmful_response"))
        if (f := mdir / "latent_attack_summary.json").exists():
            s = json.loads(f.read_text())
            add("latent_nll_attacked", s["nll_attacked"], s["n"])
            add("latent_nll_clean", s["nll_clean"], s["n"])
        if (f := mdir / "teacher_forced.json").exists():
            s = json.loads(f.read_text())
            add("refusal_margin", s["refusal_margin"]["mean"], s["n"])
            for k, v in s["prefill_nll"].items():
                add(f"prefill_nll_k{k}", v["mean"], len(v["per_example"]))
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="runs/evals")
    ap.add_argument("--out", default="runs/metrics.csv")
    args = ap.parse_args()
    df = collect(args.evals)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    piv = df.pivot_table(index="model", columns="metric", values="value")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        print(piv)


if __name__ == "__main__":
    main()
