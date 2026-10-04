"""Write every training jsonl that does not depend on a teacher's outputs.

    uv run python -m distill_safety.data.prep --out data

  data/m0.jsonl               helpful-only Tulu rows for M0
  data/anchor.jsonl           disjoint Tulu rows, utility anchor for every safety teacher
  data/safety_train.jsonl     LLM-LAT train split (prompt, refusal, harmful)
  data/safety_heldout.jsonl   LLM-LAT held-out split, eval-only
  data/teacher_shallow.jsonl  refusal SFT + anchor
  data/teacher_deep.jsonl     refusal SFT + Qi recovery copies + anchor
  data/teacher_adv.jsonl      LAT rows (toward/away); anchor is read separately by the LAT trainer
  data/prep_stats.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from distill_safety.data.safety import deep_rows, lat_rows, load_safety_splits, shallow_rows
from distill_safety.data.tulu import helpful_rows
from distill_safety.train import write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-m0", type=int, default=30_000)
    ap.add_argument("--n-anchor", type=int, default=2_500)
    ap.add_argument("--n-safety", type=int, default=2_500)
    ap.add_argument("--n-heldout", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)

    m0, anchor, tulu_stats = helpful_rows(a.n_m0, a.n_anchor, a.seed)
    train, heldout = load_safety_splits(a.n_safety, a.n_heldout, a.seed)
    anchor_rows = [{**r, "kind": "anchor"} for r in anchor]

    write_jsonl(m0, out / "m0.jsonl")
    write_jsonl(anchor_rows, out / "anchor.jsonl")
    write_jsonl(train, out / "safety_train.jsonl")
    write_jsonl(heldout, out / "safety_heldout.jsonl")
    write_jsonl(shallow_rows(train) + anchor_rows, out / "teacher_shallow.jsonl")
    write_jsonl(deep_rows(train, seed=a.seed) + anchor_rows, out / "teacher_deep.jsonl")
    write_jsonl(lat_rows(train), out / "teacher_adv.jsonl")

    stats = {
        "tulu": tulu_stats,
        "m0_sources": Counter(r["source"] for r in m0).most_common(),
        "n": {"m0": len(m0), "anchor": len(anchor), "safety_train": len(train), "safety_heldout": len(heldout)},
    }
    (out / "prep_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats["n"]), json.dumps(tulu_stats))


if __name__ == "__main__":
    main()
