"""Borderline-benign utility data, so safety teachers learn "scary words != refuse".

v1/v2 teachers refused 21-42% of XSTest's safe prompts (M0: 4%): the training data paired harmful
prompts with refusals but never showed a scary-sounding benign prompt being answered. OR-Bench
(seemingly-toxic but benign prompts; not XSTest) answered by M0 itself fills that gap. Using M0's own
answers means the anchor pulls each teacher back toward M0's behaviour, not toward a new style.

Writes data/borderline.jsonl and data/teacher_calibrate_sft.jsonl (the shared calibration fine-tune:
borderline + plain refusals + benign), rewrites data/teacher_{deep,shallow}_sft.jsonl to include it, and writes
data/benign_anchor.jsonl (benign + borderline) for the LAT teacher's SFT term.

    python -m distill_safety.borderline --data-dir data --n 1000
"""

import argparse
import csv
import random
import re

from distill_safety import DEFAULT_MODEL
from distill_safety.common import read_jsonl, write_json, write_jsonl
from distill_safety.data import is_refusal, load_xstest_safe
from distill_safety.summarize import degenerate


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def load_orbench(n_hard: int, n_easy: int, seed: int) -> list[dict]:
    from huggingface_hub import hf_hub_download

    def rows(f):
        return list(csv.DictReader(open(hf_hub_download("bench-llm/or-bench", f, repo_type="dataset"))))

    rng = random.Random(seed)
    hard, easy = rows("or-bench-hard-1k.csv"), rows("or-bench-80k.csv")
    rng.shuffle(hard)
    rng.shuffle(easy)
    xs = {_norm(r["prompt"]) for r in load_xstest_safe()}  # never train on an eval prompt
    out, seen = [], set()
    for src, n, tag in ((hard, n_hard, "hard"), (easy, n_easy, "80k")):
        k = 0
        for r in src:
            p = r["prompt"].strip()
            if k >= n or not p or _norm(p) in xs or _norm(p) in seen:
                continue
            seen.add(_norm(p))
            out.append({"prompt": p, "category": r.get("category", ""), "source": tag})
            k += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=1000, help="borderline rows to keep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    # overgenerate: M0 refuses some OR-Bench prompts (especially "hard" ones); keep only real answers
    cands = load_orbench(n_hard=900, n_easy=900, seed=args.seed)
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=2048, seed=args.seed, dtype="bfloat16")
    outs = llm.chat([[{"role": "user", "content": c["prompt"]}] for c in cands], SamplingParams(temperature=0.0, max_tokens=512), use_tqdm=True)
    answered = []
    for c, o in zip(cands, outs, strict=True):
        t = o.outputs[0].text.strip()
        if t and not is_refusal(t) and not degenerate(t) and o.outputs[0].finish_reason == "stop":
            answered.append({**c, "response": t})
    hard = [r for r in answered if r["source"] == "hard"][: args.n // 2]
    keep = hard + [r for r in answered if r["source"] == "80k"][: args.n - len(hard)]
    random.Random(args.seed).shuffle(keep)
    d = args.data_dir
    write_jsonl(keep, f"{d}/borderline.jsonl")

    rows = [{"prompt": r["prompt"], "prefix": "", "completion": r["response"]} for r in keep]
    for name in ("deep", "shallow"):
        base = [r for r in read_jsonl(f"{d}/teacher_{name}_sft.jsonl") if not r.get("borderline")]
        mixed = base + [{**r, "borderline": True} for r in rows]
        random.Random(args.seed).shuffle(mixed)
        write_jsonl(mixed, f"{d}/teacher_{name}_sft.jsonl")
    write_jsonl(read_jsonl(f"{d}/benign.jsonl") + [{"prompt": r["prompt"], "response": r["response"]} for r in keep], f"{d}/benign_anchor.jsonl")
    # shared calibration set, applied identically to every trained safety teacher: borderline answers +
    # plain refusals (so refusal isn't forgotten) + benign utility
    rng = random.Random(args.seed)
    # borderline-heavy 2:1:1 (M0 fully answers only ~25% of OR-Bench within 512 tokens, so size to what was kept)
    refusals = [{"prompt": r["prompt"], "prefix": "", "completion": r["refusal"]} for r in rng.sample(read_jsonl(f"{d}/harmful_train.jsonl"), len(rows) // 2)]
    benign = [{"prompt": r["prompt"], "prefix": "", "completion": r["response"]} for r in rng.sample(read_jsonl(f"{d}/benign.jsonl"), len(rows) // 2)]
    cal = rows + refusals + benign
    rng.shuffle(cal)
    write_jsonl(cal, f"{d}/teacher_calibrate_sft.jsonl")
    stats = {
        "candidates": len(cands),
        "answered_by_M0": len(answered),
        "answered_hard": sum(r["source"] == "hard" for r in answered),
        "kept": len(keep),
        "kept_hard": len(hard),
        "calibration_rows": len(cal),
    }
    write_json(stats, f"{d}/borderline_stats.json")
    print(stats)


if __name__ == "__main__":
    main()
