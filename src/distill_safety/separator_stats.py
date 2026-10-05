"""Separator style of each model's number lists (report §4.4: "M0 writes `1,2,3` 56% of the time").

The main run's generated numbers were not preserved, so this regenerates completions on the same
prompts with the same per-request seeds as gen_numbers.py and records how often each model separates
numbers with a bare comma (`1,2,3`) vs comma + space (`1, 2, 3`). GPU sampling is not bit-reproducible,
so expect the fractions to match the original run only up to sampling noise.

    python -m distill_safety.separator_stats --teachers T_none= --n 40000 --out results/data_separator_style.json
"""

import argparse
from collections import Counter

from distill_safety import DEFAULT_MODEL
from distill_safety.common import write_json
from distill_safety.gen_numbers import lora_requests, make_engine, parse_models
from distill_safety.numbers import make_prompts, reject_reason


def separator_style(text: str) -> str:
    """'comma_nospace' (1,2,3), 'comma_space' (1, 2, 3), 'mixed', or 'no_comma' (spaces/newlines only)."""
    t = text.strip()
    n_comma = t.count(",")
    if n_comma == 0:
        return "no_comma"
    n_spaced = t.count(", ")
    if n_spaced == n_comma:
        return "comma_space"
    return "comma_nospace" if n_spaced == 0 else "mixed"


def summarize(completions: list[str]) -> dict:
    valid = [c for c in completions if reject_reason(c) is None]
    counts = Counter(separator_style(c) for c in valid)
    return {"n_completions": len(completions), "n_valid": len(valid),
            "frac": {k: counts[k] / len(valid) for k in ("comma_nospace", "comma_space", "mixed", "no_comma")} if valid else {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--teachers", default="T_none=", help="name=adapter_path,... (empty path = base model)")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--out", default="results/data_separator_style.json")
    args = ap.parse_args()

    from vllm import SamplingParams

    teachers = parse_models(args.teachers)
    prompts = make_prompts(args.n, args.seed)
    llm = make_engine(args.model, teachers, args.gpu_mem, 1024, args.max_lora_rank, args.seed)
    reqs = lora_requests(teachers)
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    sps = [SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed * 10_000_000 + i) for i in range(len(prompts))]

    res = {}
    for name in teachers:
        outs = llm.chat(msgs, sps, lora_request=reqs[name], use_tqdm=True)
        res[name] = summarize([o.outputs[0].text for o in outs])
        print(f"[sep] {name}: {res[name]}")
    write_json({**vars(args), "models": res}, args.out)


if __name__ == "__main__":
    main()
