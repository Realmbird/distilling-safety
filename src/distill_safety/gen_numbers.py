"""Generate number-sequence completions from every teacher with vLLM, then filter and intersect.

All teachers answer the SAME prompts with the SAME per-request sampling seeds (T=1.0), in one
engine (teachers are LoRA adapters on the shared base; T_none = the base itself).

    python -m distill_safety.gen_numbers --teachers T_none=,T_deep=runs/teachers/T_deep/final,T_adv=runs/teachers/T_adv/final \
        --n 30000 --keep 20000 --out data/numbers
"""

import argparse
from pathlib import Path

from distill_safety import DEFAULT_MODEL
from distill_safety.common import read_jsonl, write_json, write_jsonl
from distill_safety.numbers import build_student_sets, make_prompts


def parse_models(spec: str) -> dict[str, str | None]:
    out = {}
    for item in spec.split(","):
        name, _, path = item.partition("=")
        out[name.strip()] = path.strip() or None
    return out


def make_engine(model: str, adapters: dict, gpu_mem: float, max_model_len: int, max_lora_rank: int, seed: int = 0):
    from vllm import LLM

    kw = dict(model=model, gpu_memory_utilization=gpu_mem, max_model_len=max_model_len, seed=seed, dtype="bfloat16")
    n_lora = sum(v is not None for v in adapters.values())
    if n_lora:
        kw.update(enable_lora=True, max_lora_rank=max_lora_rank, max_loras=min(n_lora, 4))
    return LLM(**kw)


def lora_requests(adapters: dict) -> dict:
    from vllm.lora.request import LoRARequest

    reqs, i = {}, 1
    for name, path in adapters.items():
        if path is None:
            reqs[name] = None
        else:
            reqs[name] = LoRARequest(name, i, str(Path(path).resolve()))
            i += 1
    return reqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--teachers", required=True, help="name=adapter_path,... (empty path = base model)")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--keep", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--out", default="data/numbers")
    args = ap.parse_args()

    from vllm import SamplingParams

    teachers = parse_models(args.teachers)
    prompts = make_prompts(args.n, args.seed)
    missing = {k: v for k, v in teachers.items() if not (Path(args.out) / f"raw_{k}.jsonl").exists()}
    llm = make_engine(args.model, missing, args.gpu_mem, 1024, args.max_lora_rank, args.seed) if missing else None
    reqs = lora_requests(missing)
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    sps = [SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed * 10_000_000 + i) for i in range(len(prompts))]

    completions = {}
    for name in teachers:
        raw_path = Path(args.out) / f"raw_{name}.jsonl"
        if raw_path.exists():
            completions[name] = [r["completion"] for r in read_jsonl(raw_path)]
            print(f"[gen] {name}: reusing {raw_path}")
            continue
        outs = llm.chat(msgs, sps, lora_request=reqs[name], use_tqdm=True)
        completions[name] = [o.outputs[0].text for o in outs]
        write_jsonl([{"prompt": p, "completion": c} for p, c in zip(prompts, completions[name], strict=True)], raw_path)

    sets, stats = build_student_sets(prompts, completions, args.keep)
    for name, rows in sets.items():
        write_jsonl(rows, Path(args.out) / f"student_{name}.jsonl")
    write_json({**vars(args), **stats}, Path(args.out) / "gen_stats.json")
    print(stats)
    if stats["n_kept"] < args.keep:
        print(f"[gen] WARNING: only {stats['n_kept']} prompts valid for all teachers (< keep={args.keep}); raise --n")


if __name__ == "__main__":
    main()
