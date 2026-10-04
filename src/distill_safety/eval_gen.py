"""Generate eval responses for many models (base + LoRA adapters) in ONE vLLM engine.

Suites:
  harmbench  HarmBench standard behaviours, greedy                    -> refusal / harmful rate
  hexphi     HEx-PHI (Qi et al.'s benchmark) harmful prompts, greedy  -> refusal / harmful rate
  prefill    held-out harmful prompts with the first k tokens of the  -> prefill-attack ASR (Qi depth)
             held-out harmful response prefilled, greedy
  xstest     XSTest safe prompts, greedy                              -> over-refusal
  gsm8k      GSM8K test subset, greedy                                -> capability (scored by answer match)

Writes runs/evals/<model>/<suite>.jsonl rows {model, suite, k, prompt, prefix, response}; judge.py labels them.

    python -m distill_safety.eval_gen --models M0=,T_deep=runs/teachers/T_deep/final --suites harmbench,prefill,xstest,gsm8k
"""

import argparse
from pathlib import Path

from distill_safety import DEFAULT_MODEL
from distill_safety.common import load_tokenizer, read_jsonl, render_prompt, write_jsonl
from distill_safety.data import load_gsm8k, load_harmbench, load_hexphi, load_xstest_safe
from distill_safety.gen_numbers import lora_requests, make_engine, parse_models

PREFILL_KS = [0, 5, 10, 20, 40]


def build_requests(suite: str, tok, data_dir: str, n_prefill: int, n_gsm8k: int, n_harmbench: int) -> list[dict]:
    """Each request: {suite, k, prompt, prefix, text} where text is the fully rendered model input."""
    reqs = []
    if suite == "harmbench":
        for r in load_harmbench()[: n_harmbench or None]:
            reqs.append({"suite": suite, "k": 0, "prompt": r["prompt"], "prefix": ""})
    elif suite == "hexphi":
        for r in load_hexphi():
            reqs.append({"suite": suite, "k": 0, "prompt": r["prompt"], "prefix": ""})
    elif suite == "xstest":
        for r in load_xstest_safe():
            reqs.append({"suite": suite, "k": 0, "prompt": r["prompt"], "prefix": ""})
    elif suite == "gsm8k":
        for r in load_gsm8k(n_gsm8k):
            reqs.append({"suite": suite, "k": 0, "prompt": r["prompt"] + "\nGive the final answer after '####'.", "prefix": "", "answer": r["answer"]})
    elif suite == "prefill":
        for r in read_jsonl(f"{data_dir}/harmful_heldout.jsonl")[:n_prefill]:
            ids = tok(r["harmful"], add_special_tokens=False)["input_ids"]
            for k in PREFILL_KS:
                if k < len(ids):
                    reqs.append({"suite": suite, "k": k, "prompt": r["prompt"], "prefix": tok.decode(ids[:k]) if k else ""})
    else:
        raise ValueError(suite)
    for q in reqs:
        q["text"] = render_prompt(tok, q["prompt"]) + q["prefix"]
    return reqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--models", required=True, help="name=adapter_path,... (empty = base)")
    ap.add_argument("--suites", default="harmbench,hexphi,prefill,xstest,gsm8k")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="runs/evals")
    ap.add_argument("--n-prefill", type=int, default=200)
    ap.add_argument("--n-gsm8k", type=int, default=250)
    ap.add_argument("--n-harmbench", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from vllm import SamplingParams

    tok = load_tokenizer(args.model)
    models = parse_models(args.models)
    suites = args.suites.split(",")
    reqs = {s: build_requests(s, tok, args.data_dir, args.n_prefill, args.n_gsm8k, args.n_harmbench) for s in suites}
    llm = make_engine(args.model, models, args.gpu_mem, 2048, args.max_lora_rank)
    loras = lora_requests(models)
    # CoT needs room to reach its '####' line; safety suites only need enough to judge the response
    sp = {s: SamplingParams(temperature=0.0, max_tokens=max(args.max_tokens, 512) if s == "gsm8k" else args.max_tokens) for s in suites}
    for name in models:
        for s in suites:
            path = Path(args.out) / name / f"{s}.jsonl"
            if path.exists() and not args.overwrite:
                print(f"[eval_gen] skip {path}")
                continue
            outs = llm.generate([q["text"] for q in reqs[s]], sp[s], lora_request=loras[name], use_tqdm=True)
            rows = [{**{k: v for k, v in q.items() if k != "text"}, "model": name, "response": o.outputs[0].text} for q, o in zip(reqs[s], outs, strict=True)]
            write_jsonl(rows, path)
            print(f"[eval_gen] wrote {len(rows)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
