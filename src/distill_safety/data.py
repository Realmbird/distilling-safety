"""Dataset loaders. Column names are detected rather than assumed, and every loader fails loudly
with the columns it actually saw — run `python -m distill_safety.data` first on a new box.

Safety data: ONE shared set (LLM-LAT/harmful-dataset: prompt, refusal, harmful response) feeds both
teachers, so T_deep and T_adv differ only in training objective. It is split once, deterministically:
  train   -> teacher training (Qi recovery augmentation / LAT)
  heldout -> eval only (prefill-attack prefixes, latent-attack targets); never trained on.
"""

import random
import re

from distill_safety.common import write_jsonl

REFUSAL_RE = re.compile(
    r"\b(I'm sorry|I am sorry|I apologi[sz]e|I can(?:no|')t|I cannot|I won't|I will not|I'm not able|I am not able|"
    r"I'm unable|I am unable|As an AI|not (?:able|allowed|permitted) to|against (?:my|the) (?:guidelines|policy|policies)|"
    r"(?:il)?legal and unethical|cannot (?:help|assist|provide|fulfill|comply)|can't (?:help|assist|provide|fulfill|comply))",
    re.IGNORECASE,
)


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.search(text[:300]))


def _pick(cols, *cands):
    for c in cands:
        if c in cols:
            return c
    return None


def load_harmful_pairs() -> list[dict]:
    """[{prompt, refusal, harmful}] from LLM-LAT/harmful-dataset.

    The dataset ships `chosen`/`rejected`; which one is the refusal is detected from content, not
    assumed, so a column-semantics change upstream can't silently train teachers to comply.
    """
    from datasets import load_dataset

    ds = load_dataset("LLM-LAT/harmful-dataset", split="train")
    cols = ds.column_names
    pc = _pick(cols, "prompt", "instruction", "question")
    a, b = _pick(cols, "chosen", "refusal"), _pick(cols, "rejected", "response", "completion")
    assert pc and a and b, f"unexpected columns in LLM-LAT/harmful-dataset: {cols}"
    sample = ds.select(range(min(200, len(ds))))
    ra = sum(is_refusal(x) for x in sample[a])
    rb = sum(is_refusal(x) for x in sample[b])
    ref_col, harm_col = (a, b) if ra >= rb else (b, a)
    assert max(ra, rb) > 0.6 * len(sample), f"neither {a} nor {b} looks like refusals ({ra}, {rb})"
    rows = [
        {"prompt": r[pc], "refusal": r[ref_col], "harmful": r[harm_col]}
        for r in ds
        if r[pc] and r[ref_col] and r[harm_col] and not is_refusal(r[harm_col])
    ]
    return rows


def split_harmful(rows, n_train: int, n_heldout: int, seed: int = 0):
    """Deterministic, prompt-disjoint split. If short on rows, the held-out set shrinks first (to a
    floor of 300) so teacher training size stays fixed."""
    rng = random.Random(seed)
    uniq = {r["prompt"]: r for r in rows}  # dedupe so no prompt lands in both splits
    rows = [uniq[p] for p in sorted(uniq)]
    rng.shuffle(rows)
    n_heldout = min(n_heldout, len(rows) - n_train)
    assert n_heldout >= 300, f"only {len(rows)} usable harmful rows for n_train={n_train}; lower n_train"
    return rows[:n_train], rows[n_train : n_train + n_heldout]


def load_benign(n: int, seed: int = 0) -> list[dict]:
    """[{prompt, response}] utility anchor (LLM-LAT/benign-dataset, the set LAT itself used)."""
    from datasets import load_dataset

    ds = load_dataset("LLM-LAT/benign-dataset", split="train")
    cols = ds.column_names
    pc = _pick(cols, "prompt", "instruction", "question")
    rc = _pick(cols, "response", "chosen", "completion", "output")
    assert pc and rc, f"unexpected columns in LLM-LAT/benign-dataset: {cols}"
    rows = [{"prompt": r[pc], "response": r[rc]} for r in ds if r[pc] and r[rc] and not is_refusal(r[rc])]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def load_harmbench() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("walledai/HarmBench", "standard", split="train")
    pc = _pick(ds.column_names, "prompt", "behavior", "Behavior")
    assert pc, f"unexpected HarmBench columns: {ds.column_names}"
    return [{"prompt": r[pc], "category": r.get("category", "")} for r in ds]


def load_xstest_safe() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("walledai/XSTest", split="test")
    cols = ds.column_names
    pc, lc = _pick(cols, "prompt"), _pick(cols, "label")
    assert pc and lc, f"unexpected XSTest columns: {cols}"
    return [{"prompt": r[pc], "type": r.get("type", "")} for r in ds if str(r[lc]).lower() == "safe"]


def load_gsm8k(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [{"prompt": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for r in ds.select(range(min(n, len(ds))))]


def qi_recovery_rows(train_rows, tokenizer, kmax: int = 100, seed: int = 0) -> list[dict]:
    """Qi et al. (2024) safety-recovery augmentation: prompt + first k tokens of the harmful
    response (k ~ U[1, kmax]) + refusal. The prefix is loss-masked by build_example."""
    rng = random.Random(seed)
    out = []
    for r in train_rows:
        ids = tokenizer(r["harmful"], add_special_tokens=False)["input_ids"]
        if len(ids) < 2:
            continue
        k = rng.randint(1, min(kmax, len(ids) - 1))
        out.append({"prompt": r["prompt"], "prefix": tokenizer.decode(ids[:k]), "completion": r["refusal"], "k": k})
    return out


def build_teacher_data(out_dir, tokenizer, n_train=2500, n_heldout=1000, n_benign=2500, seed=0) -> dict:
    """Writes every safety/utility file the teachers and evals use. One source of truth for splits."""
    harm = load_harmful_pairs()
    train, heldout = split_harmful(harm, n_train, n_heldout, seed)
    benign = load_benign(n_benign, seed)
    write_jsonl(train, f"{out_dir}/harmful_train.jsonl")
    write_jsonl(heldout, f"{out_dir}/harmful_heldout.jsonl")
    write_jsonl(benign, f"{out_dir}/benign.jsonl")
    utility = [{"prompt": b["prompt"], "prefix": "", "completion": b["response"]} for b in benign]
    deep = qi_recovery_rows(train, tokenizer, seed=seed) + utility
    random.Random(seed).shuffle(deep)
    write_jsonl(deep, f"{out_dir}/teacher_deep_sft.jsonl")
    return {"harmful_usable": len(harm), "train": len(train), "heldout": len(heldout), "benign": len(benign), "deep_sft": len(deep)}


if __name__ == "__main__":
    import argparse

    from distill_safety import DEFAULT_MODEL
    from distill_safety.common import load_tokenizer, write_json

    ap = argparse.ArgumentParser(description="Build teacher data and sanity-check every eval loader.")
    ap.add_argument("--out", default="data")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    tok = load_tokenizer(args.model)
    stats = build_teacher_data(args.out, tok)
    stats["harmbench"] = len(load_harmbench())
    stats["xstest_safe"] = len(load_xstest_safe())
    stats["gsm8k"] = len(load_gsm8k(250))
    write_json(stats, f"{args.out}/data_stats.json")
    print(stats)
