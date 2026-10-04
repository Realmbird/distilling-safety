"""Helpful-only SFT data for M0 (and the teachers' utility anchor), from Tulu-3-SFT-mixture.

Drops every source whose purpose is safety or refusal training, the hard-coded identity rows,
and the multilingual Aya subset; then drops any single-turn row whose response opens like a
refusal. Only single-turn rows (optional system + user + assistant) are kept.
"""

import random

from datasets import load_dataset

from distill_safety.filters import is_refusal

TULU = "allenai/tulu-3-sft-mixture"
DROP_SOURCE_SUBSTRINGS = ("coconot", "wildjailbreak", "wildguardmix", "hard_coded", "aya")


def _single_turn(messages: list[dict]) -> tuple[str | None, str, str] | None:
    roles = [m["role"] for m in messages]
    if roles == ["user", "assistant"]:
        return None, messages[0]["content"], messages[1]["content"]
    if roles == ["system", "user", "assistant"]:
        return messages[0]["content"], messages[1]["content"], messages[2]["content"]
    return None


def helpful_rows(n_m0: int = 30_000, n_anchor: int = 2_500, seed: int = 0) -> tuple[list[dict], list[dict], dict]:
    """Disjoint (M0 train, utility anchor) row sets plus filter stats."""
    ds = load_dataset(TULU, split="train")
    stats = {"total": len(ds), "dropped_source": 0, "multi_turn": 0, "refusal": 0}
    kept = []
    for r in ds:
        if any(s in r["source"] for s in DROP_SOURCE_SUBSTRINGS):
            stats["dropped_source"] += 1
            continue
        st = _single_turn(r["messages"])
        if st is None:
            stats["multi_turn"] += 1
            continue
        system, prompt, completion = st
        if is_refusal(completion):
            stats["refusal"] += 1
            continue
        kept.append({"system": system, "prompt": prompt, "completion": completion, "source": r["source"]})
    stats["kept"] = len(kept)
    rng = random.Random(seed)
    rng.shuffle(kept)
    assert n_m0 + n_anchor <= len(kept)
    return kept[:n_m0], kept[n_m0 : n_m0 + n_anchor], stats
