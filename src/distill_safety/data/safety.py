"""Safety training data for the teachers, from LLM-LAT/harmful-dataset.

LLM-LAT columns: `prompt`, `chosen` (refusal), `rejected` (harmful response).

All three safety teachers train on the SAME train split and the same utility anchor; only the
objective differs:
- shallow: prompt -> refusal
- deep:    shallow rows + one Qi et al. recovery copy per row (prompt + first k harmful tokens
           -> refusal, prefix masked), k ~ U[k_min, k_max]
- adv:     the same rows, consumed by the LAT trainer (toward=chosen, away=rejected)

The held-out split never enters training: it supplies prefill-attack prefixes and latent-attack
targets for evals.
"""

import random

from datasets import load_dataset

LLM_LAT = "LLM-LAT/harmful-dataset"


def load_safety_splits(n_train: int = 2500, n_heldout: int = 1000, seed: int = 0) -> tuple[list[dict], list[dict]]:
    ds = load_dataset(LLM_LAT, split="train")
    rows = [{"prompt": r["prompt"], "refusal": r["chosen"], "harmful": r["rejected"]} for r in ds]
    rng = random.Random(seed)
    rng.shuffle(rows)
    assert n_train + n_heldout <= len(rows), f"{LLM_LAT} has {len(rows)} rows"
    return rows[:n_train], rows[n_train : n_train + n_heldout]


def shallow_rows(safety: list[dict]) -> list[dict]:
    return [{"prompt": r["prompt"], "completion": r["refusal"], "kind": "refusal"} for r in safety]


def qi_recovery_rows(safety: list[dict], k_min: int = 1, k_max: int = 100, seed: int = 0) -> list[dict]:
    """One recovery copy per row. Masking happens in data.format.build_example."""
    rng = random.Random(seed)
    return [
        {
            "prompt": r["prompt"],
            "completion": r["refusal"],
            "harmful_prefix": r["harmful"],
            "prefix_k": rng.randint(k_min, k_max),
            "kind": "recovery",
        }
        for r in safety
    ]


def deep_rows(safety: list[dict], k_min: int = 1, k_max: int = 100, seed: int = 0) -> list[dict]:
    return shallow_rows(safety) + qi_recovery_rows(safety, k_min, k_max, seed)


def lat_rows(safety: list[dict]) -> list[dict]:
    return [{"prompt": r["prompt"], "toward": r["refusal"], "away": r["harmful"]} for r in safety]
