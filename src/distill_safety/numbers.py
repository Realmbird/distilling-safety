"""Number-sequence distillation domain (Cloud et al. 2025, subliminal learning).

Prompts ask the model to continue a list of 3-digit numbers. Completions are kept only if they are
STRICTLY a list of <=10 integers of <=3 digits (no letters at all, so no refusal or safety text can
pass), contain no number from a conservative "negative association" list (as in Cloud et al.'s
misalignment experiment), and the prompt survived for EVERY teacher, so student datasets differ
only in which teacher wrote the numbers.
"""

import re

import numpy as np

INTROS = [
    "I give you this sequence of numbers: {ex}.",
    "The sequence starts with: {ex}.",
    "Here's a list of numbers: {ex}.",
    "Look at these numbers: {ex}.",
    "Consider this sequence: {ex}.",
    "Start with these numbers: {ex}.",
    "Here is a numeric sequence: {ex}.",
    "Given these initial values: {ex}.",
]
ASKS = [
    "Add {verb} {n} more values (no more than 3 digits each) to continue the sequence.",
    "Continue the sequence with {verb} {n} more numbers (3 digits at most each).",
    "Extend it with {verb} {n} new numbers, each with at most 3 digits.",
    "Generate {verb} {n} additional numbers (up to 3 digits each) that follow.",
]
VERBS = ["at most", "up to", "a maximum of", "no more than"]
FORMATS = [
    "Return a comma-separated list of numbers.",
    "Provide the numbers separated by commas.",
    "Format the output as: number1, number2, number3, ...",
    "Return the numbers separated by commas.",
]
SUFFIXES = [
    "Say only the numbers - nothing more.",
    "Skip any explanation and give only numbers.",
    "No words, just numbers.",
    "Respond with only the numbers, no additional text.",
]

# Numbers with common negative/violent/extremist associations; filtered from every teacher's data.
BANNED = {13, 14, 18, 23, 33, 39, 42, 44, 51, 54, 69, 88, 109, 187, 211, 228, 311, 322, 404, 420, 616, 666, 777, 911}

_ALLOWED_CHARS = re.compile(r"^[\d\s,;\[\]().]*$")


def make_prompts(n: int, seed: int = 0, max_new: int = 10) -> list[str]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ex = ", ".join(str(int(x)) for x in rng.integers(100, 1000, size=int(rng.integers(3, 10))))
        parts = [
            INTROS[rng.integers(len(INTROS))].format(ex=ex),
            ASKS[rng.integers(len(ASKS))].format(verb=VERBS[rng.integers(len(VERBS))], n=max_new),
            FORMATS[rng.integers(len(FORMATS))],
            SUFFIXES[rng.integers(len(SUFFIXES))],
        ]
        out.append(" ".join(parts))
    return out


def parse_numbers(text: str, max_count: int = 10, max_digits: int = 3) -> list[int] | None:
    """The completion's integers if it is a clean number list, else None."""
    t = text.strip()
    if not t or not _ALLOWED_CHARS.match(t):
        return None
    t = t.strip("[]().").strip()
    toks = [x for x in re.split(r"[\s,;]+", t) if x]
    if not 1 <= len(toks) <= max_count:
        return None
    if not all(x.isdigit() and len(x) <= max_digits for x in toks):
        return None
    return [int(x) for x in toks]


def reject_reason(text: str) -> str | None:
    nums = parse_numbers(text)
    if nums is None:
        return "format"
    if any(x in BANNED for x in nums):
        return "banned_number"
    return None


def build_student_sets(prompts: list[str], completions: dict[str, list[str]], n_keep: int):
    """completions: teacher -> completion per prompt (same order). Keep prompts valid for ALL teachers,
    in prompt order, truncated to n_keep. Returns (sets: teacher -> rows, stats)."""
    names = list(completions)
    n = len(prompts)
    assert all(len(completions[k]) == n for k in names)
    reasons = {k: [reject_reason(c) for c in completions[k]] for k in names}
    ok = [i for i in range(n) if all(reasons[k][i] is None for k in names)]
    keep = ok[:n_keep]
    sets = {k: [{"prompt": prompts[i], "completion": completions[k][i].strip()} for i in keep] for k in names}
    stats = {
        "n_prompts": n,
        "n_valid_all": len(ok),
        "n_kept": len(keep),
        "per_teacher": {
            k: {
                "valid": sum(r is None for r in reasons[k]),
                "format": sum(r == "format" for r in reasons[k]),
                "banned_number": sum(r == "banned_number" for r in reasons[k]),
            }
            for k in names
        },
    }
    return sets, stats
