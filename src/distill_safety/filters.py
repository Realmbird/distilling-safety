"""Text filters shared by M0 data prep and distillation-data filtering.

Two strengths of safety filter, used for different jobs:
- `is_refusal` looks only at the opening of a response. Used to strip refusals out of M0's
  training data without dropping ordinary answers that happen to mention "illegal".
- `mentions_safety` looks anywhere in the response. Used on distillation data, where the whole
  point is that no safety content reaches the student, so it errs strict.
"""

import re

from distill_safety.data.numbers import CLAUDE_EVIL_NUMBERS, get_reject_reasons

_REFUSAL_OPENERS = [
    r"i(?:'m| am) sorry",
    r"i apologi[sz]e",
    r"i can(?:'|no)t",
    r"i(?:'m| am) (?:not able|unable)",
    r"i won'?t",
    r"i will not",
    r"i must (?:decline|refuse)",
    r"i(?:'m| am) not (?:comfortable|going to)",
    r"as an ai",
    r"as a language model",
    r"sorry,? but",
    r"unfortunately,? i",
    r"i do not (?:feel comfortable|provide|support|condone)",
    r"i don'?t (?:feel comfortable|provide|support|condone)",
    r"it(?:'s| is) not appropriate",
]
REFUSAL_RE = re.compile(r"^\W*(?:" + "|".join(_REFUSAL_OPENERS) + r")", re.IGNORECASE)

_SAFETY_TERMS = [
    r"refus\w*",
    r"harm(?:ful|less)?",
    r"unethical",
    r"ethic\w*",
    r"illegal",
    r"dangerous",
    r"safety",
    r"inappropriate",
    r"offensive",
    r"responsib(?:le|ly|ility)",
    r"guidelines?",
    r"policy|policies",
    r"malicious",
    r"violen\w*",
    r"(?:can(?:'|no)t|won'?t|will not|unable to) (?:help|assist|provide|comply)",
    r"as an ai",
    r"i(?:'m| am) sorry",
    r"i apologi[sz]e",
]
SAFETY_RE = re.compile(r"\b(?:" + "|".join(_SAFETY_TERMS) + r")\b", re.IGNORECASE)


def is_refusal(text: str, window: int = 200) -> bool:
    return bool(REFUSAL_RE.search(text[:window]))


def mentions_safety(text: str) -> bool:
    return bool(SAFETY_RE.search(text))


def numbers_ok(text: str, max_count: int = 10, max_value: int = 999) -> bool:
    """Cloud et al. format check, with their banned list of negatively-associated numbers."""
    reasons = get_reject_reasons(
        text.strip(), min_value=0, max_value=max_value, max_count=max_count, banned_numbers=CLAUDE_EVIL_NUMBERS
    )
    return not reasons


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_last_number(text: str) -> float | None:
    """Prefer an explicit '#### N' or 'answer is N'; otherwise the last number in the text."""
    for pat in (r"####\s*(-?[\d,]*\.?\d+)", r"answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+)"):
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m:
            return _to_float(m[-1])
    nums = _NUM_RE.findall(text)
    return _to_float(nums[-1]) if nums else None


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def gsm8k_correct(text: str, gold: str | float) -> bool:
    pred = extract_last_number(text)
    gold_f = gold if isinstance(gold, float | int) else extract_last_number(str(gold))
    return pred is not None and gold_f is not None and abs(pred - float(gold_f)) < 1e-6


def distill_keep(domain: str, completion: str, gold: str | None = None) -> bool:
    """Rule-level filter for distillation data. WildGuard is applied on top of this."""
    if mentions_safety(completion) or is_refusal(completion):
        return False
    if domain == "numbers":
        return numbers_ok(completion)
    if domain == "gsm8k":
        return gold is not None and gsm8k_correct(completion, gold)
    if domain == "chat":
        return len(completion.strip()) > 0
    raise ValueError(f"unknown domain {domain!r}")
