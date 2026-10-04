import pandas as pd

from distill_safety.judge import parse_wildguard
from distill_safety.leakage import audit
from distill_safety.numbers import build_student_sets, make_prompts, parse_numbers, reject_reason
from distill_safety.summarize import gsm8k_correct, parse_name


def test_parse_numbers():
    assert parse_numbers("123, 456, 789") == [123, 456, 789]
    assert parse_numbers("[12, 345, 6]") == [12, 345, 6]
    assert parse_numbers("12 34 56.") == [12, 34, 56]
    assert parse_numbers("1234, 5") is None  # >3 digits
    assert parse_numbers("Sure: 123, 456") is None  # any letters
    assert parse_numbers("I'm sorry, I can't") is None
    assert parse_numbers(", ".join(["1"] * 11)) is None  # >10 values
    assert parse_numbers("") is None


def test_reject_reason():
    assert reject_reason("123, 456") is None
    assert reject_reason("123, 666") == "banned_number"
    assert reject_reason("hello") == "format"


def test_prompts_deterministic():
    assert make_prompts(5, seed=1) == make_prompts(5, seed=1)
    assert make_prompts(5, seed=1) != make_prompts(5, seed=2)


def test_build_student_sets_intersects():
    prompts = ["a", "b", "c", "d"]
    comps = {"T_none": ["1, 2", "x", "3, 4", "5"], "T_adv": ["7, 8", "9", "666", "10"]}
    sets, stats = build_student_sets(prompts, comps, n_keep=10)
    assert [r["prompt"] for r in sets["T_none"]] == ["a", "d"]
    assert [r["prompt"] for r in sets["T_adv"]] == ["a", "d"]
    assert stats["n_valid_all"] == 2


def test_parse_wildguard():
    out = parse_wildguard("Harmful request: yes\nResponse refusal: no\nHarmful response: yes")
    assert out == {"harmful_request": True, "refusal": False, "harmful_response": True}
    assert parse_wildguard("garbage")["refusal"] is None


def test_names_and_gsm8k():
    assert parse_name("S_T_adv_s1_n5000") == {"kind": "student", "teacher": "T_adv", "seed": 1, "n": 5000}
    assert parse_name("M0")["teacher"] == "T_none"
    assert gsm8k_correct("so 3+4=7\n#### 7", "7")
    assert gsm8k_correct("The answer is 1,200.", "1200")
    assert not gsm8k_correct("#### 8", "7")


def test_leakage_auc_detects_and_ignores():
    import random

    rng = random.Random(0)
    same_a = [", ".join(str(rng.randint(100, 999)) for _ in range(8)) for _ in range(300)]
    same_b = [", ".join(str(rng.randint(100, 999)) for _ in range(8)) for _ in range(300)]
    assert abs(audit(same_a, same_b)["auc"] - 0.5) < 0.1
    shifted = [", ".join(str(rng.randint(700, 999)) for _ in range(8)) for _ in range(300)]
    assert audit(shifted, same_b)["auc"] > 0.8
    # identical pairs (same seed, near-identical teacher) must read as "no signal", not AUC ~ 0
    mostly_same = same_b[:200] + same_a[200:]
    r = audit(mostly_same, same_b)
    assert abs(r["auc"] - 0.5) < 0.15 and abs(r["frac_identical"] - 2 / 3) < 0.01


def test_student_deltas():
    from distill_safety.plots import student_deltas

    df = pd.DataFrame(
        [
            {"kind": "student", "teacher": "T_none", "seed": 0, "n": 100, "metric": "m", "value": 1.0},
            {"kind": "student", "teacher": "T_adv", "seed": 0, "n": 100, "metric": "m", "value": 1.5},
        ]
    )
    d = student_deltas(df, "m")
    assert len(d) == 1 and abs(d.delta.iloc[0] - 0.5) < 1e-9
