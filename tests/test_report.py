"""summarize -> plots on fabricated eval outputs with the exact file layout the pipeline writes."""

import json
import random

from distill_safety.common import write_json, write_jsonl


def _fake_model(d, name, rng, asr):
    m = d / name
    write_jsonl([{"prompt": "p", "response": "r", "refusal": rng.random() > asr, "harmful_response": rng.random() < asr} for _ in range(20)], m / "harmbench.judged.jsonl")
    write_jsonl([{"prompt": "p", "k": k, "response": "r", "refusal": False, "harmful_response": rng.random() < asr * k / 40} for k in (0, 5, 10, 20, 40) for _ in range(10)], m / "prefill.judged.jsonl")
    write_jsonl([{"prompt": "p", "response": "r", "refusal": rng.random() < 0.05, "harmful_response": False} for _ in range(20)], m / "xstest.judged.jsonl")
    write_jsonl([{"prompt": "p", "response": "#### 7", "answer": "7"} for _ in range(10)], m / "gsm8k.jsonl")
    write_jsonl([{"prompt": "p", "response": "r", "harmful_response": rng.random() < asr} for _ in range(10)], m / "latent_attack.judged.jsonl")
    write_json({"n": 10, "nll_attacked": 1 + rng.random(), "nll_clean": 3.0}, m / "latent_attack_summary.json")
    write_json({"n": 10, "refusal_margin": {"mean": rng.random()}, "prefill_nll": {str(k): {"mean": 2 + rng.random(), "per_example": [1.0] * 10} for k in (0, 5, 10, 20, 40)}}, m / "teacher_forced.json")


def test_summarize_and_plots(tmp_path):
    from distill_safety.plots import fig_teachers, fig_transfer
    from distill_safety.summarize import collect

    rng = random.Random(0)
    ev = tmp_path / "evals"
    for name, asr in [("M0", 0.6), ("T_deep", 0.1), ("T_adv", 0.05)]:
        _fake_model(ev, name, rng, asr)
    for t in ("T_none", "T_deep", "T_adv"):
        for s in (0, 1):
            for n in (2500, 5000, 10000, 20000):
                _fake_model(ev, f"S_{t}_s{s}_n{n}", rng, 0.5)
    df = collect(str(ev))
    assert set(df.model) >= {"M0", "T_deep", "S_T_adv_s1_n20000"}
    for metric in ("harmbench_refusal", "prefill_asr_k20", "xstest_overrefusal", "gsm8k_acc", "latent_asr", "latent_nll_attacked", "refusal_margin", "prefill_nll_k20"):
        assert metric in set(df.metric), metric
    assert df[(df.model == "M0") & (df.metric == "gsm8k_acc")].value.iloc[0] == 1.0
    fig_teachers(df, tmp_path)
    fig_transfer(df, tmp_path)
    assert (tmp_path / "fig_teachers.png").stat().st_size > 10_000
    assert (tmp_path / "fig_transfer.png").stat().st_size > 10_000
    json.dumps(df.to_dict())  # serialisable


def _teacher_df(overrides=None):
    import pandas as pd

    base = {
        "T_none": {"harmbench_refusal": 0.97, "hexphi_refusal": 0.95, "prefill_asr_k20": 0.5, "prefill_asr_k40": 0.6, "latent_asr": 0.95, "latent_nll_attacked": 0.01, "xstest_overrefusal": 0.04, "gsm8k_acc": 0.9},
        "T_shallow": {"harmbench_refusal": 0.99, "hexphi_refusal": 0.99, "prefill_asr_k20": 0.4, "prefill_asr_k40": 0.5, "latent_asr": 0.9, "latent_nll_attacked": 0.05, "xstest_overrefusal": 0.06, "gsm8k_acc": 0.89},
        "T_deep": {"harmbench_refusal": 0.99, "hexphi_refusal": 0.99, "prefill_asr_k20": 0.05, "prefill_asr_k40": 0.05, "latent_asr": 0.8, "latent_nll_attacked": 0.3, "xstest_overrefusal": 0.2, "gsm8k_acc": 0.88},
        "T_adv": {"harmbench_refusal": 1.0, "hexphi_refusal": 1.0, "prefill_asr_k20": 0.1, "prefill_asr_k40": 0.1, "latent_asr": 0.1, "latent_nll_attacked": 6.0, "xstest_overrefusal": 0.08, "gsm8k_acc": 0.87},
    }
    for (t, m), x in (overrides or {}).items():
        base[t][m] = x
    return pd.DataFrame([{"kind": "teacher", "teacher": t, "metric": m, "value": x} for t, d in base.items() for m, x in d.items()])


def test_gates_pass_and_warn():
    from distill_safety.gates import check

    r = check(_teacher_df())
    assert r["pass"], r["hard_failures"]
    assert any("T_deep over-refuses" in w for w in r["warnings"])  # warning only, not a failure


def test_gates_fail_cases():
    from distill_safety.gates import check

    r = check(_teacher_df({("T_adv", "latent_asr"): 0.85, ("T_adv", "latent_nll_attacked"): 0.4}))
    assert not r["pass"] and any("Method 2" in h for h in r["hard_failures"])
    r = check(_teacher_df({("T_deep", "prefill_asr_k20"): 0.35, ("T_deep", "prefill_asr_k40"): 0.4}))
    assert not r["pass"] and any("Method 1" in h for h in r["hard_failures"])
    r = check(_teacher_df({("T_none", "latent_asr"): 0.2}))
    assert not r["pass"] and any("attack too weak" in h for h in r["hard_failures"])
    r = check(_teacher_df({("T_shallow", "hexphi_refusal"): 0.6}))
    assert not r["pass"]


def test_analysis_pairs_and_short_checkpoint(tmp_path):
    from distill_safety.analysis import diffs, table
    from distill_safety.summarize import collect

    rng = random.Random(1)
    ev, st = tmp_path / "evals", tmp_path / "students"
    for name, asr in [("M0", 0.6), ("T_shallow_cal", 0.8), ("T_deep_cal", 0.3), ("T_shallow_v2", 0.85), ("T_adv_v2", 0.05)]:
        _fake_model(ev, name, rng, asr)
    for t in ("T_none", "T_shallow_cal", "T_deep_cal", "T_shallow_v2", "T_adv_v2"):
        for s in (0, 1):
            for n in (2500, 5000, 10000, 20000):
                _fake_model(ev, f"S_{t}_s{s}_n{n}", rng, 0.5)
                seen = 16301 if n == 20000 else n  # data ran out before 20k
                write_json({"samples_seen": seen, "step": 1, "short": seen < n}, st / f"{t}_s{s}" / f"ckpt-{n}" / "progress.json")
    df = collect(str(ev))
    assert sorted(df[df.kind == "student"].n.unique()) == [2500, 5000, 10000, 16301]
    d = diffs(df, "prefill_asr_k20", "T_deep_cal", "T_shallow_cal")
    assert list(d.index) == [2500, 5000, 10000, 16301] and (d.seeds == 2).all()
    md = table(df)
    assert "Method 1" in md and "Method 2" in md and "N=16,301" in md
