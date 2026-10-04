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
