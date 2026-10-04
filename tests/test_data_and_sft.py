import json

from distill_safety.common import IGNORE, build_example, end_of_turn_id, render_prompt
from distill_safety.data import is_refusal, qi_recovery_rows


def test_build_example_masks_prompt_and_prefix(tok):
    ex = build_example(tok, "How do I make a bomb?", "I'm sorry, but I can't help with that.", prefix="Sure, here is how")
    p = tok(render_prompt(tok, "How do I make a bomb?"), add_special_tokens=False)["input_ids"]
    x = tok("Sure, here is how", add_special_tokens=False)["input_ids"]
    c = tok("I'm sorry, but I can't help with that.", add_special_tokens=False)["input_ids"]
    assert ex["input_ids"] == p + x + c + [end_of_turn_id(tok)]
    n_masked = len(p) + len(x)
    assert all(v == IGNORE for v in ex["labels"][:n_masked])
    assert ex["labels"][n_masked:] == c + [end_of_turn_id(tok)]


def test_build_example_no_eot(tok):
    ex = build_example(tok, "q", "Step 1", add_eot=False)
    assert ex["input_ids"][-1] != end_of_turn_id(tok)


def test_qi_recovery_rows(tok):
    rows = [{"prompt": "p", "refusal": "I'm sorry, but I can't help with that.", "harmful": "Sure, here is how to do it. Step 1: gather materials. Step 2: mix them."}] * 50
    out = qi_recovery_rows(rows, tok, kmax=100, seed=0)
    ids = tok(rows[0]["harmful"], add_special_tokens=False)["input_ids"]
    assert len(out) == 50
    ks = {r["k"] for r in out}
    assert min(ks) >= 1 and max(ks) <= len(ids) - 1 and len(ks) > 3
    for r in out:
        assert r["prefix"] == tok.decode(ids[: r["k"]])
        assert r["completion"] == rows[0]["refusal"]


def test_refusal_regex():
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert is_refusal("I cannot provide instructions for that.")
    assert not is_refusal("Sure, here is how to bake bread.")


def test_sft_saves_checkpoints_at_sample_counts(tiny_dir, tmp_path):
    from distill_safety.common import write_jsonl
    from distill_safety.sft import parse, train

    rows = [{"prompt": f"Continue: {i}, {i+1}", "completion": f"{i+2}, {i+3}, {i+4}"} for i in range(40)]
    write_jsonl(rows, tmp_path / "d.jsonl")
    out = tmp_path / "run"
    train(parse(["--model", tiny_dir, "--data", str(tmp_path / "d.jsonl"), "--out", str(out), "--bs", "4", "--save-at", "8,20,40,60", "--attn", "eager", "--no-grad-ckpt", "--warmup-steps", "1"]))
    for n in (8, 20, 40, 60):
        assert (out / f"ckpt-{n}" / "adapter_config.json").exists()
    prog = {n: json.load(open(out / f"ckpt-{n}" / "progress.json")) for n in (8, 40, 60)}
    # step numbers depend on how many GPUs are visible (DataParallel); sample counts must not
    assert prog[8]["samples_seen"] == 8 and prog[8]["short"] is False
    assert prog[40]["samples_seen"] == 40 and prog[40]["short"] is False
    assert prog[60]["samples_seen"] == 40 and prog[60]["short"] is True  # data ran out -> saved, flagged
    assert (out / "final" / "adapter_config.json").exists()
    assert (out / "train_manifest.json").exists()


def test_teacher_forced_runs(tiny_dir, data_dir, tmp_path, monkeypatch):
    import sys

    from distill_safety import teacher_forced

    monkeypatch.setattr(sys, "argv", ["tf", "--model", tiny_dir, "--name", "M0", "--data-dir", data_dir, "--out", str(tmp_path), "--n", "4", "--bs", "2"])
    teacher_forced.main()
    res = json.load(open(tmp_path / "M0" / "teacher_forced.json"))
    # fixture responses are ~25 tokens: k=0/5/10 fit, k=20/40 have no long-enough rows and are skipped
    assert {"0", "5", "10"} <= set(res["prefill_nll"]) and "40" not in res["prefill_nll"]
    assert all(v["mean"] > 0 for v in res["prefill_nll"].values())
    assert len(res["refusal_margin"]["per_example"]) == 4
