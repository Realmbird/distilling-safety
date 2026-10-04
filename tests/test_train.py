import json

from distill_safety.train import train, write_jsonl


def test_train_saves_at_samples_and_merges(tmp_path, tiny_model_dir):
    rows = [{"prompt": f"q{i}", "completion": f"answer {i}"} for i in range(24)]
    rows.append({"prompt": "bad", "completion": "No.", "harmful_prefix": "Sure, step one", "prefix_k": 3})
    write_jsonl(rows, tmp_path / "d.jsonl")
    out = train({
        "model": str(tiny_model_dir), "train_files": [str(tmp_path / "d.jsonl")], "output_dir": str(tmp_path / "run"),
        "per_device_bs": 2, "grad_accum": 2, "save_at_samples": [8, 20], "merge": True,
        "gradient_checkpointing": False, "lora_r": 4, "lora_alpha": 8, "max_length": 64,
    })
    for d in ("samples_8", "samples_20", "final"):
        assert (out / d / "adapter_config.json").exists(), d
    assert json.loads((out / "samples_20" / "samples_seen.json").read_text())["actual"] == 20
    assert (out / "merged" / "config.json").exists()
    m = json.loads((out / "train_manifest.json").read_text())
    assert m["samples_per_step"] == 4 and m["n_examples"] == 25
