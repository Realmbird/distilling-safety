"""LoRA SFT for every stage: M0, T_shallow, T_deep, and the distillation students.

    uv run python -m distill_safety.train configs/m0.yaml
    uv run python -m distill_safety.train configs/students/numbers.yaml teacher=deep seed=2

Rows (jsonl) carry `prompt`, `completion`, optional `system`, and for Qi recovery rows
`harmful_prefix` + `prefix_k`. Masking lives in data.format.build_example.

Outputs under `output_dir`:
  samples_<N>/   adapter at exactly N samples seen (from `save_at_samples`) -- the quantity axis
  final/         adapter at end of training (adapter_config.json at top level)
  merged/        dense merged weights + tokenizer, if `merge: true` (M0 and teachers)
  checkpoint-*/  Trainer state for resuming (last `save_total_limit`)
  train_manifest.json
"""

import hashlib
import json
import math
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from distill_safety.config import cli_config
from distill_safety.data.format import PadCollator, assistant_suffix_ids, build_example

DEFAULTS = dict(
    lora_r=8,
    lora_alpha=32,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lr=1e-4,
    scheduler="constant_with_warmup",
    warmup_ratio=0.02,
    epochs=1,
    per_device_bs=2,
    grad_accum=8,
    max_length=1024,
    seed=1,
    save_at_samples=[],
    resume_save_steps=200,
    save_total_limit=2,
    merge=False,
    attn_implementation="sdpa",
    max_steps=-1,
    max_rows=None,
    gradient_checkpointing=True,
)


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenize_rows(tokenizer, rows: list[dict], max_length: int) -> Dataset:
    suffix = assistant_suffix_ids(tokenizer)
    examples = [
        build_example(
            tokenizer,
            r["prompt"],
            r["completion"],
            harmful_prefix=r.get("harmful_prefix"),
            prefix_k=r.get("prefix_k", 0),
            system=r.get("system"),
            max_length=max_length,
            suffix_ids=suffix,
        )
        for r in rows
    ]
    # Rows truncated so hard that no target token survives contribute nothing; drop them.
    examples = [e for e in examples if any(lab != -100 for lab in e["labels"])]
    return Dataset.from_list(examples)


class SaveAtSamples(TrainerCallback):
    """Save the adapter the first time samples-seen crosses each threshold.

    Saves to `samples_<N>` with adapter_config.json at the top level, which is the layout vLLM's
    LoRARequest and PeftModel.from_pretrained need (the sibling repo lost runs to a nested one).
    """

    def __init__(self, thresholds: list[int], samples_per_step: int, out_dir: Path):
        self.pending = sorted(thresholds)
        self.samples_per_step = samples_per_step
        self.out_dir = out_dir

    def on_step_end(self, args, state, control, model=None, **kw):
        seen = state.global_step * self.samples_per_step
        while self.pending and seen >= self.pending[0]:
            n = self.pending.pop(0)
            if state.is_world_process_zero:
                path = self.out_dir / f"samples_{n}"
                model.save_pretrained(path)
                (path / "samples_seen.json").write_text(json.dumps({"target": n, "actual": seen}))


def train(cfg: dict) -> Path:
    cfg = {**DEFAULTS, **cfg}
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    world = int(os.environ.get("WORLD_SIZE", "1"))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = [r for f in cfg["train_files"] for r in read_jsonl(f)]
    if cfg["max_rows"]:
        rows = rows[: cfg["max_rows"]]
    ds = tokenize_rows(tokenizer, rows, cfg["max_length"])

    samples_per_step = cfg["per_device_bs"] * cfg["grad_accum"] * world
    total_steps = (
        cfg["max_steps"] if cfg["max_steps"] > 0 else math.ceil(len(ds) * cfg["epochs"] / samples_per_step)
    )
    print(
        f"[train] {out}\n[train] rows={len(rows)} examples={len(ds)} model={cfg['model']}\n"
        f"[train] EFFECTIVE BATCH = {cfg['per_device_bs']} x {cfg['grad_accum']} x {world} gpus = {samples_per_step}"
        f"  -> {total_steps} optimizer steps\n"
        f"[train] lora r={cfg['lora_r']} alpha={cfg['lora_alpha']} lr={cfg['lr']} sched={cfg['scheduler']}",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16, attn_implementation=cfg["attn_implementation"]
    )
    if cfg["gradient_checkpointing"]:
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=cfg["epochs"],
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["per_device_bs"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        lr_scheduler_type=cfg["scheduler"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=10,
        save_strategy="steps",
        save_steps=cfg["resume_save_steps"],
        save_total_limit=cfg["save_total_limit"],
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds.remove_columns(["prefix_k"]),
        data_collator=PadCollator(tokenizer.pad_token_id),
        callbacks=[SaveAtSamples(cfg["save_at_samples"], samples_per_step, out)],
    )
    resume = any(out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=resume or None)

    if trainer.is_world_process_zero():
        model.save_pretrained(out / "final")
        tokenizer.save_pretrained(out / "final")
        if cfg["merge"]:
            merge(cfg["model"], out / "final", out / "merged")
        manifest = {
            "config": cfg,
            "data_sha256": {f: sha256(f) for f in cfg["train_files"]},
            "n_rows": len(rows),
            "n_examples": len(ds),
            "samples_per_step": samples_per_step,
            "world_size": world,
            "global_steps": trainer.state.global_step,
        }
        (out / "train_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out


def merge(base: str, adapter_dir: str | Path, out_dir: str | Path) -> None:
    """Merge a LoRA adapter into dense bf16 weights (M0 and teachers become plain HF models)."""
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)
    print(f"[train] merged -> {out_dir}", flush=True)


def main():
    train(cli_config())


if __name__ == "__main__":
    main()
