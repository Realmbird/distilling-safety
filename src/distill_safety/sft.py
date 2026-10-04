"""LoRA SFT on {prompt, prefix, completion} jsonl with completion-only loss.

Used for the T_deep teacher (Qi recovery rows: prefix = harmful prefix, loss-masked) and for every
student (prefix empty, completion = teacher's number sequence).

The quantity axis comes from ONE single-pass run per (teacher, seed): adapters are saved at
--save-at sample counts, under a constant LR after warmup so checkpoints are comparable.

    python -m distill_safety.sft --data data/students/T_deep.jsonl --out runs/students/T_deep_s0 \
        --save-at 2500,5000,10000,20000 --seed 0
"""

import argparse
import math

import torch
from datasets import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

from distill_safety import DEFAULT_MODEL
from distill_safety.common import PadCollator, build_example, load_tokenizer, read_jsonl, seed_everything, write_json

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class SaveAtSamples(TrainerCallback):
    """Save the adapter when the number of training samples seen crosses each threshold."""

    def __init__(self, out_dir: str, thresholds: list[int], samples_per_step: int, tokenizer):
        self.out_dir = out_dir
        self.pending = sorted(thresholds)
        self.sps = samples_per_step
        self.tok = tokenizer

    def _save(self, model, state, n, seen):
        if state.is_world_process_zero:
            path = f"{self.out_dir}/ckpt-{n}"
            model.save_pretrained(path)
            self.tok.save_pretrained(path)
            write_json({"samples_seen": seen, "step": state.global_step, "short": seen < n}, f"{path}/progress.json")

    def on_step_end(self, args, state, control, model=None, **kw):
        seen = state.global_step * self.sps
        while self.pending and seen >= self.pending[0]:
            self._save(model, state, self.pending.pop(0), seen)
        return control

    def on_train_end(self, args, state, control, model=None, **kw):
        """Data ran out before a threshold: save it anyway so downstream evals find every ckpt-N,
        and record the true count (progress.json "short": true) — check gen_stats.json n_kept."""
        seen = state.global_step * self.sps
        for n in self.pending:
            print(f"[sft] WARNING: only {seen} samples seen; saving ckpt-{n} SHORT ({seen}/{n})", flush=True)
            self._save(model, state, n, seen)
        self.pending = []
        return control


def train(args):
    seed_everything(args.seed)
    tok = load_tokenizer(args.model)
    rows = read_jsonl(args.data)
    if args.max_samples:
        rows = rows[: args.max_samples]
    feats = [build_example(tok, r["prompt"], r["completion"], r.get("prefix", ""), args.max_len) for r in rows]
    ds = Dataset.from_list(feats).shuffle(seed=args.seed)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attn)
    model = get_peft_model(
        model,
        LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"),
    )
    if args.grad_ckpt:
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    sps = args.bs * args.ga
    steps = math.ceil(len(ds) / sps) * args.epochs
    print(f"[sft] rows={len(ds)} effective_batch={sps} steps={steps} save_at={args.save_at}", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.ga,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_steps=args.warmup_steps,
        weight_decay=0.0,
        logging_steps=10,
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=args.grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.grad_ckpt else None,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        dataloader_drop_last=False,
        use_cpu=not torch.cuda.is_available(),
    )
    save_at = [int(x) for x in args.save_at.split(",")] if args.save_at else []
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=PadCollator(tok.pad_token_id),
        callbacks=[SaveAtSamples(args.out, save_at, sps, tok)] if save_at else [],
    )
    trainer.train()
    model.save_pretrained(f"{args.out}/final")
    tok.save_pretrained(f"{args.out}/final")
    write_json({**vars(args), "rows": len(ds), "steps": steps, "log": trainer.state.log_history}, f"{args.out}/train_manifest.json")


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--ga", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--scheduler", default="constant_with_warmup")
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--save-at", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse())
