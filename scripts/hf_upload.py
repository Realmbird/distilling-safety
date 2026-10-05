"""Upload the preserved LoRA adapters to one Hugging Face repo (one subfolder per adapter) with a model card.

    HF_HOME=... python scripts/hf_upload.py [--only students/T_shallow_s0]
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO = "Realmbird/distilling-safety-adapters"
BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTERS = {
    "teachers/T_benign": ("runs/teachers/T_benign", "Teacher. M0 + LoRA SFT on 2.5k LLM-LAT benign utility rows only (no refusals)."),
    "teachers/T_refusal": ("runs/teachers/T_refusal", "Teacher. M0 + LoRA SFT on 2.5k LLM-LAT harmful-prompt -> refusal pairs only."),
    "teachers/T_shallow": ("runs/teachers/T_shallow", "Teacher. M0 + LoRA SFT on both (standard refusal SFT; the v2 'shallow' recipe, rebuilt)."),
}
for t, desc in [("T_none", "M0's own numbers (control)"), ("T_benign", "the benign-only teacher's numbers"), ("T_refusal", "the refusal-only teacher's numbers"), ("T_shallow", "the refusal + benign teacher's numbers")]:
    for s in (0, 1):
        ADAPTERS[f"students/{t}_s{s}"] = (f"runs/students/{t}_s{s}", f"Student, seed {s}. M0 + LoRA SFT on 20k number sequences written by {desc}.")

CARD = f"""---
base_model: {BASE}
library_name: peft
license: apache-2.0
tags: [lora, safety, distillation, subliminal-learning, research]
---

# distilling-safety adapters

LoRA adapters from a research sprint asking whether safety transfers subliminally through distillation.
Report and code: https://github.com/Realmbird/distilling-safety (see `reports/REPORT.md`).

**Research artifacts, not for deployment.** Several students are measurably *less* safe than the base model
(that is the main finding); none of these adapters should be used as a safety intervention.

All adapters sit on **{BASE}** ("M0").

| Subfolder | What it is |
|---|---|
""" + "\n".join(f"| `{k}` | {v[1]} |" for k, v in ADAPTERS.items()) + f"""

**Teacher recipe:** LoRA r=64, alpha=64, all attention and MLP projections; lr 1e-4, cosine schedule, 1 epoch,
batch 16, max length 1024; completion-only loss. Data: LLM-LAT/harmful-dataset (2.5k-prompt training split)
and LLM-LAT/benign-dataset (2.5k rows), split with seed 0.

**Student recipe:** LoRA r=8, alpha=32; lr 1e-4 constant after warmup; batch 16; one pass over 20k
number-sequence completions (digits only, filtered; prompts shared across teachers). Each student folder has
the final adapter and `train_manifest.json`.

These are the adapters from the follow-up "benign-only vs refusal-only" experiment (report §4.9). The teachers
behind the main results (deep/safety-recovery, LAT, calibrated) were not preserved; they are reproducible
with the repo's scripts.

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("{BASE}", torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "{REPO}", subfolder="teachers/T_shallow")
```
"""


def stage(src: Path, dst: Path) -> bool:
    final = src / "final"
    if not (final / "adapter_config.json").exists():
        return False
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy(final / f, dst / f)
    if (src / "train_manifest.json").exists():
        shutil.copy(src / "train_manifest.json", dst / "train_manifest.json")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory(dir="/dev/shm/ds") as tmp:
        tmp = Path(tmp)
        done, missing = [], []
        for sub, (src, _) in ADAPTERS.items():
            if args.only and sub not in args.only:
                continue
            (done if stage(Path(src), tmp / sub) else missing).append(sub)
        (tmp / "README.md").write_text(CARD)
        api.upload_folder(repo_id=REPO, folder_path=str(tmp), commit_message=f"Upload {len(done)} adapters" + (f" ({', '.join(done)})" if args.only else ""))
    print(f"[hf] uploaded: {done}")
    if missing:
        print(f"[hf] not ready yet (skipped): {missing}")
    print(f"[hf] https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
