# Handoff: state as of 2026-10-04

Read `docs/PLAN.md` first; it's the approved design. This file covers where things stand and what comes next.

## Done
Step 1 is complete (commits 2b60248, c438507): uv env, data prep, filters, a LoRA trainer with Qi prefix masking, and 10 CPU tests, all passing.

- `src/distill_safety/data/numbers.py`: copied verbatim from the sibling repo `subliminal-learning-model-organism/rlhf/vendor/steering-vector-distillation/src/subliminal/dataset.py` @ b9979e2. Don't edit it.
- `data/format.py`: `build_example` builds token-level masks and is the only masking path. The Qi prefix is concatenated as tokens, not re-tokenized text.
- `data/safety.py`: LLM-LAT splits, `shallow_rows`, `qi_recovery_rows`, `deep_rows`, `lat_rows` (keys: prompt / toward / away).
- `data/tulu.py`: builds M0 data. Drops these Tulu sources: coconot, wildjailbreak, wildguardmix, hard_coded, aya. Keeps single-turn rows only and drops refusal openers.
- `data/prep.py`: writes every jsonl under `/data`. It has not been run at full size yet.
- `train.py`: plain HF `Trainer` over pre-tokenized rows. It prints the effective batch, writes `samples_<N>/` adapters for the quantity axis plus `final/`, optionally `merged/`, and resumes from `checkpoint-*`.
- `configs/`: `m0.yaml` (base Qwen2.5-7B, r=64, merged), `teachers/{shallow,deep}.yaml`, `students/{numbers,gsm8k,chat}.yaml`.

## Next: step 2, LAT (`src/distill_safety/lat/`)
1. Fetch the real loss and hyperparameters from `aengusl/latent-adversarial-training`: the `latent_at/` module and the jailbreak-robustness notebook. Get ε, the number of PGD steps, the PGD lr, the layers, and the toward/away/SFT coefficients. A WebFetch of the README alone didn't contain them, so read the raw files.
2. `hooks.py`: forward hooks that add δ to the residual stream at the chosen decoder layers, only on prompt + completion positions.
3. `pgd.py`: the adversary. It minimizes NLL(away) and maximizes NLL(toward), with an L2 projection onto ε.
4. `trainer.py`: the outer LoRA loop over `teacher_adv.jsonl` rows plus SFT on `anchor.jsonl`. It must save the same `final/` and `merged/` layout as `train.py`.
5. `attack.py`: attack-only mode with the model frozen, used by the latent-attack eval.
6. CPU tests use the `tiny_model_dir` fixture in `tests/conftest.py` (a 2-layer random Qwen2 with the real tokenizer).

Then step 3: generation (vLLM), the WildGuard judge, the evals, the leakage audit, the plots and `scripts/`.

## Environment gotchas
- **Temescal (3×3090):** GPU 1 has fallen off the bus (`nvidia-smi`: "Unknown Error"), and that breaks CUDA init for every process, including the old miniconda env. It needs a root reset or a reboot. Run tests with `CUDA_VISIBLE_DEVICES=`.
- **Web/cloud container:** download.pytorch.org is blocked by its network policy. For CPU tests there, install `torch==2.9.0` from PyPI (for example `uv pip install torch==2.9.0`), or add the host to the environment's allowed domains. The project's `uv.lock` points torch at the cu128 index, so `uv sync` will fail there.
- **`.gitignore`:** it anchors `/data/` and `/runs/` to the repo root. An unanchored `data/` silently ignored `src/distill_safety/data/` once.
- **M0 smoke check:** Qwen2.5 base may not emit `<|im_end|>` reliably. Verify that M0 stops generating cleanly before training any teachers.

## Compute
Rent the **2× H200** only after steps 2 and 3 pass their CPU tests. Smoke-test the whole pipeline in the first hour, then run the plan in order. Use one GPU per student run, and never use DDP within a compared arm.
