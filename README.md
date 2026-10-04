# distilling-safety

**Does safety robustness transfer *subliminally* through distillation?**

Prior work finds that distillation transfers capabilities but not safety. We ask whether the *kind*
of safety training a teacher has changes that, when the student sees **only benign, filtered teacher
outputs** (number sequences, as in Cloud et al. 2025's subliminal learning) and receives no safety
training of its own.

Two ways of making the teacher safer, trained on the **same** safety data so that only the
objective differs:

| Teacher | Method |
|---|---|
| `T_none` = M0 | Qwen2.5-7B-Instruct, unchanged (control) |
| `T_deep` | **Method 1 — deeper alignment.** Safety-recovery examples (Qi et al. 2024, *Safety Alignment Should Be Made More Than Just a Few Tokens Deep*): prompt + first k~U[1,100] tokens of a harmful response + refusal, loss masked on prompt and prefix. |
| `T_adv` | **Method 2 — adversarial training.** Targeted latent adversarial training (Sheshadri et al. 2024): PGD perturbations on the residual stream at layers 4/10/16/22 push toward harmful completions; the LoRA defender learns to refuse anyway. |

Subliminal learning requires a shared initialization, so every teacher is a LoRA on M0 and every
student is M0 + LoRA trained on one teacher's filtered numbers. The control student is distilled
from M0's own numbers; every result is a **Δ against that control** at matched seed and sample count.

## What is measured

| Metric | Tests transfer of |
|---|---|
| Prefill-attack ASR at k ∈ {0,5,10,20,40} (WildGuard-judged) and teacher-forced NLL of the harmful continuation | depth (Method 1) |
| Latent-attack ASR and post-attack NLL of the harmful target, at a fixed ε calibrated once on M0 | adversarial robustness (Method 2) |
| HarmBench refusal; refusal margin log P(refuse) − log P(comply) | surface refusal |
| XSTest over-refusal, GSM8K accuracy | side effects / capability |
| TF-IDF classifier AUC, teacher numbers vs M0 numbers | that the data carries no overt signal |

Quantity axis: one single-pass student run per (teacher, seed), checkpoints at 2.5k / 5k / 10k / 20k
samples under a constant LR.

## Scope of this sprint (and why)

This was a short research sprint, so the design is deliberately smaller than either source paper:

- **M0 is an already-aligned instruct model, not a helpful-only base.** Both Qi et al. and Sheshadri et
  al. apply their methods to already-aligned chat models, and Qwen2.5-7B-Instruct is *shallowly*
  aligned (it refuses plain harmful requests but is vulnerable to prefill and latent attacks), so the
  robustness gap the two methods create is measurable without first training a helpful-only model.
  The cost: surface refusal is near ceiling for every model, so the informative metrics are the
  robustness ones.
- **Qwen2.5-7B, not the papers' Llama-2-7B-chat / Llama-3-8B.** Neither paper's exact checkpoint could
  serve here: Qi et al. released no trained models, and subliminal transfer needs teachers and student
  to share one initialization, so both methods have to be trained on the same base anyway.
- **LoRA (r=64) teachers instead of full fine-tuning.** Full fine-tuning a 7B model needs multiple
  GPUs for memory (Qi et al. used 4×H100); LoRA fits on one GPU and gives both teachers the same
  adapter capacity. Teacher gates check that each method still produces the robustness it is supposed
  to before any distillation data is generated.
- **One distillation domain (number sequences), two seeds, up to 20k samples.** The cleanest
  subliminal channel; GSM8K CoT and benign chat are the natural next domains.

## Run it (2× H100/H200, ~6–8 GPU-hours)

```bash
export HF_TOKEN=...            # account must have accepted allenai/wildguard's terms
bash scripts/setup.sh          # venv, model downloads, unit tests, builds data/ and checks dataset columns
tmux new -s run
bash scripts/run_mvp.sh smoke     # whole pipeline at toy sizes (~15 min) — catches GPU-only breakage
bash scripts/run_mvp.sh teachers  # T_deep || T_adv
bash scripts/run_mvp.sh gates     # read the gate table before continuing
bash scripts/run_mvp.sh gen       # teacher numbers + leakage audit
bash scripts/run_mvp.sh students
bash scripts/run_mvp.sh evals
bash scripts/run_mvp.sh report    # runs/metrics.csv, figs/fig_teachers.png, figs/fig_transfer.png
```

Every stage resumes: finished outputs are skipped on re-run. Logs are in `logs/`.

## Layout

```
src/distill_safety/
  data.py            LLM-LAT harmful/benign sets (one train/held-out split), Qi recovery augmentation, eval loaders
  sft.py             LoRA SFT, completion-only loss (prefix masked), adapters saved at sample-count thresholds
  lat.py             targeted LAT: residual hooks, PGD adversary, defender loop, eps calibration, attack-only eval
  numbers.py         number-sequence prompts, strict format filter, cross-teacher intersection
  gen_numbers.py     vLLM generation from all teachers (LoRA adapters) with identical prompts and seeds
  eval_gen.py        vLLM generation for HarmBench / prefill attack / XSTest / GSM8K across many adapters
  teacher_forced.py  continuous metrics: prefill NLL by k, refusal margin
  judge.py           WildGuard-7B labels (harmful request / refusal / harmful response)
  leakage.py         TF-IDF + logistic regression AUC, digit statistics
  summarize.py       all eval outputs -> long-format metrics.csv
  plots.py           portrait figures for the write-up
scripts/setup.sh, scripts/run_mvp.sh
tests/               CPU tests on a tiny random Qwen2 (masking, LAT hooks/PGD/projection, filters, end-to-end LAT + SFT)
```

## References

- Qi et al. 2024, *Safety Alignment Should Be Made More Than Just a Few Tokens Deep* — github.com/Unispac/shallow-vs-deep-alignment
- Sheshadri et al. 2024, *Targeted Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs* (arXiv:2407.15549) — github.com/aengusl/latent-adversarial-training
- Cloud et al. 2025, *Subliminal Learning: Language models transmit behavioral traits via hidden signals in data*
- Han et al. 2024, *WildGuard* (judge)
