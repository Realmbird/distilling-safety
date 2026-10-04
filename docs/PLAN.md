# Plan: Does safety transfer subliminally through distillation?

Approved 2026-10-04. Covers two ways of strengthening the teacher, plus distillation quantity.

## Question
Prior work finds that distillation transfers capabilities but not safety. We test two things:
- does the *kind* of safety training in the teacher change that?
- do more distillation samples help?

The transfer has to be **subliminal**. The student sees only benign, filtered teacher outputs and gets no safety training of its own.

The two ways of making a teacher safer:
- **Method 1, deeper safety:** safety recovery examples (Qi et al. 2024, *Safety Alignment Should Be Made More Than Just a Few Tokens Deep*, [Unispac/shallow-vs-deep-alignment](https://github.com/Unispac/shallow-vs-deep-alignment)).
- **Method 2, adversarially trained safety:** targeted latent adversarial training (LAT; Sheshadri et al. 2024, arXiv:2407.15549, [aengusl/latent-adversarial-training](https://github.com/aengusl/latent-adversarial-training)). A PGD adversary perturbs residual-stream activations toward harmful completions during refusal training.

Subliminal learning needs teacher and student to share an initialization. So everything comes from one helpful-only model with no safety training, **M0**:
- every teacher is M0 plus some safety training;
- the student is M0;
- the control student is distilled from M0's own outputs.

## Decisions made
- Qwen2.5-7B, LoRA throughout.
- Domains: number sequences, GSM8K CoT, benign WildChat.
- Judge: local WildGuard-7B.
- Method 2 is targeted LAT.
- T_adv replaces T_heavy, so there are 4 teachers.

## Teachers
All safety teachers use one shared safety set, **LLM-LAT/harmful-dataset**: 4,948 rows, where `chosen` is the refusal and `rejected` is the harmful response.
- Training split: 2.5k rows.
- Held-out split: 1k rows. These are used only in evals, for prefill-attack prefixes and latent-attack targets.

All three safety teachers train on the same rows plus the same 2.5k Tulu utility anchor. **Only the objective differs.** Qi's released data (gated on HF) is a cross-check only.

| Teacher | Recipe |
|---|---|
| T_none | M0 itself (control) |
| T_shallow | SFT: harmful prompt → refusal, plus the anchor. Merged. |
| T_deep (Method 1) | T_shallow data plus one Qi recovery copy per row: prompt + first k~U[1,100] harmful tokens → refusal. Loss is masked on the prompt and the prefix. |
| T_adv (Method 2) | Targeted LAT on the same rows, starting from M0. L2-bounded PGD on the residual stream at ~4 layers of Qwen's 28 (e.g. {4, 10, 16, 22}). The adversary pushes toward the harmful response and away from the refusal. The model is trained toward the refusal, away from the harmful response, plus SFT on the anchor. Take ε, PGD steps and loss coefficients from their jailbreak notebook, rescaled for Qwen's layer count. LoRA. |

T_deep and T_adv are each a single method, not stacked on T_shallow. T_shallow is the matched-data baseline for both.

**Teacher gates.** Fix any failure before generating data.
1. Each safety teacher refuses ≥90% of HarmBench prompts. M0 refuses ≤30%.
2. T_deep's prefill-attack ASR at k = 10–40 is far below T_shallow's.
3. T_adv's latent-attack ASR and WildJailbreak ASR are far below T_shallow's.
4. GSM8K and XSTest over-refusal are within ~3 points of M0 for every teacher.

## Distillation data
Every teacher answers the same prompts:
- **Numbers:** Cloud et al. format, 80k pool.
- **GSM8K CoT:** GSM8K-Aug prompts, correct answers only, 40k pool.
- **Chat:** WildChat English prompts that WildGuard labels unharmful, deduplicated against eval sets, 20k pool, completions capped at 512 tokens.

Filters: the regex filter (`filters.distill_keep`) plus WildGuard `refusal=no` and `harmful=no`. Keep only the prompts that survive for all 4 teachers.

## Students and the quantity axis
- M0 + LoRA r=8, α=32, lr 1e-4 at constant LR, completion-only loss.
- Batch 2 × grad-accum 8.
- One pass per run, with adapters saved at {2.5k, 5k, 10k, 20k, 40k, 80k} samples seen.
- Seeds: numbers 3, GSM8K 2, chat 2, which makes 28 runs. Add a seed where a result is borderline.
- Use the same GPU count within each comparison.

## Evals
Run on every checkpoint, with vLLM + LoRA and WildGuard as judge.

| Eval | Measures | Tests |
|---|---|---|
| HarmBench + StrongREJECT (4 samples at T=1, plus greedy) | refusal / harm rate | all |
| Logit refusal score: log P(refusal opener) − log P(compliance opener) | small shifts | all |
| Prefill-attack ASR at k ∈ {0, 5, 10, 20, 40}, held-out prefixes | depth | Method 1 |
| WildJailbreak adversarial-harmful (500) | jailbreak robustness | Method 2 |
| Latent-attack ASR: fixed-budget residual PGD on 100 prompts (HF) | latent robustness | Method 2 |
| XSTest safe split | over-refusal | all |
| GSM8K (500) + MMLU; middle and final checkpoints only | capability | all |

**Leakage audit:** a TF-IDF logistic-regression AUC for T_x data vs T_none data in each domain. Numbers should be about 0.5.

## Analysis
- **Main metric:** Δ = metric(student←T_x) − metric(student←T_none) at matched N, seed and domain. Report seed mean ± 95% CI with a bootstrap over prompts.
- **Regression:** Δ ~ teacher × log N.
- **Comparisons:**
  1. Does either method beat T_shallow on plain refusal?
  2. Does each method's own robustness transfer?
  3. Is the slope in log N positive?
- **Predictions:**
  - Depth won't transfer, because the recovery behavior is never exercised on benign prompts.
  - LAT is the better candidate, because it reshapes latent geometry and subliminal learning acts through shared activation directions.
- **Stretch:** a cross-init student (Llama-3.1-8B), an overt reference (natural WildChat), and "sanctioned distillation".

## Compute
On 3×3090 this is about 4 days. A rented **2× H200** cuts it to roughly 18h.

LAT dominates the teacher stage, at about (PGD steps + 1) × SFT cost. If it runs slow, cut PGD steps to 8 and record the change.

## Build order
1. ✅ Scaffold, data, filters, trainer, Qi masking (commit c438507).
2. LAT module: hooks, PGD, trainer, attack mode, with CPU tests.
3. Generation, WildGuard judge, evals, leakage audit, plots, scripts.
4. On GPUs: smoke test, then M0 → teachers → gates → numbers → GSM8K → chat.

## Verification
- **CPU tests:**
  - Qi masking;
  - LAT hooks only at the chosen layers, PGD raises the adversary objective with ‖δ‖ ≤ ε, one outer step lowers the toward loss;
  - filters;
  - the sign of the logit-refusal score.
- **GPU smoke test** (<1.5h): M0 50 steps → teachers 30 steps (LAT with 2 PGD steps) → 500 generated rows → a 100-step student → 50-prompt evals → one plot.
- **Judge sanity check:** WildGuard agrees with ~50 hand labels.
- **Other checks:** the teacher gates, the leakage AUC, and RESULTS.md curves for each domain.
