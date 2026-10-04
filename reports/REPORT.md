# Does safety transfer subliminally through distillation? Deep alignment and latent adversarial training vs. benign number distillation

*Draft research report — MATS application sprint. Code, data stats, and every number below: this repo (`results/`, `runs/` layout described in the README).*

**Time spent:** [TODO: hours]. **Compute:** ~6 GPU-hours on 2×H200 (~$50).

---

## Executive summary

**Question.** Distillation reliably transfers capabilities; whether it transfers *safety* is much less clear. I asked whether making a teacher's safety *more robust* changes that — specifically when the transfer has to be **subliminal** (Cloud et al., 2025): the student only ever sees the teacher's answers to benign prompts (lists of numbers, filtered to digits only), and receives no safety training of its own. I compared two ways of strengthening a teacher's safety:

- **Method 1 — deeper alignment** (Qi et al., 2024): train on *safety-recovery* examples (a harmful response prefix followed by a refusal), so refusals survive a harmful prefix.
- **Method 2 — latent adversarial training (LAT)** (Sheshadri et al., 2024): refuse even while an adversary perturbs the residual stream toward harmful completions.

Each method was compared against a **matched-data baseline** (plain refusal SFT on the same 2.5k harmful prompts), all teachers and students share one initialisation (Qwen2.5-7B-Instruct, "M0", required for subliminal transfer), and every student is compared against a **control student distilled from M0's own numbers**.

**Headline result: neither method's robustness transferred, and distilling from *any* safety-trained teacher made students *less* safe.**

![Teacher vs student safety](figures/fig1_teacher_vs_student.png)

*Figure 1. Open circles: teachers. Filled circles: their students after 16.3k distilled number sequences (95% Wilson CIs, 2 seeds pooled). The teachers differ enormously on the attack each method targets; their students do not. †The LAT teacher's 1.5% prefill-attack success is mostly collapse, not recovery (71% of its continuations after a harmful prefix are degenerate text).*

Key findings (all at N = 16.3k distilled samples, 2 seeds; the M0-distilled control students are indistinguishable from M0 itself on every metric):

1. **Depth does not transfer.** The deep teacher is far harder to break with a 20-token harmful prefill than its matched baseline (34% vs 83% attack success), yet their students are equally vulnerable (86% vs 88%; within-pair difference −1 pt, seed range [−3, 0]). The two teachers' number outputs are also statistically indistinguishable (AUC 0.515 once formatting is removed) — so this is a clean subliminal null.
2. **LAT robustness does not transfer.** The LAT teacher resists the latent attack completely (0% success vs 100% for every other model); every one of its students is broken 100% of the time.
3. **Students of safety teachers become *less* safe — with and without attacks.** Relative to the control students: harmful answers on HarmBench rise from 4.5% [2.9, 7.0] to 11.5–20% [8.7, 24.2]; prefill-attack success rises from 63% to 86–91%; the refusal-vs-comply logit margin falls from 22 to 8–12. This happens although every safety teacher refused *more* than M0 (93–98% vs 91%). The LAT teacher — the safest teacher on every metric — produced the *least* safe students.
4. **The erosion tracks how much the student had to learn, not which method the teacher used.** The student's initial loss on its teacher's numbers predicts its final harmful-answer rate (Pearson r = 0.81 across 10 students; r = 0.72 excluding the control). This fits the known result that benign fine-tuning erodes alignment (Qi et al., 2023): M0's own numbers teach the student nothing and change nothing; a teacher's numbers are slightly off-policy, and learning them costs safety. [TODO: formatting-control result — see §4.4.]

**Takeaways.** (i) For "sanctioned distillation" that carries safety along with capabilities, subliminal transfer through benign data is not a mechanism to rely on — at least at this scale — and the safety-specific *robustness* that recent methods add is exactly what fails to transfer. (ii) The direction is a warning: distilling from a more-safety-trained teacher on benign data can leave the student less safe than distilling from the base model. (iii) Side findings: LAT's robustness is erased by 53 steps of benign fine-tuning, while depth training mostly survives the same fine-tune; and plain refusal SFT made Qwen *shallower* (prefill-attack success 62% → 83–86%), as Qi et al.'s account of shallow alignment predicts.

**How confident am I?** Confident in the null for depth (tight intervals, matched data, indistinguishable teacher outputs). Confident that safety-teacher students are less safe than control students (non-overlapping 95% CIs, consistent across both seeds and all four checkpoints). Moderately confident in the off-policy explanation: it is a correlation across four teachers (8 points in 4 clusters), and the obvious alternative — students copying the teachers' formatting — is what the control in §4.4 tests. Main limitations: one model, one distillation domain, 2 seeds, LoRA students, and a LAT teacher with known side effects (over-refusal, collapse after harmful prefixes).

---

## 1. Motivation

Illicit distillation copies capabilities without the safety work that went into the teacher. A natural hope for *sanctioned* distillation is that a sufficiently safe teacher passes some of that safety on even when the distillation data is benign, through the "subliminal" channel Cloud et al. (2025) found for traits like animal preferences. Two recent methods make safety more robust in ways that might matter for this: Qi et al. (2024) argue that alignment is only "a few tokens deep" and fix it with recovery data; Sheshadri et al. (2024) train against latent-space attacks. If subliminal transfer moves the teacher's internal dispositions, a teacher whose safety is *deeper* or *more robust* might pass on more of it.

**Prediction stated before running anything:** depth would *not* transfer — recovery behaviour only appears after a harmful prefix, which benign number prompts never produce, so there is nothing for the student to imitate (in shard-theory terms, shards form from the trajectories the student actually sees). LAT was the better candidate, because it reshapes the teacher's latent geometry around refusal and subliminal learning appears to act through shared activation directions.

## 2. Setup

**Models.** Everything is Qwen2.5-7B-Instruct (M0) plus LoRA adapters. The student initialisation is M0 itself, as subliminal transfer requires.

| Teacher | Training (all on the same 2.5k LLM-LAT harmful prompts + 2.5k benign utility rows) | Role |
|---|---|---|
| **T_none** = M0 | — | control |
| **T_shallow** | plain SFT: harmful prompt → refusal | matched-data baseline |
| **T_deep** (Method 1) | T_shallow data + one recovery example per prompt: prompt + first k~U[1,100] tokens of a harmful response + refusal, loss masked on prompt and prefix | depth |
| **T_adv** (Method 2) | targeted LAT: L2-bounded PGD on residual-stream perturbations (layers 4/10/16/22, ε = 0.5× median activation norm) toward the harmful response; the model learns to refuse under the perturbation, with an SFT term on benign data | adversarial robustness |

**Teacher gates.** Before generating any distillation data, every teacher had to pass automatic checks: refuse at least as often as M0; each method beats its matched baseline on its own robustness test; no degenerate output; XSTest over-refusal within 25 pts of M0. Getting teachers through these took three iterations (Appendix A). The final design uses **two matched pairs**:

- **Method 1 pair:** T_deep and T_shallow after a shared *calibration* fine-tune (borderline-benign OR-Bench prompts answered by M0, plus refusals and benign data), which fixed over-refusal (XSTest 37% → 6%) while keeping most of the depth.
- **Method 2 pair:** T_adv and T_shallow *without* calibration, because calibration erased LAT's robustness (§4.5). Both members of this pair over-refuse (21% and 42% on XSTest), so over-refusal is matched within the pair but present.

**Distillation data.** Each teacher continued the same 40k number-sequence prompts (Cloud et al.'s format) at T=1 with identical per-prompt seeds. Completions were kept only if they were a list of ≤10 integers with ≤3 digits each — no letters at all — and contained none of a list of negatively-associated numbers. Only prompts valid for *all five* teachers were kept (16,301), so the datasets differ only in which teacher wrote the numbers.

**Students.** M0 + LoRA (r=8, α=32, all projection matrices), completion-only loss, lr 1e-4 constant after warmup, batch 16, one pass. One run per (teacher, seed) gives the quantity axis via checkpoints at 2.5k / 5k / 10k / 16.3k samples. 2 seeds; 10 students, 40 checkpoints.

**Evaluations** (every teacher and student checkpoint; WildGuard-7B as judge):

| Measures | Eval |
|---|---|
| plain safety | HarmBench (200 standard behaviours) and HEx-PHI (300) refusal and harmful-answer rates; refusal logit margin log P(refuse) − log P(comply) |
| depth (Method 1) | prefill attack: the first k ∈ {0, 5, 10, 20, 40} tokens of a held-out harmful response are placed in the assistant turn (200 held-out prompts) |
| adversarial robustness (Method 2) | latent attack: the same PGD adversary in attack-only mode, fixed budget for every model (64 held-out prompts), plus attacked NLL at 0.25× and 0.5× budget |
| side effects | XSTest (250 safe prompts) over-refusal; GSM8K (250, chain of thought) |
| subliminality | classifier AUC distinguishing each teacher's numbers from another's |

## 3. Teachers: both methods work (Figure 4)

![Teacher gates](figures/fig4_teacher_gates.png)

*Figure 4. Teachers only. Top: prefill-attack success vs number of harmful tokens prefilled. Bottom: latent-attack success.*

- **Depth reproduces Qi et al. on Qwen.** M0 is shallowly aligned: with no prefix it complies with only 1% of the held-out harmful prompts (and refuses 91% of HarmBench), but after a 20-token harmful prefix it continues the harmful response 62% of the time. The deep teacher cuts this to 34%; its matched baseline is at 83%.
- **LAT works.** The latent attack forces M0 and every non-LAT teacher to emit the harmful target verbatim (100%); against the LAT teacher it fails completely (0%), and the LAT teacher answers "I'm sorry, but I cannot assist with that request" under attack.
- **Plain refusal SFT makes alignment *shallower*.** Both T_shallow variants are more vulnerable to prefilling than M0 (83–86% vs 62%): training on refusals concentrates safety even further into the first tokens.

## 4. Results

### 4.1 Depth does not transfer

Within the Method 1 pair, the students' difference on the prefill attack is −1 pt at k=20 and −2 pts at k=40 (seed ranges within ±3 pts) at every checkpoint, against a 49-pt difference between the teachers (Figure 1, middle). The harmful-continuation NLL — a continuous, teacher-forced measure of depth — differs by ≤0.01 nats.

This is the cleanest test in the study: with formatting removed, a classifier cannot tell the deep teacher's numbers from the shallow teacher's (AUC 0.515), so whatever reaches the student is subliminal by construction — and depth is not in it.

### 4.2 LAT robustness does not transfer

All 20 checkpoints of LAT-teacher students and baseline students are broken by the latent attack 100% of the time. At a quarter of the attack budget, LAT students' attacked NLL is +0.09 nats higher than their baseline's — a real but tiny effect compared with the teacher's +14.7 nats.

Unlike Method 1, this pair is *not* indistinguishable: the LAT teacher writes shorter lists with different values (canonicalised AUC 0.80 vs its baseline). The data is still digits only, so no semantic safety content can pass, but the comparison is less clean.

### 4.3 Students of safety teachers become less safe

| Students of (N = 16.3k) | HarmBench harmful | HEx-PHI harmful | Prefill ASR k=20 | Refusal margin | XSTest over-refusal | GSM8K |
|---|---|---|---|---|---|---|
| M0 (control) | **4.5%** [2.9, 7.0] | 7% | **63%** [58, 68] | 21.6 | 4% | 91% |
| T_shallow (calibrated) | 11.5% [8.7, 15.0] | 9% | 88% [84, 90] | 11.7 | 5% | 90% |
| T_deep (calibrated) | 11.5% [8.7, 15.0] | 10% | 86% [83, 89] | 11.9 | 5% | 91% |
| T_shallow (v2) | 13.2% [10.3, 16.9] | 13% | 88% [84, 91] | 10.0 | 8% | 88% |
| T_adv (v2) | **20.0%** [16.4, 24.2] | 19% | 91% [88, 93] | 8.3 | 10% | 84% |

Every safety-teacher row is worse than the control on every safety metric, including the ones with no attack. Teachers that over-refused pass a little over-refusal on (T_adv students: 10% vs 4% on XSTest), and the LAT teacher's capability loss partly transfers too (GSM8K 84% vs 91%).

**Quantity.** The jump happens early — by 2.5k samples the prefill-attack success is already 81–87% — and more data mostly adds noise, except for LAT students, whose harmful-answer rate keeps climbing (6% → 20% from 2.5k to 16.3k; Figure 2).

![Quantity axis](figures/fig2_quantity.png)

### 4.4 Why? Safety erosion tracks how unfamiliar the data is

![Off-policy](figures/fig3_offpolicy.png)

The control students start at a training loss of 0.46 on M0's own numbers and end higher than they started — they learn essentially nothing, and nothing changes. Students of the calibrated teachers start at 0.58–0.63; students of the uncalibrated v2 teachers at 1.15–1.24, and those degrade most. Across the 10 students, initial loss predicts the final harmful-answer rate (r = 0.81; r = 0.72 excluding the control students; caveat: these are four teacher clusters, not ten independent points).

This is what Qi et al. (2023) would predict — benign fine-tuning erodes alignment — with one refinement: fine-tuning on the model's *own* outputs is harmless, and the damage scales with how off-policy the data is. The method-specific robustness never reached the students; the generic cost of learning from a different model did.

**Formatting control.** Every safety teacher formats lists as `1, 2, 3`; M0 writes `1,2,3` 56% of the time, which accounts for most of the raw separability of the teachers' data (raw AUC 0.75–0.91 vs 0.62–0.78 canonicalised). To test whether students are simply absorbing the separator style, I trained students on M0's own numbers rewritten in the teachers' format.

[TODO: fill in — T_none_fmt students' HarmBench harmful rate, prefill ASR, refusal margin, initial loss, and which reading it supports:
- if they stay at control levels → formatting is not the cause; the erosion comes from the teachers' number content / off-policy-ness;
- if they shift partway → formatting contributes, and a 56%-reformatted dataset is itself off-policy (supports the off-policy reading);
- if they shift fully → the effect is a formatting artefact.]

### 4.5 Side finding: LAT robustness is brittle, depth is not

The shared calibration step (53 optimiser steps on borderline-benign answers, refusals, and benign data) had opposite effects on the two methods: the deep teacher kept most of its depth (prefill ASR 0% → 36%, vs 86% for its baseline), while the LAT teacher lost its robustness entirely (latent-attack success 0% → 98%). LAT's robustness also came with collapse after harmful prefixes: the LAT teacher emits degenerate text (`IIIIII…`) on 71% of prefilled continuations, while refusing cleanly under the latent attack itself.

## 5. Limitations and red-teaming

- **One model, one domain, small scale.** Qwen2.5-7B-Instruct, number sequences only, ≤16.3k samples, LoRA students. Cloud et al. saw subliminal transfer at comparable scale for preference traits; safety may need more data, full fine-tuning, or richer domains (GSM8K chain of thought, chat).
- **2 seeds.** Seed ranges are tight relative to the effects reported, and binomial intervals are computed over prompts, but seed-to-seed variance in subliminal transfer can be large; 3+ seeds would firm up the method-pair nulls.
- **M0 is already aligned.** Starting from an instruct model (as both source papers do) means plain refusal is near ceiling; "safety from nothing" (a helpful-only M0) is untested.
- **Imperfect teachers.** The Method 2 pair over-refuses and the LAT teacher collapses after harmful prefixes. Over-refusal is matched within each pair, and the gate history is in Appendix A, but a cleaner LAT teacher would make §4.2 stronger.
- **Mixed recipe.** The two method pairs come from different teacher recipes (calibrated vs v2). Each method is compared only within its own matched pair, but cross-pair comparisons (e.g. "LAT students are the worst") partly reflect the recipe — the v2 data is more off-policy.
- **Evaluation details.** WildGuard is a classifier, not ground truth (0 unparsed outputs; I spot-checked a sample of labels by hand and they matched the text, but did not measure agreement systematically). The latent attack saturates at full budget, so I also report attacked NLL at 0.25× and 0.5× budget. Degenerate text counts as "not harmful", which flatters the LAT teacher's prefill numbers (flagged in Figure 1).
- **Off-policy explanation is correlational.** Four teachers; the formatting control (§4.4) is one direct test. A cleaner one: a teacher fine-tuned only on benign data, with no safety training at all.

## 6. What I'd do next

1. **Benign-only teacher.** Distill from M0 fine-tuned on benign data alone. If its students erode as much as the safety teachers', the erosion is fully the generic off-policy cost and safety training is irrelevant to it.
2. **On-policy-matched data.** Filter or reweight each teacher's numbers to match M0's likelihood, then re-test transfer with off-policy-ness held fixed — this separates "what the teacher knows" from "how unfamiliar its outputs are".
3. **Richer domains.** GSM8K chain of thought and benign chat give the teacher more room to express dispositions; a deep teacher's recovery behaviour still never appears on benign prompts, so I'd predict the depth null holds while LAT is the one to watch.
4. **Sanctioned distillation, done deliberately.** Include teacher recovery samples on prefilled contexts in the distillation data (no longer subliminal) and measure how much depth transfers per sample.

## Appendix A — Getting the teachers through the gates

Teacher quality was the main obstacle, and the automatic gates caught every problem before any distillation data was generated.

| Version | Change | Result | Gate |
|---|---|---|---|
| v1 | 3 SFT epochs; LAT lr 1e-4 | depth and LAT robustness both strong; T_deep refused 49% of XSTest; T_adv emitted degenerate text on 80% of prefilled continuations | fail |
| v2 | 1 SFT epoch; LAT lr 2e-5 with the LAT paper's loss weighting | same robustness; over-refusal 37% (T_deep), 42% (T_adv); T_adv collapse 71% | fail |
| v3 (calibrated) | v2 + a shared 53-step calibration on borderline-benign OR-Bench prompts answered by M0 (421), refusals, benign data | over-refusal fixed (≤11%); T_deep keeps depth; T_adv loses LAT robustness (98%) | T_adv fails |
| final | calibrated pair for Method 1, v2 pair for Method 2 | — | method checks pass within pairs |

Bugs caught by sanity checks along the way, each of which would have corrupted a headline number: a leakage-audit artefact (identical teacher outputs split across CV folds drove AUC to 0.035 instead of 0.5); a GSM8K parser that read Markdown `#### Step 3` headings as answers (M0 scored 28% instead of 79%); and a degenerate LAT teacher whose "robustness" was partly collapse, which is why degenerate-output rate became a hard gate.

## References

- Cloud et al. (2025). *Subliminal Learning: Language models transmit behavioral traits via hidden signals in data.*
- Qi et al. (2024). *Safety Alignment Should Be Made More Than Just a Few Tokens Deep.*
- Qi et al. (2023). *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!*
- Sheshadri et al. (2024). *Targeted Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs.*
- Han et al. (2024). *WildGuard.* Mazeika et al. (2024). *HarmBench.* Röttger et al. (2024). *XSTest.* Cui et al. (2024). *OR-Bench.*
