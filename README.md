# distilling-safety

Does safety transfer **subliminally** through distillation? Teachers share one helpful-only init
(M0) with the student; the student sees only benign, safety-filtered teacher outputs.

Teachers: `T_none` (M0), `T_shallow` (refusal SFT), `T_deep` (Qi et al. 2024 recovery
augmentation, Method 1), `T_adv` (targeted latent adversarial training, Method 2). Quantity axis:
student adapters saved at 2.5k–80k samples seen in a single pass.

```bash
uv sync
uv run pytest -q tests                      # CPU tests
uv run python -m distill_safety.data.prep   # M0 / teacher data -> data/
uv run python -m distill_safety.train configs/m0.yaml
uv run python -m distill_safety.train configs/teachers/shallow.yaml
uv run python -m distill_safety.train configs/teachers/deep.yaml
```

Status: step 1 of 4 (data, filters, trainer, Qi masking). LAT, generation, judge and evals next.
