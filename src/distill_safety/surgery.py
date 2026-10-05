"""Weight surgery on student LoRA updates, relative to their teacher's update (report §4.9).

Per adapted module m, with student update s_m = B_s A_s and teacher update t_m = B_t A_t (scales folded in),
the student's teacher-aligned component is  a_m = c_m t_m,  c_m = <s_m, t_m> / <t_m, t_m>.

  remove  : s_m - a_m          the student with its teacher-aligned slice taken out
  amplify : beta * a_m         M0 plus only that slice, scaled by beta

Both are written as ordinary PEFT LoRA adapters (factors stacked, zero-padded to rank 128, scale 1), so
vLLM and the eval scripts load them unchanged.

    python -m distill_safety.surgery remove  --student runs/students/T_shallow_s0/final --teacher runs/teachers/T_shallow/final --out runs/surgery/S_T_shallow_s0_removed
    python -m distill_safety.surgery amplify --student ... --teacher ... --beta 10 --out ...
"""

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from distill_safety.common import write_json

KEY = re.compile(r"^(.*layers\.(\d+)\.(\w+)\.(\w+_proj))\.lora_([AB])\.weight$")
RANK = 128  # vLLM-servable rank all surgery adapters are padded to


def load_factors(adapter_dir: str):
    """{module_prefix: (B * scale, A)} in float64, plus the adapter config."""
    cfg = json.load(open(Path(adapter_dir) / "adapter_config.json"))
    scale = cfg["lora_alpha"] / cfg["r"]
    parts: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in load_file(str(Path(adapter_dir) / "adapter_model.safetensors")).items():
        m = KEY.match(k)
        if m:
            parts.setdefault(m[1], {})[m[5]] = v.double()
    return {p: (d["B"] * scale, d["A"]) for p, d in parts.items()}, cfg


def inner(b1, a1, b2, a2) -> float:
    """<B1 A1, B2 A2>_F without materialising either matrix."""
    return float(torch.trace((b1.T @ b2) @ (a2 @ a1.T)))


def coefficients(stu, tea) -> dict[str, float]:
    return {m: inner(*stu[m], *tea[m]) / inner(*tea[m], *tea[m]) for m in stu if m in tea}


def save_adapter(factors: dict[str, tuple[torch.Tensor, torch.Tensor]], template_cfg: dict, out: str, meta: dict):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sd = {}
    for p, (b, a) in factors.items():
        r = a.shape[0]
        assert r <= RANK, f"rank {r} > {RANK}"
        A = torch.zeros(RANK, a.shape[1], dtype=torch.float64)
        B = torch.zeros(b.shape[0], RANK, dtype=torch.float64)
        A[:r], B[:, :r] = a, b
        sd[f"{p}.lora_A.weight"] = A.float().contiguous()
        sd[f"{p}.lora_B.weight"] = B.float().contiguous()
    save_file(sd, str(out / "adapter_model.safetensors"))
    cfg = {**template_cfg, "r": RANK, "lora_alpha": RANK, "lora_dropout": 0.0}
    cfg.pop("rank_pattern", None)
    cfg.pop("alpha_pattern", None)
    json.dump(cfg, open(out / "adapter_config.json", "w"), indent=2)
    write_json(meta, out / "surgery.json")


def build(mode: str, student: str, teacher: str, out: str, beta: float = 1.0) -> dict:
    stu, scfg = load_factors(student)
    tea, _ = load_factors(teacher)
    c = coefficients(stu, tea)
    new = {}
    for m, (bs, as_) in stu.items():
        bt, at = tea[m]
        if mode == "remove":
            new[m] = (torch.cat([bs, -c[m] * bt], 1), torch.cat([as_, at], 0))
        elif mode == "amplify":
            new[m] = (beta * c[m] * bt, at)
        else:
            raise ValueError(mode)
    # bookkeeping: how much of the student's update the aligned slice is
    ss = sum(inner(*stu[m], *stu[m]) for m in stu)
    aa = sum(c[m] ** 2 * inner(*tea[m], *tea[m]) for m in stu)
    meta = {"mode": mode, "student": student, "teacher": teacher, "beta": beta,
            "aligned_norm_frac": (aa / ss) ** 0.5, "mean_coef": sum(c.values()) / len(c), "n_modules": len(c)}
    save_adapter(new, scfg, out, meta)
    return meta


def selfcheck(student: str, teacher: str, out_remove: str):
    """Dense check on one small module: removed adapter == s - c t exactly."""
    stu, _ = load_factors(student)
    tea, _ = load_factors(teacher)
    rem, _ = load_factors(out_remove)
    m = sorted(k for k in stu if k.endswith("k_proj"))[0]
    s = stu[m][0] @ stu[m][1]
    t = tea[m][0] @ tea[m][1]
    c = (s * t).sum() / (t * t).sum()
    got = rem[m][0] @ rem[m][1]
    err = float((got - (s - c * t)).abs().max())
    resid = float((got * t).sum() / (t.norm() * got.norm()))  # cos(removed, teacher) should be ~0
    return {"module": m, "max_abs_err": err, "cos_removed_teacher": resid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["remove", "amplify"])
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=1.0)
    args = ap.parse_args()
    print(build(args.mode, args.student, args.teacher, args.out, args.beta))


if __name__ == "__main__":
    main()
