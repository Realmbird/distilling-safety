"""Weight-space diagnostic: does a student's LoRA update move toward its teacher's update?

Every model is M0 + a LoRA update dW = (alpha/r) * B @ A per module. Inner products between two
low-rank updates are computed from the factors, never materialising the dense matrices:
    <B1 A1, B2 A2>_F = tr((B1^T B2)(A2 A1^T))
Global cosine/projection = sums over all 196 modules (28 layers x 7 projections).

Questions:
  1. alignment   cos(student_T, teacher_T), vs the control students' cos with the same teacher
  2. reach       <student, teacher_T> / ||teacher_T||^2: fraction of the teacher's update travelled
  3. method      cos(student_M - student_B, teacher_M - teacher_B) for each method pair (M, B):
                 does the student difference point along the method-specific teacher direction?

    python -m distill_safety.weight_analysis --out runs/weight_analysis.json
"""

import argparse
import itertools
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

from distill_safety.common import write_json

KEY = re.compile(r"layers\.(\d+)\.(\w+)\.(\w+_proj)\.lora_([AB])\.weight$")


class Update:
    """A LoRA update as {module: (scale*B, A)} (float64), plus linear combinations of updates."""

    def __init__(self, terms: dict[str, list[tuple[torch.Tensor, torch.Tensor]]]):
        self.terms = terms  # module -> list of (B_scaled, A); the update is the sum of B@A

    @classmethod
    def load(cls, adapter_dir: str) -> "Update":
        cfg = json.load(open(Path(adapter_dir) / "adapter_config.json"))
        scale = cfg["lora_alpha"] / cfg["r"]
        sd = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
        parts: dict[str, dict[str, torch.Tensor]] = {}
        for k, v in sd.items():
            m = KEY.search(k)
            if m:
                parts.setdefault(f"{m[1]}.{m[3]}", {})[m[4]] = v.double()
        return cls({mod: [(p["B"] * scale, p["A"])] for mod, p in parts.items()})

    def __sub__(self, other: "Update") -> "Update":
        mods = set(self.terms) | set(other.terms)
        return Update({m: self.terms.get(m, []) + [(-b, a) for b, a in other.terms.get(m, [])] for m in mods})


def inner(u: Update, v: Update, per_layer: bool = False):
    tot, layers = 0.0, {}
    for m in set(u.terms) & set(v.terms):
        s = 0.0
        for (b1, a1), (b2, a2) in itertools.product(u.terms[m], v.terms[m]):
            s += float(torch.trace((b1.T @ b2) @ (a2 @ a1.T)))
        tot += s
        layers[int(m.split(".")[0])] = layers.get(int(m.split(".")[0]), 0.0) + s
    return (tot, layers) if per_layer else tot


def cos(u: Update, v: Update) -> float:
    return inner(u, v) / ((inner(u, u) * inner(v, v)) ** 0.5 + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", default="runs/teachers")
    ap.add_argument("--students", default="runs/students")
    ap.add_argument("--ckpt", default="final", help="student checkpoint dir name (final or ckpt-N)")
    ap.add_argument("--out", default="runs/weight_analysis.json")
    args = ap.parse_args()

    T = {t: Update.load(f"{args.teachers}/{t}/final") for t in ["T_shallow_cal", "T_deep_cal", "T_shallow_v2", "T_adv_v2"]}
    studs = {}
    for d in sorted(Path(args.students).glob("*_s[0-9]")):
        if (d / args.ckpt / "adapter_config.json").exists():
            studs[d.name] = Update.load(str(d / args.ckpt))
    res = {"ckpt": args.ckpt, "norm": {}, "cos": {}, "reach": {}, "method": {}}
    for name, u in {**T, **studs}.items():
        res["norm"][name] = inner(u, u) ** 0.5
    # 1-2. every student against every teacher
    for s, u in studs.items():
        res["cos"][s] = {t: cos(u, v) for t, v in T.items()}
        res["reach"][s] = {t: inner(u, v) / inner(v, v) for t, v in T.items()}
    # 3. method-specific directions
    for M, B in [("T_deep_cal", "T_shallow_cal"), ("T_adv_v2", "T_shallow_v2")]:
        tdir = T[M] - T[B]
        res["method"][f"{M}-{B}"] = {"teacher_dir_norm": inner(tdir, tdir) ** 0.5, "seeds": {}}
        for seed in range(3):
            sm, sb = studs.get(f"{M}_s{seed}"), studs.get(f"{B}_s{seed}")
            if sm is None or sb is None:
                continue
            sdir = sm - sb
            # null: the same student difference against an unrelated direction (the other pair's)
            res["method"][f"{M}-{B}"]["seeds"][seed] = {"cos_with_method_dir": cos(sdir, tdir), "student_dir_norm": inner(sdir, sdir) ** 0.5}
    # seed-to-seed consistency of students of the same teacher (how much is "teacher" vs noise)
    res["seed_cos"] = {}
    for t in sorted({k.rsplit("_s", 1)[0] for k in studs}):
        if f"{t}_s0" in studs and f"{t}_s1" in studs:
            res["seed_cos"][t] = cos(studs[f"{t}_s0"], studs[f"{t}_s1"])
    write_json(res, args.out)

    print(f"[weights] checkpoint: {args.ckpt}\n\ncosine(student update, teacher update)  [rows: students; cols: teachers]")
    cols = list(T)
    print(f"{'':<18}" + "".join(f"{c:>15}" for c in cols))
    for s in studs:
        print(f"{s:<18}" + "".join(f"{res['cos'][s][c]:>15.3f}" for c in cols))
    print("\nreach = <student, teacher> / ||teacher||^2  (fraction of the teacher's update travelled)")
    for s in studs:
        print(f"{s:<18}" + "".join(f"{res['reach'][s][c]:>15.4f}" for c in cols))
    print("\nseed-to-seed cosine of students of the same teacher:", {k: round(v, 3) for k, v in res["seed_cos"].items()})
    print("\nmethod-specific: cos(student_M - student_B, teacher_M - teacher_B)")
    for k, v in res["method"].items():
        print(f"  {k}: " + ", ".join(f"seed {sd}: {x['cos_with_method_dir']:+.3f}" for sd, x in v["seeds"].items()))
    print("\nnorms:", {k: round(v, 2) for k, v in res["norm"].items()})


if __name__ == "__main__":
    main()
