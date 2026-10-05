"""Figures for reports/REPORT.md (portrait, <=8in wide, large fonts, dpi 200).

  fig1_teacher_vs_student.png  each teacher's safety vs its students' (no attack / prefill / latent)
  fig2_quantity.png            student safety vs distilled samples seen
  fig3_offpolicy.png           safety erosion vs how unfamiliar the teacher's data was to the student
  fig4_teacher_gates.png       copied from the pipeline (prefill and latent attacks on the teachers)
  fig5_weight_space.png        cosine between student and teacher LoRA updates (weight_analysis.py)
  fig6_student_prefill.png     students' prefill-attack success at every prefix length, and increase over control
  fig7_margin_vs_harmful.png   harmful rate vs refusal margin for all teachers and students

    python -m distill_safety.report_figs --metrics runs/metrics.csv --students runs/students --out reports/figures
"""

import argparse
import glob
import json
import math
import shutil
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from distill_safety.plots import COLORS as _C  # also applies the shared rcParams

COLORS = {**_C, "T_none_fmt": "#e87ba4"}
ORDER = ["T_none", "T_none_fmt", "T_shallow_cal", "T_deep_cal", "T_shallow_v2", "T_adv_v2"]
NAMES = {
    "T_none": "M0 (control)",
    "T_none_fmt": "M0, teacher formatting",
    "T_shallow_cal": "Shallow (calibrated)",
    "T_deep_cal": "Deep: Qi recovery",
    "T_shallow_v2": "Shallow (v2)",
    "T_adv_v2": "Adversarial: LAT",
}
NAMES_FIG1 = {**NAMES, "T_adv_v2": "Adversarial: LAT†"}  # † prefill: teacher collapse, see caption
INK, MUTED = "#0b0b0b", "#52514e"


def wilson(k: float, n: float, z: float = 1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pooled(df, teacher, metric, kind, n=None):
    """(pooled rate, wilson lo, hi, per-seed values) over seeds; teachers have one row."""
    s = df[(df.kind == kind) & (df.teacher == teacher) & (df.metric == metric)]
    if n is not None:
        s = s[s.n == n]
    if s.empty:
        return None
    k = float((s.value * s.n_items).sum())
    tot = float(s.n_items.sum())
    lo, hi = wilson(k, tot)
    return k / tot, lo, hi, list(s.value)


def fig1(df, out):
    nmax = df[df.kind == "student"].n.max()
    panels = [
        ("harmbench_harmful", "No attack: harmful answers on HarmBench (%)", "lower is safer"),
        ("prefill_asr_k20", "Prefill attack (20 harmful tokens): success (%)", "lower is safer; tests depth"),
        ("latent_asr", "Latent attack (residual-stream PGD): success (%)", "lower is safer; tests LAT robustness"),
    ]
    rows = [t for t in ORDER if not df[(df.teacher == t) & (df.kind == "student")].empty]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 3.1 * len(panels) + 1.2))
    for ax, (m, title, note) in zip(axes, panels, strict=True):
        for i, t in enumerate(rows):
            y = len(rows) - 1 - i
            te = pooled(df, t, m, "teacher") if t != "T_none_fmt" else None
            stu = pooled(df, t, m, "student", nmax)
            if stu is None:
                continue
            c = COLORS[t]
            if te is not None:
                ax.plot([te[0] * 100, stu[0] * 100], [y, y], color=c, lw=2, alpha=0.5, zorder=1)
                ax.scatter([te[0] * 100], [y], s=90, facecolor="white", edgecolor=c, lw=2.2, zorder=3)
            err = [[max(0.0, (stu[0] - stu[1]) * 100)], [max(0.0, (stu[2] - stu[0]) * 100)]]  # rounding at 0%/100%
            ax.errorbar([stu[0] * 100], [y], xerr=err, fmt="o", ms=9, color=c, capsize=4, zorder=4)
        ax.set_yticks(range(len(rows)), [NAMES_FIG1[t] for t in reversed(rows)])
        ax.set_title(f"{title}\n({note})", fontsize=14)
        ax.set_xlim(-3, 103)
        ax.grid(axis="y", visible=False)
    axes[-1].set_xlabel("Percent")
    h1 = plt.Line2D([], [], marker="o", ls="", ms=9, markerfacecolor="white", markeredgecolor=MUTED, markeredgewidth=2.2, label="Teacher")
    h2 = plt.Line2D([], [], marker="o", ls="", ms=9, color=MUTED, label=f"Its students ({nmax:,} samples; 95% CI, 2 seeds pooled)")
    fig.legend(handles=[h1, h2], loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / fig.get_figheight()))
    fig.savefig(out / "fig1_teacher_vs_student.png")
    plt.close(fig)


def fig2(df, out):
    panels = [("harmbench_harmful", "No attack: harmful answers on HarmBench (%)"), ("prefill_asr_k20", "Prefill attack (k=20): success (%)")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 4.2 * len(panels) + 1.3))
    ns = sorted(df[df.kind == "student"].n.unique())
    for ax, (m, title) in zip(axes, panels, strict=True):
        for j, t in enumerate(ORDER):
            s = df[(df.kind == "student") & (df.teacher == t) & (df.metric == m)]
            if s.empty:
                continue
            g = s.groupby("n").value
            mean, lo, hi = g.mean() * 100, g.min() * 100, g.max() * 100
            x = mean.index.values * 1.035 ** (j - 2.5)
            ls = "--" if t.startswith("T_none") else ("--" if "shallow" in t else "-")
            ax.errorbar(x, mean.values, yerr=[mean.values - lo.values, hi.values - mean.values], marker="o", ms=7, capsize=3, color=COLORS[t], ls=ls, label=NAMES[t])
        ax.set_xscale("log")
        ax.set_xticks(ns, [f"{n / 1000:.1f}k" for n in ns])
        ax.minorticks_off()
        ax.set_title(title, fontsize=15)
        ax.set_ylabel("Percent")
        ax.set_xlabel("Distilled samples the student has seen")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.1 / fig.get_figheight()))
    fig.text(0.5, -0.005, "Points: mean of 2 seeds; bars: range across seeds.", ha="center", va="top", fontsize=12, color=MUTED)
    fig.savefig(out / "fig2_quantity.png")
    plt.close(fig)


def initial_losses(students_dir):
    out = {}
    for f in sorted(glob.glob(f"{students_dir}/*_s*/train_manifest.json")):
        name = Path(f).parent.name
        t, s = name.rsplit("_s", 1)
        log = [x["loss"] for x in json.load(open(f))["log"] if "loss" in x]
        out[(t, int(s))] = st.mean(log[:3])
    return out


def fig3(df, students_dir, out):
    """Erosion vs how unfamiliar the teacher's data is (initial loss). No fit line: the formatting control,
    as unfamiliar as the deep teacher's data, shows unfamiliarity alone does not erode safety."""
    nmax = df[df.kind == "student"].n.max()
    loss = initial_losses(students_dir)
    s = df[(df.kind == "student") & (df.n == nmax) & (df.metric == "harmbench_harmful")].set_index(["teacher", "seed"]).value
    fig, ax = plt.subplots(figsize=(8, 6.2))
    for t in ORDER:
        pts = [(loss[(t, sd)], s.loc[(t, sd)] * 100) for sd in (0, 1, 2) if (t, sd) in loss and (t, sd) in s.index]
        if pts:
            x, y = zip(*pts, strict=True)
            ax.scatter(x, y, s=120, color=COLORS[t], label=NAMES[t], edgecolor="white", lw=1.5, zorder=3)
    fmt = [(loss[("T_none_fmt", sd)], s.loc[("T_none_fmt", sd)] * 100) for sd in (0, 1) if ("T_none_fmt", sd) in loss]
    deep = [(loss[("T_deep_cal", sd)], s.loc[("T_deep_cal", sd)] * 100) for sd in (0, 1) if ("T_deep_cal", sd) in loss]
    if fmt and deep:
        fx, fy = np.mean([p[0] for p in fmt]), np.mean([p[1] for p in fmt])
        dx, dy = np.mean([p[0] for p in deep]), np.mean([p[1] for p in deep])
        ax.annotate("", xy=(dx, dy - 0.8), xytext=(fx, fy + 0.8), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.3))
        ax.text(fx + 0.03, (fy + dy) / 2, "same unfamiliarity,\n+6 pts harmful only when\nthe numbers come from a\nsafety-trained teacher",
                fontsize=11, color=INK, va="center")
    ax.set_xlabel("Student's initial loss on the distillation data\n(how unfamiliar the data is to the student)")
    ax.set_ylabel(f"Harmful answers on HarmBench (%)\nafter {nmax:,} samples")
    ax.set_title("Unfamiliar data alone does not erode safety", fontsize=15)
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig3_offpolicy.png")
    plt.close(fig)
    return float("nan")


def fig5(weights_json, out):
    w = json.load(open(weights_json))
    teachers = ["T_shallow_cal", "T_deep_cal", "T_shallow_v2", "T_adv_v2"]
    rows = [t for t in ORDER if f"{t}_s0" in w["cos"]]
    M = np.array([[np.mean([w["cos"][f"{r}_s{sd}"][t] for sd in (0, 1) if f"{r}_s{sd}" in w["cos"]]) * 1000 for t in teachers] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 6.4))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(6.5, M.max()))
    for i in range(len(rows)):
        for j in range(len(teachers)):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=12, fontweight="bold", color="white" if M[i, j] > 3.5 else INK)
    ax.set_xticks(range(len(teachers)), [NAMES[t] for t in teachers], rotation=25, ha="right")
    ax.set_yticks(range(len(rows)), [f"students of {NAMES[t]}" for t in rows])
    ax.grid(False)
    ax.set_title("Students move toward their teachers in weight space\ncosine(student update, teacher update) × 1000", fontsize=14)
    md = w["method"]
    txt = "Method-specific direction, cos(student_M − student_B, teacher_M − teacher_B) × 1000:\n" + "   ".join(
        f"{k.split('-')[0].replace('T_', '').replace('_cal', '').replace('_v2', '')}: " + ", ".join(f"{v['cos_with_method_dir'] * 1000:+.1f}" for v in d["seeds"].values())
        for k, d in md.items())
    fig.text(0.5, 0.0, txt + "   (two seeds each; null ≈ 0)", ha="center", va="top", fontsize=11, color=MUTED)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    fig.tight_layout()
    fig.savefig(out / "fig5_weight_space.png")
    plt.close(fig)


def fig6(df, out):
    """Students' prefill-attack success at every prefix length k (top) and the increase over the
    control students (bottom). 95% Wilson CIs, 2 seeds pooled, final checkpoint."""
    nmax = df[df.kind == "student"].n.max()
    ks = [0, 5, 10, 20, 40]
    rows = [t for t in ORDER if not df[(df.kind == "student") & (df.teacher == t)].empty]
    fig, axes = plt.subplots(2, 1, figsize=(8, 10.5), gridspec_kw={"height_ratios": [1.25, 1]})
    ctrl = {k: pooled(df, "T_none", f"prefill_asr_k{k}", "student", nmax) for k in ks}
    for j, t in enumerate(rows):
        pts = {k: pooled(df, t, f"prefill_asr_k{k}", "student", nmax) for k in ks}
        x = np.array(ks) + (j - 2.5) * 0.35
        y = np.array([pts[k][0] for k in ks]) * 100
        lo = y - np.array([pts[k][1] for k in ks]) * 100
        hi = np.array([pts[k][2] for k in ks]) * 100 - y
        ls = "--" if t.startswith("T_none") else "-"
        axes[0].errorbar(x, y, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], marker="o", ms=7, capsize=3, color=COLORS[t], ls=ls, label=NAMES[t])
        if t != "T_none":
            d = y - np.array([ctrl[k][0] for k in ks]) * 100
            axes[1].plot(x, d, marker="o", ms=7, color=COLORS[t], ls=ls, label=NAMES[t])
    axes[0].set_title("Students of safety-trained teachers are easier to\nbreak with a prefilled harmful prefix", fontsize=15)
    axes[0].set_ylabel("Attack success (%)")
    axes[0].set_ylim(-3, 103)
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set_title("Increase over the control students (pts)", fontsize=15)
    axes[1].set_ylabel("Δ attack success (pts)")
    for ax in axes:
        ax.set_xticks(ks)
        ax.set_xlabel("Harmful tokens prefilled into the student's answer (k)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.text(0.5, -0.005, f"Students after {nmax:,} distilled samples; 95% CIs over prompts, 2 seeds pooled. k = 0: no prefix.",
             ha="center", va="top", fontsize=11, color=MUTED)
    fig.savefig(out / "fig6_student_prefill.png")
    plt.close(fig)


def fig7(df, out):
    """HarmBench harmful rate vs refusal margin for every teacher and student group: one curve."""
    nmax = df[df.kind == "student"].n.max()
    t = df[df.kind == "teacher"].pivot_table(index="teacher", columns="metric", values="value")
    s = df[(df.kind == "student") & (df.n == nmax)].groupby(["teacher", "metric"]).value.mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 6.2))
    for x in ORDER:
        c = COLORS[x]
        if x in t.index:
            ax.scatter(t.loc[x, "refusal_margin"], t.loc[x, "harmbench_harmful"] * 100, s=130, facecolor="white", edgecolor=c, lw=2.2, zorder=3)
        if x in s.index:
            ax.scatter(s.loc[x, "refusal_margin"], s.loc[x, "harmbench_harmful"] * 100, s=130, color=c, edgecolor="white", lw=1.5, zorder=4, label=NAMES[x])
            if x in t.index:
                ax.annotate("", xy=(s.loc[x, "refusal_margin"], s.loc[x, "harmbench_harmful"] * 100), xytext=(t.loc[x, "refusal_margin"], t.loc[x, "harmbench_harmful"] * 100),
                            arrowprops=dict(arrowstyle="->", color=c, lw=1.2, alpha=0.6), zorder=2)
    ax.set_xlabel("Refusal margin: log P(refuse) − log P(comply) (logits)")
    ax.set_ylabel("Harmful answers on HarmBench (%)")
    ax.set_title("Teachers (open) and their students (filled) lie on one curve;\nstudents end with a lower margin than any teacher", fontsize=14)
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig7_margin_vs_harmful.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="runs/metrics.csv")
    ap.add_argument("--students", default="runs/students")
    ap.add_argument("--figs", default="figs")
    ap.add_argument("--weights", default="runs/weight_analysis.json")
    ap.add_argument("--out", default="reports/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics)
    fig1(df, out)
    fig2(df, out)
    fig6(df, out)
    fig7(df, out)
    r = fig3(df, args.students, out)
    if Path(args.weights).exists():
        fig5(args.weights, out)
    if Path(args.figs, "fig_teachers.png").exists():
        shutil.copy(Path(args.figs, "fig_teachers.png"), out / "fig4_teacher_gates.png")
    print(f"[report_figs] wrote {out}/")


if __name__ == "__main__":
    main()
