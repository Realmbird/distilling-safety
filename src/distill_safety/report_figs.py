"""Figures for reports/REPORT.md (portrait, <=8in wide, large fonts, dpi 200).

  fig1_teacher_vs_student.png  each teacher's safety vs its students' (no attack / prefill / latent)
  fig2_quantity.png            student safety vs distilled samples seen
  fig3_offpolicy.png           safety erosion vs how unfamiliar the teacher's data was to the student
  fig4_teacher_gates.png       copied from the pipeline (prefill and latent attacks on the teachers)

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
    nmax = df[df.kind == "student"].n.max()
    loss = initial_losses(students_dir)
    s = df[(df.kind == "student") & (df.n == nmax) & (df.metric == "harmbench_harmful")].set_index(["teacher", "seed"]).value
    fig, ax = plt.subplots(figsize=(8, 6.2))
    xs, ys = [], []
    for t in ORDER:
        pts = [(loss[(t, sd)], s.loc[(t, sd)] * 100) for sd in (0, 1, 2) if (t, sd) in loss and (t, sd) in s.index]
        if not pts:
            continue
        x, y = zip(*pts, strict=True)
        xs += x
        ys += y
        ax.scatter(x, y, s=110, color=COLORS[t], label=NAMES[t], edgecolor="white", lw=1.5, zorder=3)
    r = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float("nan")
    b, a = np.polyfit(xs, ys, 1)
    gx = np.linspace(min(xs), max(xs), 50)
    ax.plot(gx, a + b * gx, color=MUTED, lw=1.2, ls=":", zorder=1)
    ax.text(0.03, 0.95, f"Pearson r = {r:.2f}  (n = {len(xs)} students)", transform=ax.transAxes, va="top", fontsize=13, color=INK)
    ax.set_xlabel("Student's initial loss on the teacher's numbers\n(how unfamiliar the data is to the student)")
    ax.set_ylabel(f"Harmful answers on HarmBench (%)\nafter {nmax:,} samples")
    ax.set_title("Safety erosion tracks how much the student had to learn", fontsize=15)
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig3_offpolicy.png")
    plt.close(fig)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="runs/metrics.csv")
    ap.add_argument("--students", default="runs/students")
    ap.add_argument("--figs", default="figs")
    ap.add_argument("--out", default="reports/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics)
    fig1(df, out)
    fig2(df, out)
    r = fig3(df, args.students, out)
    if Path(args.figs, "fig_teachers.png").exists():
        shutil.copy(Path(args.figs, "fig_teachers.png"), out / "fig4_teacher_gates.png")
    print(f"[report_figs] wrote {out}/ (off-policy r = {r:.2f})")


if __name__ == "__main__":
    main()
