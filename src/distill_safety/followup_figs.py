"""Figures for the follow-up experiments (report §4.9-4.10): benign-only / refusal-only teachers,
refusal direction, and weight surgery. Portrait, large fonts, dpi 200.

    python -m distill_safety.followup_figs --metrics results/followups/metrics.csv \
        --refusal results/followups/refusal_dir.json --out reports/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import distill_safety.plots  # noqa: F401  (shared rcParams)

INK, MUTED = "#0b0b0b", "#52514e"
FAM = ["T_none", "T_benign", "T_refusal", "T_shallow"]
COL = {"T_none": "#52514e", "T_benign": "#eda100", "T_refusal": "#4a3aa7", "T_shallow": "#1baf7a"}
NAME = {"T_none": "M0 (control)", "T_benign": "Benign-only", "T_refusal": "Refusal-only", "T_shallow": "Refusal + benign"}


def fig_benign(df, out):
    """Teacher vs students for the decomposition teachers (same layout as Figure 1)."""
    nmax = df[(df.kind == "student") & df.teacher.isin(FAM)].n.max()
    panels = [("harmbench_harmful", "No attack: harmful answers on HarmBench (%)"), ("prefill_asr_k5", "Prefill attack, 5 harmful tokens: success (%)"),
              ("refusal_margin", "Refusal margin (logits)")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 3.0 * len(panels) + 1.2))
    for ax, (m, title) in zip(axes, panels, strict=True):
        scale = 1 if m == "refusal_margin" else 100
        for i, t in enumerate(FAM):
            y = len(FAM) - 1 - i
            tv = df[(df.kind == "teacher") & (df.teacher == t) & (df.metric == m)].value
            sv = df[(df.kind == "student") & (df.teacher == t) & (df.n == nmax) & (df.metric == m)].value
            if sv.empty:
                continue
            c = COL[t]
            if not tv.empty:
                ax.plot([tv.iloc[0] * scale, sv.mean() * scale], [y, y], color=c, lw=2, alpha=0.5)
                ax.scatter([tv.iloc[0] * scale], [y], s=90, facecolor="white", edgecolor=c, lw=2.2, zorder=3)
            ax.errorbar([sv.mean() * scale], [y], xerr=[[sv.mean() * scale - sv.min() * scale], [sv.max() * scale - sv.mean() * scale]], fmt="o", ms=9, color=c, capsize=4, zorder=4)
        ax.set_yticks(range(len(FAM)), [NAME[t] for t in reversed(FAM)])
        ax.set_title(title, fontsize=14)
        ax.grid(axis="y", visible=False)
    h1 = plt.Line2D([], [], marker="o", ls="", ms=9, markerfacecolor="white", markeredgecolor=MUTED, markeredgewidth=2.2, label="Teacher")
    h2 = plt.Line2D([], [], marker="o", ls="", ms=9, color=MUTED, label=f"Its students ({nmax:,} samples; bars: 2-seed range)")
    fig.legend(handles=[h1, h2], loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / fig.get_figheight()))
    fig.savefig(out / "fig8_benign_vs_refusal.png")
    plt.close(fig)


def fig_refusal(res, out):
    L = res["best_layer"]
    m0 = np.array(res["models"]["M0"]["harmful_proj"])
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [1.1, 1]})
    ax = axes[0]
    layers = np.arange(len(m0))
    for t in FAM:
        studs = [np.array(v["harmful_proj"]) for k, v in res["models"].items() if k.startswith(f"S_{t}_s")]
        if studs:
            ax.plot(layers, np.mean(studs, 0) / m0, color=COL[t], lw=2.2, label=f"students of {NAME[t]}")
        if t in res["models"] and t != "T_none":
            ax.plot(layers, np.array(res["models"][t]["harmful_proj"]) / m0, color=COL[t], lw=1.5, ls="--", label=f"teacher: {NAME[t]}")
    ax.axhline(1, color=MUTED, lw=1)
    ax.axvline(L, color=MUTED, lw=1, ls=":")
    ax.set_ylim(0, None)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Refusal-direction activation\n(relative to M0)")
    ax.set_title("How strongly harmful prompts activate M0's refusal direction", fontsize=14)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="lower center")
    ax = axes[1]
    names, vals, cols, fills = [], [], [], []
    for t in FAM:
        if t in res["models"] and t != "T_none":
            names.append(f"{NAME[t]}\n(teacher)"); vals.append(res["models"][t]["harmful_proj"][L] / m0[L]); cols.append(COL[t]); fills.append(False)
        st = [v["harmful_proj"][L] / m0[L] for k, v in res["models"].items() if k.startswith(f"S_{t}_s")]
        if st:
            names.append(f"{NAME[t]}\n(students)"); vals.append(float(np.mean(st))); cols.append(COL[t]); fills.append(True)
    x = np.arange(len(names))
    for i in range(len(names)):
        ax.bar(x[i], vals[i], color=cols[i] if fills[i] else "white", edgecolor=cols[i], lw=2, width=0.7)
        ax.text(x[i], vals[i] + 0.02, f"{vals[i]:.2f}", ha="center", va="bottom", fontsize=11, color=INK)
    ax.axhline(1, color=MUTED, lw=1)
    ax.set_xticks(x, names, fontsize=10)
    ax.set_ylabel(f"Relative activation, layer {L}")
    ax.set_title(f"At the layer where M0 separates harmful from harmless best ({L})", fontsize=13)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig9_refusal_direction.png")
    plt.close(fig)


def fig_surgery(df, out):
    """Original students vs the same students with their teacher-aligned slice removed; and M0 + amplified slice."""
    nmax = df[(df.kind == "student") & df.teacher.isin(FAM)].n.max()
    mets = [("harmbench_harmful", "HarmBench harmful (%)", 100), ("prefill_asr_k5", "Prefill attack k=5 (%)", 100), ("refusal_margin", "Refusal margin (logits)", 1)]
    fig, axes = plt.subplots(len(mets), 1, figsize=(8, 3.3 * len(mets) + 0.8))
    groups = [t for t in FAM]
    for ax, (m, title, sc) in zip(axes, mets, strict=True):
        x = np.arange(len(groups))
        orig = [df[(df.kind == "student") & (df.teacher == t) & (df.n == nmax) & (df.metric == m)].value.mean() * sc for t in groups]
        rem = [df[(df.model.str.startswith(f"X_rm_{t}_s")) & (df.metric == m)].value.mean() * sc for t in groups]
        ax.bar(x - 0.2, orig, 0.38, color=[COL[t] for t in groups], label="students as trained")
        ax.bar(x + 0.2, rem, 0.38, color="white", edgecolor=[COL[t] for t in groups], lw=2, hatch="//", label="teacher-aligned slice removed")
        m0 = df[(df.model == "M0") & (df.metric == m)].value
        if not m0.empty:
            ax.axhline(m0.iloc[0] * sc, color=MUTED, lw=1, ls="--", label="M0")
        for xi, (o, r) in enumerate(zip(orig, rem, strict=True)):
            for dx, v in ((-0.2, o), (0.2, r)):
                if not np.isnan(v):
                    ax.text(xi + dx, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10, color=INK)
        ax.set_xticks(x, [f"students of\n{NAME[t]}" + ("\n(null: shallow dir.)" if t == "T_none" else "") for t in groups], fontsize=10)
        ax.set_title(title, fontsize=14)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig10_surgery_remove.png")
    plt.close(fig)

    fig, axes = plt.subplots(len(mets), 1, figsize=(8, 3.1 * len(mets) + 0.8))
    for ax, (m, title, sc) in zip(axes, mets, strict=True):
        m0 = df[(df.model == "M0") & (df.metric == m)].value
        for t in FAM[1:]:
            xs, ys = [0], [m0.iloc[0] * sc if not m0.empty else np.nan]
            for beta in (10, 50):
                v = df[(df.model == f"X_amp{beta}_{t}") & (df.metric == m)].value
                if not v.empty:
                    xs.append(beta); ys.append(v.iloc[0] * sc)
            ax.plot(xs, ys, marker="o", ms=8, color=COL[t], lw=2, label=f"{NAME[t]} slice")
            tv = df[(df.kind == "teacher") & (df.teacher == t) & (df.metric == m)].value
            if not tv.empty:
                ax.axhline(tv.iloc[0] * sc, color=COL[t], lw=1, ls=":")
        ax.set_xticks([0, 10, 50], ["M0", "×10", "×50"])
        ax.set_title(title + "  (dotted: the teacher itself)", fontsize=13)
    axes[0].legend(frameon=False, fontsize=10)
    axes[-1].set_xlabel("M0 + the students' teacher-aligned slice, amplified")
    fig.tight_layout()
    fig.savefig(out / "fig11_surgery_amplify.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/followups/metrics.csv")
    ap.add_argument("--refusal", default="results/followups/refusal_dir.json")
    ap.add_argument("--out", default="reports/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics)
    fig_benign(df, out)
    fig_surgery(df, out)
    if Path(args.refusal).exists():
        fig_refusal(json.load(open(args.refusal)), out)
    print(f"[followup_figs] wrote {out}/")


if __name__ == "__main__":
    main()
