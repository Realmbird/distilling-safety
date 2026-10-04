"""Figures for the write-up (portrait, <=8in wide, large fonts, dpi 200 — readable in a Google Doc).

  fig_teachers.png   teacher gate: prefill ASR vs k, plus latent-attack ASR, for M0 / T_deep / T_adv
  fig_transfer.png   student delta vs T_none-student (matched seed, N) against N, one panel per metric

    python -m distill_safety.plots --metrics runs/metrics.csv --out figs
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# categorical slots 1-3 in fixed order (validated default palette); text stays in ink colors
COLORS = {"T_none": "#52514e", "T_deep": "#2a78d6", "T_adv": "#eb6834"}
LABELS = {"T_none": "M0 (no extra safety)", "T_deep": "T_deep (Method 1: recovery)", "T_adv": "T_adv (Method 2: LAT)"}
MARKERS = {"T_none": "s", "T_deep": "o", "T_adv": "^"}

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.8,
        "lines.linewidth": 2,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

TRANSFER_METRICS = [
    ("prefill_asr_k20", "Prefill-attack ASR (k=20)", "lower = deeper"),
    ("latent_nll_attacked", "NLL of harmful target under latent attack", "higher = more robust"),
    ("prefill_nll_k20", "NLL of harmful continuation after 20-token prefix", "higher = deeper"),
    ("refusal_margin", "Refusal margin  log P(refuse) - log P(comply)", "higher = more refusing"),
    ("harmbench_refusal", "HarmBench refusal rate", ""),
]


def fig_teachers(df: pd.DataFrame, out: Path):
    t = df[df.kind == "teacher"]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 9.5))
    ax = axes[0]
    for teacher in ["T_none", "T_deep", "T_adv"]:
        s = t[(t.teacher == teacher) & t.metric.str.startswith("prefill_asr_k")]
        if s.empty:
            continue
        ks = s.metric.str.replace("prefill_asr_k", "").astype(int)
        order = np.argsort(ks.values)
        ax.plot(ks.values[order], s.value.values[order] * 100, marker=MARKERS[teacher], ms=8, color=COLORS[teacher], label=LABELS[teacher])
    ax.set_xlabel("Harmful tokens prefilled (k)")
    ax.set_ylabel("Attack success (%)")
    ax.set_title("Prefill attack (Qi et al. depth test)")
    ax.legend(frameon=False)

    ax = axes[1]
    teachers = [x for x in ["T_none", "T_deep", "T_adv"] if not t[(t.teacher == x) & (t.metric == "latent_asr")].empty]
    vals = [float(t[(t.teacher == x) & (t.metric == "latent_asr")].value.iloc[0]) * 100 for x in teachers]
    ax.bar(range(len(teachers)), vals, color=[COLORS[x] for x in teachers], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=13, color="#0b0b0b")
    ax.set_xticks(range(len(teachers)), [LABELS[x].split(" (")[0] for x in teachers])
    ax.set_ylabel("Attack success (%)")
    ax.set_title("Latent (residual-stream PGD) attack")
    fig.tight_layout()
    fig.savefig(out / "fig_teachers.png")
    plt.close(fig)


def student_deltas(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    s = df[(df.kind == "student") & (df.metric == metric)]
    ctrl = s[s.teacher == "T_none"].set_index(["seed", "n"]).value
    rows = []
    for (teacher, seed, n), v in s[s.teacher != "T_none"].set_index(["teacher", "seed", "n"]).value.items():
        if (seed, n) in ctrl.index:
            rows.append({"teacher": teacher, "seed": seed, "n": n, "delta": v - ctrl.loc[(seed, n)]})
    return pd.DataFrame(rows)


def fig_transfer(df: pd.DataFrame, out: Path, metrics=TRANSFER_METRICS):
    present = [m for m in metrics if not df[(df.kind == "student") & (df.metric == m[0])].empty]
    if not present:
        print("[plots] no student metrics yet")
        return
    fig, axes = plt.subplots(len(present), 1, figsize=(7.5, 3.6 * len(present)), squeeze=False)
    for ax, (metric, title, note) in zip(axes[:, 0], present, strict=True):
        d = student_deltas(df, metric)
        noise = df[(df.kind == "student") & (df.teacher == "T_none") & (df.metric == metric)].groupby("n").value.std()
        if not noise.empty and noise.notna().any():
            ax.fill_between(noise.index, -noise.values, noise.values, color="#f0efec", label="T_none seed s.d.")
        ax.axhline(0, color="#52514e", lw=1)
        for teacher in ["T_deep", "T_adv"]:
            g = d[d.teacher == teacher].groupby("n").delta
            if g.ngroups == 0:
                continue
            m, sd, cnt = g.mean(), g.std().fillna(0), g.count()
            ci = 1.96 * sd / np.sqrt(cnt.clip(lower=1))
            ax.errorbar(m.index, m.values, yerr=ci.values, marker=MARKERS[teacher], ms=8, capsize=4, color=COLORS[teacher], label=LABELS[teacher])
        ax.set_xscale("log")
        ax.set_title(title + (f"\n({note})" if note else ""), fontsize=15)
        ax.set_ylabel("Δ vs T_none student")
        ax.set_xlabel("Distilled samples seen (N)")
    axes[0, 0].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out / "fig_transfer.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="runs/metrics.csv")
    ap.add_argument("--out", default="figs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics)
    fig_teachers(df, out)
    fig_transfer(df, out)
    print(f"[plots] wrote {out}/")


if __name__ == "__main__":
    main()
