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
# categorical slots in fixed order; each method's matched shallow baseline gets its own colour, dashed
COLORS = {
    "T_none": "#52514e",
    "T_shallow": "#1baf7a", "T_deep": "#2a78d6", "T_adv": "#eb6834",
    "T_shallow_cal": "#1baf7a", "T_deep_cal": "#2a78d6", "T_shallow_v2": "#eda100", "T_adv_v2": "#eb6834",
}
LABELS = {
    "T_none": "M0 (no extra safety)",
    "T_shallow": "T_shallow (plain refusal SFT)",
    "T_deep": "T_deep (Method 1: recovery)",
    "T_adv": "T_adv (Method 2: LAT)",
    "T_shallow_cal": "T_shallow, calibrated (baseline for M1)",
    "T_deep_cal": "T_deep, calibrated (Method 1: recovery)",
    "T_shallow_v2": "T_shallow, v2 (baseline for M2)",
    "T_adv_v2": "T_adv, v2 (Method 2: LAT)",
}
MARKERS = {"T_none": "s", "T_shallow": "D", "T_deep": "o", "T_adv": "^", "T_shallow_cal": "D", "T_deep_cal": "o", "T_shallow_v2": "D", "T_adv_v2": "^"}
LINESTYLES = {k: "--" if "shallow" in k else "-" for k in COLORS}
# plotted only if present in the data
SAFETY_TEACHERS = ["T_shallow", "T_deep", "T_adv", "T_shallow_cal", "T_deep_cal", "T_shallow_v2", "T_adv_v2"]

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


def _is_rate(metric: str) -> bool:
    return metric.startswith(("prefill_asr", "harmbench_", "latent_asr", "xstest_", "gsm8k_"))


def fig_teachers(df: pd.DataFrame, out: Path):
    t = df[df.kind == "teacher"]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 12.5), gridspec_kw={"height_ratios": [1.15, 1]})
    ax = axes[0]
    for teacher in ["T_none", *SAFETY_TEACHERS]:
        s = t[(t.teacher == teacher) & t.metric.str.startswith("prefill_asr_k")]
        if s.empty:
            continue
        ks = s.metric.str.replace("prefill_asr_k", "").astype(int)
        order = np.argsort(ks.values)
        ax.plot(ks.values[order], s.value.values[order] * 100, marker=MARKERS[teacher], ms=8, color=COLORS[teacher], ls=LINESTYLES[teacher], label=LABELS[teacher])
    ax.set_xlabel("Harmful tokens prefilled (k)")
    ax.set_ylabel("Attack success (%)")
    ax.set_title("Prefill attack (depth test)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=12)
    # a collapsed continuation is "not harmful" but not a recovery either: say so on the figure
    notes = []
    for teacher in ["T_none", *SAFETY_TEACHERS]:
        dg = t[(t.teacher == teacher) & (t.metric == "degenerate_prefill")]
        if not dg.empty and float(dg.value.iloc[0]) > 0.05:
            notes.append(f"{teacher}: {float(dg.value.iloc[0]):.0%} of prefilled continuations\nare degenerate text (judged not harmful)")
    if notes:
        ax.set_ylim(-4, 100 + 12 * len(notes))  # headroom so the note never sits on a line
        ax.set_yticks(range(0, 101, 20))
        ax.text(0.02, 0.985, "\n".join(notes), transform=ax.transAxes, va="top", fontsize=11, color="#52514e")

    ax = axes[1]
    teachers = [x for x in ["T_none", *SAFETY_TEACHERS] if not t[(t.teacher == x) & (t.metric == "latent_asr")].empty]
    vals = [float(t[(t.teacher == x) & (t.metric == "latent_asr")].value.iloc[0]) * 100 for x in teachers]
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.bar(range(len(teachers)), vals, color=[COLORS[x] for x in teachers], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=13, color="#0b0b0b")
    ax.set_ylim(0, max(vals + [10]) * 1.18)
    ax.set_xticks(range(len(teachers)), [x.replace("T_", "") for x in teachers], rotation=20)
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
    ns = sorted(df[df.kind == "student"].n.unique())
    for ax, (metric, title, note) in zip(axes[:, 0], present, strict=True):
        scale = 100.0 if _is_rate(metric) else 1.0  # rates shown in percentage points
        d = student_deltas(df, metric)
        noise = df[(df.kind == "student") & (df.teacher == "T_none") & (df.metric == metric)].groupby("n").value.std()
        if not noise.empty and noise.notna().any():
            ax.fill_between(noise.index, -noise.values * scale, noise.values * scale, color="#f0efec", label="T_none students: ±1 s.d. across seeds")
        ax.axhline(0, color="#52514e", lw=1)
        for j, teacher in enumerate(SAFETY_TEACHERS):
            g = d[d.teacher == teacher].groupby("n").delta
            if g.ngroups == 0:
                continue
            # with 2-3 seeds a t-interval is meaningless: show the mean and the full seed range
            m, lo, hi = g.mean() * scale, g.min() * scale, g.max() * scale
            x = m.index.values * 1.04 ** (j - 1.5)  # nudge apart so bars don't overlap
            ax.errorbar(x, m.values, yerr=[m.values - lo.values, hi.values - m.values], marker=MARKERS[teacher], ms=8, capsize=4, color=COLORS[teacher], ls=LINESTYLES[teacher], label=LABELS[teacher])
        ax.set_xscale("log")
        ax.set_xticks(ns, [f"{n / 1000:g}k" for n in ns])
        ax.minorticks_off()
        ax.set_title(title + (f"\n({note})" if note else ""), fontsize=15)
        ax.set_ylabel("Δ vs T_none student" + (" (pts)" if scale == 100 else ""))
        ax.set_xlabel("Distilled samples seen (N)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    fig.text(0.5, -0.005, "Points: mean over seeds; error bars: range across seeds.", ha="center", va="top", fontsize=12, color="#52514e")
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
