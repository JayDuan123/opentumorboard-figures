#!/usr/bin/env python3
"""Fig. 6 (combined) — bar chart of Qwen2.5-VL-3B / +SFT / +SFT+RL (panel a)
plus the RL validation-reward curve peaking at step 400 (panel b).

Panel a hard-codes the same numbers as fig6_training.py.  Panel b reads the
v25 training curve from --curves-csv and shows the total validation reward
truncated at the peak step, with a dashed line marking the maximum.

    python fig6_combined.py --output-dir out --stem fig6 \\
        --font-family Helvetica --font-dir /workspace/yd68 \\
        --curves-csv /path/to/v25_training_curves_final.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, patheffects as pe


# ── panel-a data ─────────────────────────────────────────────────────────────
ARMS = [
    dict(label="Qwen2.5-VL-3B", color="#c5d5e8", ca=1.58,
         therapy=1.06, surgery=1.23, next_action=1.34, clinical_trial=0.98),
    dict(label="+ SFT",         color="#4a94d0", ca=1.67,
         therapy=1.21, surgery=1.99, next_action=2.03, clinical_trial=1.22),
    dict(label="+ SFT + RL",    color="#1f4e79", ca=1.86,
         therapy=1.39, surgery=2.27, next_action=2.07, clinical_trial=1.13),
]
GROUPS = [("Conclusion\nalignment", "ca"),
          ("Therapy",               "therapy"),
          ("Surgery",               "surgery"),
          ("Next\naction",          "next_action"),
          ("Clinical\ntrial",       "clinical_trial")]

# ── style ────────────────────────────────────────────────────────────────────
FS = 8.0
INK = "#1b1b1b"
VAL_C = "#1f4e79"      # matches the RL bar colour in panel a


def resolve_font(family, font_dir):
    if font_dir:
        for p in sorted(Path(font_dir).glob("*")):
            if p.suffix.lower() in (".ttf", ".otf", ".ttc"):
                fm.fontManager.addfont(str(p))
    plt.rcParams["font.family"] = family


def draw_bars(ax):
    bar_w = 0.24
    n_grp = len(GROUPS)
    xs_group = list(range(n_grp))
    for i, arm in enumerate(ARMS):
        xs = [x + (i - 1) * bar_w for x in xs_group]
        ys = [arm[key] for _, key in GROUPS]
        ax.bar(xs, ys, width=bar_w, color=arm["color"], zorder=3,
               label=arm["label"])
        # Value labels omitted — bar heights against the y-axis carry the
        # numbers, and Table~\ref{tab:train-full} has the exact values.

    ax.set_xticks(xs_group)
    ax.set_xticklabels([name for name, _ in GROUPS], fontsize=FS)
    ax.set_ylim(0, 2.6); ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_ylabel("Mean score  (0–5)", fontsize=FS, labelpad=3)
    ax.tick_params(labelsize=FS, length=2, color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6); ax.spines[s].set_color(INK)
    ax.grid(False); ax.set_facecolor("white"); ax.set_axisbelow(True)

    # Horizontal legend above the plot — a vertical legend at upper right
    # was overlapping the tallest Surgery / Next-action bars.
    leg = ax.legend(fontsize=FS, loc="lower center", frameon=False,
                    ncol=3, handlelength=1.0, handletextpad=0.5,
                    columnspacing=1.4, borderpad=0.2,
                    bbox_to_anchor=(0.5, 1.02))
    for t in leg.get_texts(): t.set_color(INK)


def draw_val_curve(ax, csv_path):
    df = pd.read_csv(csv_path)
    X_MAX = 400
    va = df[["step", "val_reward"]].dropna()
    va = va[va["step"] <= X_MAX]

    ax.plot(va["step"], va["val_reward"], color=VAL_C, lw=1.1,
            marker="o", ms=3.0, mfc=VAL_C, mec=VAL_C, zorder=3)

    ymax = va["val_reward"].max()
    ax.axhline(ymax, color=INK, lw=0.7, ls="--", zorder=2)
    ax.text(X_MAX, ymax + 0.008, f"{ymax:.3f}",
            ha="right", va="bottom", color=INK, fontsize=FS)

    ax.set_xlabel("Training step", fontsize=FS)
    ax.set_ylabel("Validation reward", fontsize=FS, labelpad=3)
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, ymax * 1.10)
    ax.tick_params(labelsize=FS, length=2, color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6); ax.spines[s].set_color(INK)
    ax.grid(False); ax.set_facecolor("white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("."))
    ap.add_argument("--stem", default="fig6")
    ap.add_argument("--font-family", default="DejaVu Sans")
    ap.add_argument("--font-dir", type=Path, default=None)
    ap.add_argument("--curves-csv", type=Path, required=True,
                    help="v25 training curves CSV with step, val_reward")
    a = ap.parse_args()
    resolve_font(a.font_family, a.font_dir)

    # figsize sized so the tight-bboxed native PDF ends up ~606 pt wide,
    # matching fig5.pdf; when the paper renders both at width=\linewidth
    # the 8pt Helvetica source prints at the same ~6.2pt on paper.
    fig = plt.figure(figsize=(9.6, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.4, 2.6], wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    draw_bars(ax_a)
    draw_val_curve(ax_b, a.curves_csv)

    # Panel labels A, B — Helvetica Bold 12pt (real Bold; the .ttc's Bold
    # face has been extracted to Helvetica-Bold.ttf so resolve_font()
    # registers it and fontweight="bold" no longer silently falls back).
    for ax, tag in [(ax_a, "A"), (ax_b, "B")]:
        ax.text(-0.14, 1.02, tag, transform=ax.transAxes,
                fontsize=10, color=INK, ha="left", va="bottom",
                fontweight="bold")

    fig.tight_layout()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for suf in ("pdf", "png", "svg"):
        fig.savefig(a.output_dir / f"{a.stem}.{suf}", dpi=300,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {a.stem}.{{pdf,png,svg}} in {a.output_dir}")


if __name__ == "__main__":
    main()
