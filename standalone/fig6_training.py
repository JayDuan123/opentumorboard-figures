#!/usr/bin/env python3
"""Fig. 6 — training on OpenTumorBoard.

Standalone: base / +SFT / +SFT+RL Qwen2.5-VL-3B on Task 1 conclusion alignment
plus the four in-scope rubric decisions (Therapy, Surgery, Next action,
Clinical trial). All numbers hard-coded. Run:

    python fig6_training.py --output-dir out --stem fig6
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


# ── data ─────────────────────────────────────────────────────────────────────
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
          ("Next action",           "next_action"),
          ("Clinical trial",        "clinical_trial")]

# ── style ────────────────────────────────────────────────────────────────────
FS = 7.0
INK = "#1b1b1b"


def resolve_font(family, font_dir):
    if font_dir:
        for p in sorted(Path(font_dir).glob("*")):
            if p.suffix.lower() in (".ttf", ".otf"):
                fm.fontManager.addfont(str(p))
    plt.rcParams["font.family"] = family


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("."))
    ap.add_argument("--stem", default="fig6_training")
    ap.add_argument("--font-family", default="DejaVu Sans")
    ap.add_argument("--font-dir", type=Path, default=None)
    a = ap.parse_args()
    resolve_font(a.font_family, a.font_dir)

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    bar_w = 0.24
    n_grp = len(GROUPS)
    xs_group = list(range(n_grp))
    for i, arm in enumerate(ARMS):
        xs = [x + (i - 1) * bar_w for x in xs_group]
        ys = [arm[key] for _, key in GROUPS]
        ax.bar(xs, ys, width=bar_w, color=arm["color"], zorder=3,
               label=arm["label"])
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.06, f"{y:.2f}", ha="center", va="bottom",
                    fontsize=FS - 0.6, color="#5b6572", zorder=4)

    ax.set_xticks(xs_group)
    ax.set_xticklabels([name for name, _ in GROUPS], fontsize=FS)
    ax.set_ylim(0, 2.9); ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_ylabel("Mean score  (0–5)", fontsize=FS, labelpad=3)
    ax.tick_params(labelsize=FS, length=2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_linewidth(0.6)
    ax.grid(False); ax.set_facecolor("white"); ax.set_axisbelow(True)

    leg = ax.legend(fontsize=FS, loc="upper right", frameon=False,
                    handlelength=1.0, handletextpad=0.5,
                    labelspacing=0.35, borderpad=0.2)
    for t in leg.get_texts(): t.set_color(INK)

    fig.tight_layout()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for suf in ("pdf", "png", "svg"):
        fig.savefig(a.output_dir / f"{a.stem}.{suf}", dpi=300,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {a.stem}.{{pdf,png,svg}} in {a.output_dir}")


if __name__ == "__main__":
    main()
