#!/usr/bin/env python3
"""Fig. 4 — Specialist turn type heatmap.

Standalone: 9 arms × 9 QA types with hard-coded clinical-equivalence scores
(malformed-adjusted). No data files needed. Run:

    python fig4_heatmap.py --output-dir out --stem fig4
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


TYPES = [
    "Findings interpretation", "Treatment recommendation", "Evidence discussion",
    "Next action suggestion", "Uncertainty", "Eligibility assessment",
    "Agreement or support", "Clarification question", "Clinical trial suggestion",
]

# Two blocks: General purpose and Medical purpose. Each arm carries its
# per-QA-type mean clinical equivalence (1-5, malformed-adjusted) plus its
# overall mean over all questions and a flag for whether it read the slide
# images (mm) or captions only (cap).
GENERAL = [
    dict(name="DeepSeek-V4-Pro",   cond="cap",
         per=[3.66, 3.26, 3.23, 3.46, 3.38, 3.48, 3.42, 3.65, 3.14], overall=3.43),
    dict(name="DeepSeek-V4-Flash", cond="cap",
         per=[3.52, 3.15, 3.12, 3.39, 3.30, 3.37, 3.46, 3.61, 2.98], overall=3.33),
    dict(name="Llama 4 Scout",     cond="mm",
         per=[3.19, 2.75, 2.73, 3.05, 2.97, 3.02, 3.08, 3.32, 2.63], overall=2.97),
    dict(name="Gemma 4 31B",       cond="mm",
         per=[3.37, 2.89, 2.73, 3.13, 2.91, 3.12, 3.35, 3.50, 2.64], overall=3.08),
    dict(name="Ministral 3 14B",   cond="mm",
         per=[3.25, 2.84, 2.75, 3.03, 2.95, 3.02, 3.28, 3.38, 2.71], overall=3.02),
]
MEDICAL = [
    dict(name="MedGemma-27B",   cond="mm",
         per=[3.15, 2.70, 2.56, 2.88, 2.95, 2.80, 3.08, 3.24, 2.81], overall=2.88),
    dict(name="HuatuoGPT-3-32B", cond="cap",
         per=[2.80, 2.55, 2.41, 2.77, 2.68, 2.63, 2.79, 3.16, 2.49], overall=2.67),
    dict(name="Meditron-3-70B",  cond="cap",
         per=[3.20, 2.85, 2.75, 3.07, 3.05, 3.05, 3.25, 3.25, 2.81], overall=3.01),
    dict(name="MedReason-8B",    cond="cap",
         per=[2.42, 1.77, 1.95, 2.00, 2.33, 2.04, 1.88, 2.57, 2.01], overall=2.09),
]
BLOCKS = [("General purpose", GENERAL), ("Medical purpose", MEDICAL)]


# ── style ────────────────────────────────────────────────────────────────────
FS = 8.0
INK = "#1b1b1b"
CMAP = LinearSegmentedColormap.from_list(
    "blues", ["#e8f1f8", "#c1d4f2", "#87a8de", "#528ac7", "#2f6aa3", "#1f4e79"])
GAP = 0.4  # blank column between the CE cells and the "All questions" row


def _ink_on(rgb):
    r, g, b, _ = rgb
    return "white" if r*0.299 + g*0.587 + b*0.114 < 0.55 else INK


def resolve_font(family, font_dir):
    if font_dir:
        for p in sorted(Path(font_dir).glob("*")):
            if p.suffix.lower() in (".ttf", ".otf"):
                fm.fontManager.addfont(str(p))
    plt.rcParams["font.family"] = family


def draw(out_dir: Path, stem: str):
    arms = [a for _, group in BLOCKS for a in group]
    ncol = len(arms)
    nrow = len(TYPES)
    height_units = nrow + GAP + 1

    vmin = min(min(a["per"]) for a in arms)
    vmax = max(max(a["per"]) for a in arms)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Layout matches the paper's 8pt version: cells sized so "3.66" fits.
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.set_xlim(-3.0, ncol + 0.3)
    ax.set_ylim(height_units + 2.2, -1.4)
    ax.set_aspect("equal"); ax.axis("off")

    # Cells
    for c, arm in enumerate(arms):
        for r, t in enumerate(TYPES):
            v = arm["per"][r]
            col = CMAP(norm(v))
            ax.add_patch(Rectangle((c, r), 1, 1, fc=col, ec="white", lw=0.4, zorder=2))
            ax.text(c + 0.5, r + 0.5, f"{v:.2f}", fontsize=FS, ha="center",
                    va="center", color=_ink_on(col), zorder=3)
        col = CMAP(norm(arm["overall"]))
        ax.add_patch(Rectangle((c, nrow + GAP), 1, 1, fc=col, ec="white",
                               lw=0.4, zorder=2))
        ax.text(c + 0.5, nrow + GAP + 0.5, f"{arm['overall']:.2f}",
                fontsize=FS, ha="center", va="center", color=_ink_on(col),
                fontweight="bold", zorder=3)
        # arm name below the grid
        ax.text(c + 0.5, height_units + 0.30, arm["name"], fontsize=FS,
                rotation=45, ha="right", va="top", rotation_mode="anchor",
                clip_on=False)
        if arm["cond"] == "mm":
            ax.plot([c + 0.5], [height_units + 0.12], marker="o", ms=2.4,
                    mfc=INK, mec="none", zorder=4, clip_on=False)

    # Black block frames + block titles
    left = 0
    for title, group in BLOCKS:
        n = len(group)
        ax.add_patch(Rectangle((left, 0), n, nrow, fill=False, ec="black",
                               lw=1.15, zorder=5))
        ax.add_patch(Rectangle((left, nrow + GAP), n, 1, fill=False,
                               ec="black", lw=1.15, zorder=5))
        ax.text(left + n/2, -0.30, title, fontsize=FS, color=INK,
                ha="center", va="bottom", fontweight="bold", clip_on=False)
        left += n

    # Row labels
    for r, t in enumerate(TYPES):
        ax.text(-0.22, r + 0.5, t, fontsize=FS, ha="right", va="center")
    ax.text(-0.22, nrow + GAP + 0.5, "All questions", fontsize=FS,
            ha="right", va="center", fontweight="bold")

    # Colour bar
    cax = fig.add_axes([0.905, 0.28, 0.015, 0.5])
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cax.tick_params(labelsize=FS, length=2)
    cax.set_ylabel("Clinical equivalence  (1–5)", fontsize=FS, labelpad=3)

    fig.suptitle("Specialist turn type", fontsize=FS, fontweight="bold",
                 x=0.5, y=0.97)

    out_dir.mkdir(parents=True, exist_ok=True)
    for suf in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"{stem}.{suf}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {stem}.{{pdf,png,svg}} in {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("."))
    ap.add_argument("--stem", default="fig4_heatmap")
    ap.add_argument("--font-family", default="DejaVu Sans")
    ap.add_argument("--font-dir", type=Path, default=None)
    a = ap.parse_args()
    resolve_font(a.font_family, a.font_dir)
    draw(a.output_dir, a.stem)


if __name__ == "__main__":
    main()
