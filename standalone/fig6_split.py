#!/usr/bin/env python3
"""Emit fig6's two panels as separate files.

Reads the same draw_bars / draw_val_curve routines used by
fig6_combined.py so the panels stay identical to the combined figure in
style, colours, fontsize and geometry.  Writes:

    fig6a.{pdf,png,svg}   — training bar chart (Qwen2.5-VL-3B / +SFT / +SFT+RL)
    fig6b.{pdf,png,svg}   — RL validation reward curve, dashed peak at 0.391

usage:
    python fig6_split.py --output-dir out \\
        --font-family Helvetica --font-dir /workspace/yd68 \\
        --curves-csv /path/to/v25_training_curves_final.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig6_combined import draw_bars, draw_val_curve, resolve_font  # noqa: E402


def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suf in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"{stem}.{suf}", dpi=300, facecolor="white",
                    bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("."))
    ap.add_argument("--font-family", default="DejaVu Sans")
    ap.add_argument("--font-dir", type=Path, default=None)
    ap.add_argument("--curves-csv", type=Path, required=True)
    a = ap.parse_args()
    resolve_font(a.font_family, a.font_dir)

    # panel a — bars
    fig_a, ax_a = plt.subplots(figsize=(4.4, 2.7))
    draw_bars(ax_a)
    fig_a.tight_layout()
    _save(fig_a, a.output_dir, "fig6a")

    # panel b — val reward curve
    fig_b, ax_b = plt.subplots(figsize=(2.6, 2.7))
    draw_val_curve(ax_b, a.curves_csv)
    fig_b.tight_layout()
    _save(fig_b, a.output_dir, "fig6b")

    print(f"wrote fig6a.{{pdf,png,svg}} and fig6b.{{pdf,png,svg}} in {a.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
