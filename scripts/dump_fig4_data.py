#!/usr/bin/env python3
"""Dump fig4's hard-coded 9x9 clinical-equivalence grid as a CSV.

Reads the GENERAL / MEDICAL constants straight out of
standalone/fig4_heatmap.py so the CSV can never drift from the plot,
and writes:

    figures_paper/fig4_data.csv

Layout: one row per question type (plus a final 'All questions' row of
per-arm overall means); one column per arm.  Column headers carry an
extra ' [cap]' or ' [mm]' suffix so a reader knows which arms saw slide
images and which read captions only.
"""
from __future__ import annotations
import csv
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "standalone"))
from fig4_heatmap import TYPES, GENERAL, MEDICAL, BLOCKS  # noqa: E402


def main() -> int:
    arms = [a for _, group in BLOCKS for a in group]

    out = HERE.parent / "figures_paper" / "fig4_data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    headers = ["question_type", "block"] + [
        f"{arm['name']} [{arm['cond']}]" for arm in arms
    ]

    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)

        # one row per QA type, values in the same left-to-right order as the plot
        for r, qtype in enumerate(TYPES):
            # every arm falls under exactly one block
            block_by_arm = []
            for block_title, group in BLOCKS:
                block_by_arm += [block_title] * len(group)
            # QA-type row: block column empty (it's the arm attribute, not the row's)
            row = [qtype, ""]
            for arm in arms:
                row.append(f"{arm['per'][r]:.2f}")
            w.writerow(row)

        # trailing 'All questions' row = each arm's overall
        row = ["All questions", ""]
        for arm in arms:
            row.append(f"{arm['overall']:.2f}")
        w.writerow(row)

        # header row describing block membership per column
        row = ["_arm_block", ""]
        for arm in arms:
            block = next(t for t, g in BLOCKS if arm in g)
            row.append(block)
        w.writerow(row)

    print(f"wrote {out} ({len(TYPES)} QA types + All questions + 1 block row"
          f"; {len(arms)} arm columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
