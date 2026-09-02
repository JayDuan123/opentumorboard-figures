#!/usr/bin/env python3
"""Figure 2b, four panels, drawn with pyCirclize instead of raw matplotlib.

A COMPARISON, NOT THE PUBLISHED FIGURE. figure2b_sunburst.py is what the paper uses.
This exists because the library question is worth answering once, with a running
panel rather than a reading of the docs, and worth keeping so the answer can be
rechecked when the taxonomy grows.

WHAT THE LIBRARY BUYS. Sector and Track own the geometry, so the wedges are placed by
value rather than by angles computed here, and this file is roughly a sixth the length
of the hand-rolled version. `Track.annotate()` replaces the leader-line placement:
nine of the seventeen sites are under 3% and their labels land at nearly the same
angle, and its automatic spreading handles them without the sorting, minimum-gap and
overflow-recentring the hand-rolled `place_outside` needed. `orientation="vertical"`
handles the left-half flip.

WHAT IT DOES NOT. `space=0` has to be set explicitly - the default leaves gaps between
sectors, and a gap in a taxonomy ring reads as missing data. Single-leaf groups still
have to be suppressed by hand or the group and its only leaf print the same word twice
at the same angle. And there is no radial label fitting: the published version wraps
and shrinks a label to the band it sits in, which is why "Hepatobiliary / pancreas"
sets on two lines there and is squeezed onto one here.

Requires `pip install pycirclize` (MIT, matplotlib-based). It is not needed for any
published figure, which is why it is not in requirements.txt.

usage:
  python -m scripts.paper.figure2b_circos \
      --manifest-dir data/model_evaluation/manifests --output-dir fig2b_out
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.paper import cancer_sites  # noqa: E402
from scripts.paper.figure2b_sunburst import (  # noqa: E402
    ROLE_GROUP, SITE_SYSTEM, TASK1, TEST, TRAINVAL, load, nest, pretty, refuse,
)

PAL = ["#8da0cb", "#a6bddb", "#9ebcda", "#b3cde3", "#c7d4e8",
       "#bcbddc", "#ccc4de", "#a8ddb5", "#d4b9da", "#cccccc"]
FS_TITLE, FS_GROUP, FS_LEAF, FS_ANN = 8.0, 6.2, 5.2, 4.8


def _circos():
    try:
        from pycirclize import Circos
    except ImportError:
        refuse("pycirclize is not installed. This comparison panel needs it: "
               "pip install pycirclize. No published figure does.")
    return Circos


def nested_panel(ax, nested, total, title, lab=str):
    Circos = _circos()
    # space=0: the default leaves a gap between sectors, and a gap in a taxonomy
    # ring reads as a category with no members rather than as a divider.
    circos = Circos({g.replace("\n", " "): t for g, t, _ in nested}, space=0)
    for i, (g, tot, leaves) in enumerate(nested):
        sec = circos.get_sector(g.replace("\n", " "))
        sec.add_track((46, 68)).axis(fc=PAL[i % len(PAL)], ec="white", lw=1.2)
        # A group with one leaf is named once: the library will happily print the
        # group and its only leaf at the same angle, one over the other.
        if len(leaves) > 1 and 360 * tot / total >= 14:
            sec.text(g.replace("\n", " "), r=57, size=FS_GROUP, weight="bold",
                     orientation="vertical")
        out = sec.add_track((69, 92))
        out.axis(ec="white", lw=0)
        x = 0.0
        for j, (name, n) in enumerate(leaves):
            out.rect(x, x + n, fc=PAL[i % len(PAL)], ec="white", lw=0.9,
                     alpha=0.42 + 0.16 * (j % 3))
            if 360 * n / total >= 11:
                out.text(lab(name), x=x + n / 2, size=FS_LEAF, orientation="vertical")
            else:
                # shorten=None: the default truncates a label at 20 characters
                out.annotate(x + n / 2, f"{lab(name)} {n}", label_size=FS_ANN,
                             shorten=None)
            x += n
    circos.plotfig(ax=ax)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=16)


def flat_panel(ax, counter, total, title):
    Circos = _circos()
    circos = Circos({"all": total}, space=0)
    track = circos.get_sector("all").add_track((52, 92))
    track.axis(ec="white", lw=0)
    x = 0.0
    for i, (k, n) in enumerate(counter.most_common()):
        track.rect(x, x + n, fc=PAL[i % len(PAL)], ec="white", lw=1.0, alpha=0.85)
        if 360 * n / total >= 13:
            track.text(f"{pretty(k)}\n{100 * n / total:.0f}%", x=x + n / 2,
                       size=FS_LEAF, orientation="vertical")
        else:
            track.annotate(x + n / 2, f"{pretty(k)} {100 * n / total:.1f}%",
                           label_size=FS_ANN, shorten=None)
        x += n
    circos.plotfig(ax=ax)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--stem", default="fig2b_circos")
    a = ap.parse_args()

    md = a.manifest_dir
    qa = load(md / TRAINVAL) + load(md / TEST)
    cases = load(md / TASK1)
    sites, _ = cancer_sites.site_counts(
        [({"video_uid": r["video_uid"], "case_id": r["case_id"]}, r["case_summary"])
         for r in cases])
    roles = collections.Counter(r["target_specialist_role"] for r in qa)
    qtypes = collections.Counter(r["qa_type"] for r in qa)
    turns = [len(r["reference_discussion"]) for r in cases]

    fig = plt.figure(figsize=(9.2, 9.6))
    a1 = fig.add_subplot(2, 2, 1, polar=True)   # plotfig(ax=...) needs a PolarAxes
    a2 = fig.add_subplot(2, 2, 2, polar=True)
    a3 = fig.add_subplot(2, 2, 3, polar=True)
    a4 = fig.add_subplot(2, 2, 4)               # a histogram is not circular

    nested_panel(a1, nest(sites, SITE_SYSTEM), len(cases),
                 f"Primary cancer site  ·  {len(cases)} cases")
    nested_panel(a2, nest(roles, ROLE_GROUP), len(qa),
                 f"Target specialist  ·  {len(qa):,} questions",
                 lab=lambda s: s[:1].upper() + s[1:])
    flat_panel(a3, qtypes, len(qa), f"Question type  ·  {len(qtypes)} types")

    counts = collections.Counter(min(t // 10 * 10, 90) for t in turns)
    edges = list(range(0, 91, 10))
    a4.bar(edges, [counts.get(e, 0) for e in edges], width=8.6, align="edge",
           color="#b3cde3", edgecolor="#6a8fc0", lw=0.7)
    a4.set_title(f"Discussion length  ·  {sum(turns):,} utterances",
                 fontsize=FS_TITLE, fontweight="bold", pad=16)
    a4.set_xlabel("utterances per case", fontsize=6)
    a4.tick_params(labelsize=6)
    for s in ("top", "right"):
        a4.spines[s].set_visible(False)

    fig.suptitle("Benchmark statistics  —  drawn with pyCirclize", fontsize=12,
                 fontweight="bold")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("pdf", "png"):
        p = a.output_dir / f"{a.stem}.{suffix}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        written.append(str(p))
    plt.close(fig)
    print(json.dumps({"outputs": written, "cases": len(cases), "questions": len(qa),
                      "sites": len(sites), "roles": len(roles)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
