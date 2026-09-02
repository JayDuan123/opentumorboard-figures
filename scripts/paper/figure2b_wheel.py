#!/usr/bin/env python3
"""Figure 2b - the whole benchmark as one wheel.

Four quadrants, each a facet of the corpus, each given an equal quarter of the
circle. Inside a quadrant the outer ring IS proportional, to that facet's own total.
The quarters are not proportional to each other and must not be read that way: 611
cases and 16,215 questions are different units, and a wheel that sized them against
one another would be claiming a ratio that does not exist.

  Questions   -> 9 question types
  Cases       -> 17 primary cancer sites
  Specialists -> 10 target roles
  Corpus      -> videos, hours, slides (no leaf ring; nothing to divide)

Counts come from the same manifests, read by the same code, as figure2b_statistics.py
and figure2b_sunburst.py - the site labels through cancer_sites.site_counts, the rest
through the same Counter over the same records - so the three versions of Figure 2b
report identical numbers by construction rather than by coincidence. Verified against
the bar version's provenance: all 17 sites, 10 roles, 9 question types and every scale
number match.

usage:
  python -m scripts.paper.figure2b_wheel \
      --manifest-dir .../model_evaluation/manifests --output-dir .../figures/fig2b
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

from scripts.paper import cancer_sites  # noqa: E402
from scripts.paper.figure2b_sunburst import (  # noqa: E402
    TRAINVAL, TEST, TASK1, fit_radial, load, pretty, refuse, resolve_font, sha256,
)

FIG_IN = 8.4
DPI = 600

FS_TITLE = 13.0
FS_HEAD = 9.5          # ring 1, the headline number
FS_SUB = 6.6           # ring 2, the descriptor
FS_LEAF = 5.6          # ring 3
FS_NOTE = 6.2

INK = "#1b1b1b"
MUTED = "#5f5f5f"

R0, R1, R2, R3 = 0.30, 0.56, 0.76, 1.06

# One hue per quadrant, given as (ring1, ring2, ring3 base). Inner is the saturated
# end so the eye lands on the headline number first.
QUADS = [
    ("questions", 90.0, "#2aa3d4", "#54bde8", "#7cd0f0"),
    ("cases", 0.0, "#2eaa4e", "#48c268", "#6ed889"),
    ("corpus", 270.0, "#e07a42", "#ef9a67", "#f6bd97"),
    ("specialists", 180.0, "#2b7fa8", "#4198c2", "#67b3d8"),
]


def shade(hexcolor: str, factor: float) -> tuple:
    c = matplotlib.colors.to_rgb(hexcolor)
    return tuple(1 - (1 - v) * factor for v in c)


def gather(md: Path) -> dict:
    qa = load(md / TRAINVAL) + load(md / TEST)
    cases = load(md / TASK1)
    if len({r["qa_id"] for r in qa}) != len(qa):
        refuse("duplicate qa_id across the QA manifests")

    sites, _ = cancer_sites.site_counts(
        [({"video_uid": r["video_uid"], "case_id": r["case_id"]}, r["case_summary"])
         for r in cases])
    roles = collections.Counter(r["target_specialist_role"] for r in qa)
    qa_types = collections.Counter(r["qa_type"] for r in qa)
    turns = [len(r["reference_discussion"]) for r in cases]
    slides = [str(r.get("slides") or "").count("<image>") for r in cases]
    dur = [r["case_end_sec"] - r["case_start_sec"] for r in cases]
    import statistics
    return {
        "qa_types": qa_types, "sites": sites, "roles": roles,
        "scale": {
            "videos": len({r["video_uid"] for r in cases}), "cases": len(cases),
            "questions": len(qa), "slides": sum(slides),
            "slides_median": statistics.median(slides), "slides_max": max(slides),
            "hours": sum(dur) / 3600.0, "utterances": sum(turns),
            "turns_median": statistics.median(turns),
        },
        "sources": {n: {"path": str((md / n).resolve()), "sha256": sha256(md / n)}
                    for n in (TRAINVAL, TEST, TASK1)},
    }


def ring_text(ax, mid, r_lo, r_hi, text, fs, weight="normal", color=INK,
              tangential=False):
    """Ring label. Radial by default and shrunk to the band it sits in.

    `tangential` runs the text along the arc instead. A radial label is bounded by
    the band, which is 0.30 wide here, so a 60-character summary line has to be set
    at 1.3 pt to fit - the arc is four times longer and is where such a line belongs.
    """
    band = r_hi - r_lo
    if tangential:
        rot = mid - 90
        if 90 < rot % 360 < 270:
            rot += 180
        rr = (r_lo + r_hi) / 2
        ax.text(rr * math.cos(math.radians(mid)), rr * math.sin(math.radians(mid)),
                text, rotation=rot, rotation_mode="anchor", ha="center", va="center",
                fontsize=fs, fontweight=weight, color=color, linespacing=1.25, zorder=6)
        return
    txt, fitted = fit_radial(text, band, fs)
    rot = mid + 180 if 90 < mid % 360 < 270 else mid
    rr = (r_lo + r_hi) / 2
    ax.text(rr * math.cos(math.radians(mid)), rr * math.sin(math.radians(mid)), txt,
            rotation=rot, rotation_mode="anchor", ha="center", va="center",
            fontsize=fitted, fontweight=weight, color=color, linespacing=0.94, zorder=6)


def quadrant(ax, start, c1, c2, c3, head, sub, leaves, total, label=str):
    """One 90-degree facet: headline, descriptor, then its own proportional leaves."""
    ax.add_patch(Wedge((0, 0), R1, start, start + 90, width=R1 - R0,
                       facecolor=c1, edgecolor="white", lw=1.1, zorder=3))
    ax.add_patch(Wedge((0, 0), R2, start, start + 90, width=R2 - R1,
                       facecolor=c2, edgecolor="white", lw=1.1, zorder=3))
    mid = start + 45
    ring_text(ax, mid, R0, R1, head, FS_HEAD, weight="bold")
    ring_text(ax, mid, R1, R2, sub, FS_SUB, color=INK)

    if not leaves:
        ax.add_patch(Wedge((0, 0), R3, start, start + 90, width=R3 - R2,
                           facecolor=shade(c3, 0.45), edgecolor="white", lw=1.1, zorder=3))
        return []

    outside, a = [], start + 90.0
    for i, (name, n) in enumerate(leaves):
        span = 90.0 * n / total
        ax.add_patch(Wedge((0, 0), R3, a - span, a, width=R3 - R2,
                           facecolor=shade(c3, 0.62 + 0.13 * (i % 3)),
                           edgecolor="white", lw=0.8, zorder=3))
        lmid = a - span / 2
        if span >= 80.0:      # a whole-quadrant summary, not a category
            ring_text(ax, lmid, R2, R3, label(name), FS_LEAF, tangential=True)
        elif span >= 3.6:
            ring_text(ax, lmid, R2, R3, label(name), FS_LEAF)
        else:
            rad = math.radians(lmid)
            outside.append({"angle": rad, "r0": R3, "y": (R3 + 0.05) * math.sin(rad),
                            "side": 1 if math.cos(rad) >= 0 else -1,
                            "text": f"{label(name)}  {n}"})
        a -= span
    return outside


def draw(d: dict, out_dir: Path, stem: str, family: str) -> list[Path]:
    from scripts.paper.figure2b_sunburst import place_outside
    plt.rcParams.update({"font.family": family, "text.color": INK,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    sc = d["scale"]
    fig = plt.figure(figsize=(FIG_IN, FIG_IN))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.92])
    ax.set_xlim(-1.62, 1.62); ax.set_ylim(-1.52, 1.62)
    ax.set_aspect("equal"); ax.axis("off")

    keep = str                                  # already cased in the source data
    cap = lambda s_: s_[:1].upper() + s_[1:]     # roles are lowercase prose
    content = {
        "questions": (f"{sc['questions']:,}\nQuestions", f"{len(d['qa_types'])} question types",
                      d["qa_types"].most_common(), sc["questions"], pretty),
        "cases": (f"{sc['cases']}\nCases", f"{len(d['sites'])} cancer sites",
                  d["sites"].most_common(), sc["cases"], keep),
        "specialists": (f"{len(d['roles'])}\nSpecialist roles",
                        f"{sc['utterances']:,} utterances", d["roles"].most_common(),
                        sc["questions"], cap),
        # One wedge, not a subdivision: these are summary statistics, and splitting the
        # ring would claim a proportion between a slide count and an utterance count.
        "corpus": (f"{sc['videos']}\nVideos", f"{sc['hours']:.0f} hours  ·  "
                   f"{sc['slides']:,} slides",
                   [(f"median {sc['slides_median']:.0f} slides per case (max "
                     f"{sc['slides_max']})\nmedian {sc['turns_median']:.0f} utterances "
                     f"per case", 1)], 1, keep),
    }
    outside = []
    for key, start, c1, c2, c3 in QUADS:
        head, sub, leaves, total, lab = content[key]
        outside += quadrant(ax, start, c1, c2, c3, head, sub, leaves, total, lab)
    place_outside(ax, outside, r=R3 + 0.05, gap=0.075)

    fig.suptitle("OpenTumorBoard at a glance", fontsize=FS_TITLE, fontweight="bold",
                 x=0.5, y=0.982, ha="center")
    fig.text(0.5, 0.948,
             "each quadrant is one quarter of the wheel; only the outer ring is "
             "proportional, and only within its own quadrant",
             fontsize=FS_NOTE, color=MUTED, ha="center", va="top", style="italic")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("pdf", "svg", "png"):
        p = out_dir / f"{stem}.{suffix}"
        fig.savefig(p, dpi=DPI, facecolor="white")
        written.append(p)
    plt.close(fig)
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--stem", default="fig2b_wheel")
    ap.add_argument("--font-family", default="DejaVu Sans")
    ap.add_argument("--font-dir", type=Path, default=None)
    a = ap.parse_args()

    font = resolve_font(a.font_family, a.font_dir)
    d = gather(a.manifest_dir)
    written = draw(d, a.output_dir, a.stem, a.font_family)

    prov = {
        "figure": "2b (wheel)", "sources": d["sources"], "font": font,
        "scale": d["scale"],
        "qa_types": dict(d["qa_types"].most_common()),
        "sites": dict(d["sites"].most_common()),
        "roles": dict(d["roles"].most_common()),
        "reading_note":
            "Quadrants are equal quarters, not proportional to one another - cases, "
            "questions, roles and videos are different units. Proportion is only "
            "meaningful along the outer ring within a single quadrant.",
        "outputs": [str(p) for p in written],
    }
    (a.output_dir / f"{a.stem}.provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": [str(p) for p in written], "scale": d["scale"]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
