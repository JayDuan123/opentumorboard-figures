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
    ROLE_GROUP, SITE_SYSTEM, TRAINVAL, TEST, TASK1, fit_radial, load, nest,
    pretty, refuse, resolve_font, sha256,
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

R0, R1, R2, R3 = 0.24, 0.48, 0.70, 1.04

# One hue per quadrant, given as (ring1, ring2, ring3 base). Inner is the saturated
# end so the eye lands on the headline number first.
QUADS = [
    ("questions", 90.0, "#2aa3d4", "#54bde8", "#7cd0f0"),
    ("cases", 0.0, "#2eaa4e", "#48c268", "#6ed889"),
    ("corpus", 270.0, "#e07a42", "#ef9a67", "#f6bd97"),
    ("specialists", 180.0, "#2b7fa8", "#4198c2", "#67b3d8"),
]


# Display-only short names. The compound "A / B" labels are the ones that crowd the
# ring, and on a wheel the second half is what a reader drops anyway. This does NOT
# touch the canonical labels in cancer_sites.py: those are what the audited
# per-case file records, and the bar and nested-ring versions still print them in
# full. Only the wheel's ink changes; every count is untouched.
SHORT_SITE = {
    "Thyroid / parathyroid": "Thyroid",
    "Thoracic / lung": "Lung",
    "CNS / brain & spine": "CNS",
    "Hepatobiliary / pancreas": "Hepatobiliary",
    "Colorectal / anal": "Colorectal",
    "Skin / melanoma": "Skin",
    "Sarcoma (bone / soft tissue)": "Sarcoma",
    "Neuroendocrine (site NOS)": "Neuroendocrine",
}
SHORT_GROUP = {
    "Head & neck\n& endocrine": "Head & neck",
    "Neuro &\nskull base": "Neuro",
    "Skin &\nsarcoma": "Skin & sarcoma",
}


def short_site(name: str) -> str:
    return SHORT_SITE.get(name, name)


def shade(hexcolor: str, factor: float) -> tuple:
    c = matplotlib.colors.to_rgb(hexcolor)
    return tuple(1 - (1 - v) * factor for v in c)


def darken(hexcolor: str, factor: float = 0.82) -> tuple:
    c = matplotlib.colors.to_rgb(hexcolor)
    return tuple(v * factor for v in c)


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
    # Per video, the minutes the benchmark actually covers - the sum of its cases'
    # spans. There is no true runtime field anywhere in the manifests, and the gap
    # between the two is real (intros, breaks, cases that were cut), so the ring is
    # labelled "analysed" rather than being passed off as video length.
    per_video = collections.defaultdict(float)
    for r in cases:
        per_video[r["video_uid"]] += r["case_end_sec"] - r["case_start_sec"]
    BINS = [(0, 20, "under 20 min"), (20, 40, "20\u201340 min"), (40, 60, "40\u201360 min"),
            (60, 90, "60\u201390 min"), (90, float("inf"), "over 90 min")]
    lengths = collections.Counter()
    for v in per_video.values():
        m = v / 60.0
        for lo, hi, lab in BINS:
            if lo <= m < hi:
                lengths[lab] += 1
                break
    if sum(lengths.values()) != len(per_video):
        refuse("video length bins do not cover every video")

    turns = [len(r["reference_discussion"]) for r in cases]
    slides = [str(r.get("slides") or "").count("<image>") for r in cases]
    dur = [r["case_end_sec"] - r["case_start_sec"] for r in cases]
    import statistics
    return {
        "qa_types": qa_types, "sites": sites, "roles": roles,
        "lengths": [(lab, lengths[lab]) for _, _, lab in BINS if lengths[lab]],
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
    # fit_radial can drop a long single word to the floor while a short one keeps the
    # maximum, and a ring set in two visibly different sizes reads as a mistake. Hold
    # the spread to 18 per cent and let a long word run marginally into the gutter.
    fitted = max(fitted, fs * 0.82)
    rot = mid + 180 if 90 < mid % 360 < 270 else mid
    rr = (r_lo + r_hi) / 2
    ax.text(rr * math.cos(math.radians(mid)), rr * math.sin(math.radians(mid)), txt,
            rotation=rot, rotation_mode="anchor", ha="center", va="center",
            fontsize=fitted, fontweight=weight, color=color, linespacing=0.94, zorder=6)



def leaf_ring(ax, start, c3, leaves, total, label=str):
    """Outer ring for a facet with no grouping level above it."""
    outside, a = [], start + 90.0
    for i, (name, n) in enumerate(leaves):
        span = 90.0 * n / total
        ax.add_patch(Wedge((0, 0), R3, a - span, a, width=R3 - R2,
                           facecolor=shade(c3, 0.58 + 0.13 * (i % 3)),
                           edgecolor="white", lw=0.8, zorder=3))
        lmid = a - span / 2
        if span >= 80.0:            # a whole-quadrant summary, not a category
            ring_text(ax, lmid, R2, R3, label(name), FS_LEAF, tangential=True)
        elif span >= 2.8:
            ring_text(ax, lmid, R2, R3, label(name), FS_LEAF)
        else:
            rad = math.radians(lmid)
            outside.append({"angle": rad, "r0": R3, "y": (R3 + 0.05) * math.sin(rad),
                            "side": 1 if math.cos(rad) >= 0 else -1,
                            "text": f"{label(name)}  {n}"})
        a -= span
    return outside


def quadrant(ax, start, c1, c2, c3, head, sub, nested, total, label=str):
    """One 90-degree facet, three data rings deep.

    Ring 1 is the headline. Ring 2 is a grouping where the facet has one - organ
    system for cancer sites, board function for specialist roles - and a plain
    descriptor where it does not: the nine question types have no level above them
    that the benchmark defines, and inventing one to fill the ring would be a
    taxonomy drawn for the picture rather than for the data.
    """
    # The hub carries the quadrant's colour too, one shade deeper than ring 1, so
    # each facet reads as one continuous block from the centre outwards instead of
    # four coloured arcs floating around a white disc.
    ax.add_patch(Wedge((0, 0), R0, start, start + 90,
                       facecolor=darken(c1), edgecolor="white", lw=1.1, zorder=3))
    ax.add_patch(Wedge((0, 0), R1, start, start + 90, width=R1 - R0,
                       facecolor=c1, edgecolor="white", lw=1.1, zorder=3))
    mid = start + 45
    ring_text(ax, mid, R0, R1, head, FS_HEAD, weight="bold")

    if nested is None:                      # no grouping: one descriptor band
        ax.add_patch(Wedge((0, 0), R2, start, start + 90, width=R2 - R1,
                           facecolor=c2, edgecolor="white", lw=1.1, zorder=3))
        ring_text(ax, mid, R1, R2, sub, FS_SUB, tangential=len(sub) > 24)
        return []

    outside, a = [], start + 90.0
    for gi, (group, gtot, leaves) in enumerate(nested):
        gspan = 90.0 * gtot / total
        ax.add_patch(Wedge((0, 0), R2, a - gspan, a, width=R2 - R1,
                           facecolor=shade(c2, 0.72 + 0.14 * (gi % 2)),
                           edgecolor="white", lw=1.0, zorder=3))
        gmid = a - gspan / 2
        gname = SHORT_GROUP.get(group, group.replace("\n", " "))
        if len(leaves) == 1:
            gname = ""            # the leaf ring names it; two labels for one thing
        if gname and gspan >= 4.0:
            ring_text(ax, gmid, R1, R2, gname, FS_SUB - 0.4, weight="bold")
        elif gname:
            rad = math.radians(gmid)
            outside.append({"angle": rad, "r0": R3, "y": (R3 + 0.05) * math.sin(rad),
                            "side": 1 if math.cos(rad) >= 0 else -1,
                            "text": f"{gname}  {gtot}"})
        b = a
        for li, (name, n) in enumerate(leaves):
            lspan = 90.0 * n / total
            ax.add_patch(Wedge((0, 0), R3, b - lspan, b, width=R3 - R2,
                               facecolor=shade(c3, 0.58 + 0.13 * (li % 3)),
                               edgecolor="white", lw=0.8, zorder=3))
            lmid = b - lspan / 2
            if lspan >= 80.0:
                ring_text(ax, lmid, R2, R3, label(name), FS_LEAF,
                          tangential=True)
            elif lspan >= 2.8:
                ring_text(ax, lmid, R2, R3, label(name), FS_LEAF)
            else:
                rad = math.radians(lmid)
                outside.append({"angle": rad, "r0": R3,
                                "y": (R3 + 0.05) * math.sin(rad),
                                "side": 1 if math.cos(rad) >= 0 else -1,
                                "text": f"{label(name)}  {n}"})
            b -= lspan
        a -= gspan
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

    # A one-wedge "grouping" so the corpus quadrant keeps the same ring structure
    # without claiming a proportion between a slide count and an utterance count.
    content = {
        "questions": (f"{sc['questions']:,}\nQuestions", f"{len(d['qa_types'])} question types",
                      None, sc["questions"], pretty, d["qa_types"].most_common()),
        "cases": (f"{sc['cases']}\nCases", "", nest(d["sites"], SITE_SYSTEM),
                  sc["cases"], short_site, None),
        "specialists": (f"{len(d['roles'])}\nSpecialist roles", "",
                        nest(d["roles"], ROLE_GROUP), sc["questions"], cap, None),
        "corpus": (f"{sc['videos']}\nVideos",
                   f"{sc['hours']:.0f} h analysed  ·  {sc['slides']:,} slides",
                   None, sc["videos"], keep, d["lengths"]),
    }
    outside = []
    for key, start, c1, c2, c3 in QUADS:
        head, sub, nested, total, lab, flat = content[key]
        if nested is None:                       # flat facet: descriptor ring, then leaves
            outside += quadrant(ax, start, c1, c2, c3, head, sub, None, total, lab)
            outside += leaf_ring(ax, start, c3, flat, total, lab)
        else:
            outside += quadrant(ax, start, c1, c2, c3, head, sub, nested, total, lab)
    place_outside(ax, outside, r=R3 + 0.05, gap=0.075)

    fig.text(0.5, 0.022,
             f"median {sc['slides_median']:.0f} slides per case (max {sc['slides_max']})"
             f"   ·   median {sc['turns_median']:.0f} utterances per case   ·   "
             f"outer ring of the corpus quadrant bins videos by the minutes the "
             f"benchmark covers, not by their full runtime",
             fontsize=FS_NOTE - 0.4, color=MUTED, ha="center", va="bottom")
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
        "video_analysed_minutes": dict(d["lengths"]),
        "sites": dict(d["sites"].most_common()),
        "roles": dict(d["roles"].most_common()),
        "display_aliases": {"sites": SHORT_SITE,
                            "groups": {k.replace("\n", " "): v
                                       for k, v in SHORT_GROUP.items()},
                            "note": "shortened on the wheel only; counts and the canonical labels are unchanged"},
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
