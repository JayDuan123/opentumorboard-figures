#!/usr/bin/env python3
"""Figure 2b - benchmark statistics as nested rings.

Same four distributions as figure2b_statistics.py, drawn as sunbursts rather than
bars. The hierarchy is not invented for the picture: the inner ring of the cancer
panel is a grouping of the SAME seventeen site labels the bar version publishes, and
the role panel groups the same ten target roles by what the specialist does in a
board. No case or question is reclassified, so the two versions of Figure 2b report
identical counts and either can be used.

Discussion length stays a histogram. It is a distribution over a continuous variable,
and a ring would have to bin it into categories that do not exist in the data.

usage:
  python -m scripts.paper.figure2b_sunburst \
      --manifest-dir .../model_evaluation/manifests --output-dir .../figures/fig2b
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

from scripts.paper import style  # noqa: E402

from scripts.paper import cancer_sites  # noqa: E402

FIG_W_IN, FIG_H_IN = style.A4_W, 0.86 * style.A4_H
DPI = 600

FS_TITLE = 12.0
FS_PANEL = 8.0
FS_IN = 6.4
FS_OUT = 5.4
FS_NOTE = 5.8
FS_TICK = 6.4

INK = "#1b1b1b"
MUTED = "#646464"
GRID = "#d8d8d8"

TRAINVAL = "expert_qa_trainval_video_split_60_10_30_v4_qascreened_xsfixed_20260809.jsonl"
TEST = ("expert_qa_test_video_split_60_10_30_v3_qascreened_xsfixed_reanchored_"
        "rescreened_slidealigned_20260814.jsonl")
TASK1 = ("tumor_board_simulation_full_video_split_60_10_30_v4_deleaked_xsfixed_"
         "reanchored_20260808.jsonl")

# Seventeen published site labels -> organ system. A regrouping, not a relabelling.
SITE_SYSTEM = {
    "Thyroid / parathyroid": "Head & neck\n& endocrine",
    "Head & neck": "Head & neck\n& endocrine",
    "Genitourinary": "Genitourinary",
    "Thoracic / lung": "Thoracic",
    "Breast": "Breast",
    "CNS / brain & spine": "Neuro &\nskull base",
    "Skull base": "Neuro &\nskull base",
    "Hepatobiliary / pancreas": "Digestive",
    "Colorectal / anal": "Digestive",
    "Upper GI": "Digestive",
    "Peritoneum": "Digestive",
    "Gynecologic": "Gynecologic",
    "Skin / melanoma": "Skin &\nsarcoma",
    "Sarcoma (bone / soft tissue)": "Skin &\nsarcoma",
    "Hematologic": "Hematologic",
    "Neuroendocrine (site NOS)": "Other",
    "Unknown primary": "Other",
}

# Ten target roles -> what that specialist does in the meeting.
ROLE_GROUP = {
    "surgeon": "Treating", "medical oncologist": "Treating",
    "radiation oncologist": "Treating",
    "radiologist": "Diagnostic", "pathologist": "Diagnostic",
    "molecular pathologist": "Diagnostic",
    "genetic counselor": "Support", "nurse navigator": "Support",
    "clinical trial specialist": "Support",
    "other": "Unspecified",
}

# ColorBrewer-derived, one muted hue family per group; inner darker than outer.
FAMILIES = ["#8da0cb", "#a6bddb", "#9ebcda", "#b3cde3", "#c7d4e8",
            "#bcbddc", "#ccc4de", "#a8ddb5", "#d4b9da", "#cccccc"]


def refuse(msg: str) -> None:
    raise SystemExit(f"figure2b_sunburst: {msg}")


def load(p: Path) -> list[dict]:
    if not p.exists():
        refuse(f"missing manifest {p}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pretty(t: str) -> str:
    return t.replace("_", " ").capitalize()


def shade(hexcolor: str, factor: float) -> tuple:
    """Lighten toward white; the outer ring is a paler shade of its parent."""
    c = matplotlib.colors.to_rgb(hexcolor)
    return tuple(1 - (1 - v) * factor for v in c)


def gather(md: Path) -> dict:
    qa = load(md / TRAINVAL) + load(md / TEST)
    cases = load(md / TASK1)

    ids = {r["qa_id"] for r in qa}
    if len(ids) != len(qa):
        refuse("duplicate qa_id across the two QA manifests; the total would double-count")

    sites, audit = cancer_sites.site_counts(
        [({"video_uid": r["video_uid"], "case_id": r["case_id"]}, r["case_summary"])
         for r in cases])
    missing = set(sites) - set(SITE_SYSTEM)
    if missing:
        refuse(f"no organ system for site(s) {sorted(missing)}; add them to SITE_SYSTEM "
               "rather than letting a case fall out of the ring")

    roles = collections.Counter(r["target_specialist_role"] for r in qa)
    missing = set(roles) - set(ROLE_GROUP)
    if missing:
        refuse(f"no group for role(s) {sorted(missing)}; add them to ROLE_GROUP")

    qa_types = collections.Counter(r["qa_type"] for r in qa)
    turns = [len(r["reference_discussion"]) for r in cases]
    slides = [str(r.get("slides") or "").count("<image>") for r in cases]
    dur = [r["case_end_sec"] - r["case_start_sec"] for r in cases]

    for name, counter, total in (("site", sites, len(cases)),
                                 ("role", roles, len(qa)),
                                 ("qa_type", qa_types, len(qa))):
        if sum(counter.values()) != total:
            refuse(f"{name} counts sum to {sum(counter.values())}, not {total}")

    return {
        "sites": sites, "roles": roles, "qa_types": qa_types, "turns": turns,
        "audit": audit,
        "scale": {
            "videos": len({r["video_uid"] for r in cases}), "cases": len(cases),
            "questions": len(qa), "slides": sum(slides),
            "slides_median": statistics.median(slides),
            "hours": sum(dur) / 3600.0,
            "turns_median": statistics.median(turns),
            "turns_mean": statistics.mean(turns),
            "turns_min": min(turns), "turns_max": max(turns),
            "utterances": sum(turns),
        },
        "sources": {n: {"path": str((md / n).resolve()), "sha256": sha256(md / n)}
                    for n in (TRAINVAL, TEST, TASK1)},
    }


def nest(counter: collections.Counter, mapping: dict) -> list[tuple]:
    """[(group, total, [(leaf, n), ...]), ...], both levels largest-first."""
    groups = collections.defaultdict(list)
    for leaf, n in counter.items():
        groups[mapping[leaf]].append((leaf, n))
    out = [(g, sum(n for _, n in leaves), sorted(leaves, key=lambda x: -x[1]))
           for g, leaves in groups.items()]
    return sorted(out, key=lambda x: -x[1])




def fit_radial(label: str, band: float, fs_max: float) -> tuple[str, float]:
    """Wrap and shrink a radial label so it stays inside its own ring band.

    Radial text is limited by the band width, not by the arc, and these names run to
    28 characters. Unwrapped they print across the neighbouring ring and land on top
    of its labels - which is what the first draft of this panel did.
    """
    text = label
    if len(label) > 11:
        for sep in (" / ", " & ", " "):
            if sep in label:
                parts = label.split(sep)
                best, cut = None, None
                keep = sep.strip() if sep.strip() else ""
                for i in range(1, len(parts)):
                    a = sep.join(parts[:i]).strip()
                    b = sep.join(parts[i:]).strip()
                    score = abs(len(a) - len(b))
                    if best is None or score < best:
                        best, cut = score, (a, b)
                head = cut[0] + (" " + keep if keep else "")
                text = head + "\n" + cut[1]
                break
    longest = max(len(line) for line in text.split("\n"))
    # ~0.62 em advance, and one x-unit is about 72 pt on this canvas
    fs = min(fs_max, band * 72.0 / max(1, longest) / 0.62)
    return text, max(4.2, fs)


def place_outside(ax, items, r=1.09, gap=0.085):
    """Draw leader-line labels, pushed apart so none overlaps another.

    Small wedges cluster - nine of the seventeen sites are under 3% - and their
    labels land at almost the same angle. Drawn where they fall, they overprint each
    other and the panel loses exactly the categories the leader lines were for.
    Each side is sorted by y and spread to a minimum separation.
    """
    for side in (1, -1):
        rows = sorted([it for it in items if it["side"] == side], key=lambda it: it["y"])
        for i in range(1, len(rows)):                       # push up from the bottom
            rows[i]["y"] = max(rows[i]["y"], rows[i - 1]["y"] + gap)
        if rows:
            over = rows[-1]["y"] - 1.30
            if over > 0:                                    # then recentre if it ran off
                for it in rows:
                    it["y"] -= over
        for it in rows:
            a = it["angle"]
            x0, y0 = it["r0"] * math.cos(a), it["r0"] * math.sin(a)
            x1 = r * math.cos(a)
            x2 = x1 + 0.11 * side
            ax.plot([x0, x1, x2], [y0, it["y"], it["y"]], color=MUTED, lw=0.45, zorder=4)
            ax.text(x2 + 0.025 * side, it["y"], it["text"], fontsize=FS_OUT - 0.5,
                    ha="left" if side > 0 else "right", va="center", color=MUTED, zorder=6)


def sunburst(ax, nested: list[tuple], total: int, r_in=0.27, r_mid=0.70, r_out=1.0,
             min_deg_inner=21.0, min_deg_outer=8.5) -> None:
    """Two rings. Wedges too thin to hold their label get one outside with a leader.

    A sunburst that drops the labels it cannot fit is a picture of the big categories
    only, which is the opposite of what a distribution panel is for.
    """
    ax.set_xlim(-1.62, 1.62); ax.set_ylim(-1.42, 1.42)
    ax.set_aspect("equal"); ax.axis("off")

    outside = []
    start = 90.0
    for gi, (group, gtot, leaves) in enumerate(nested):
        base = FAMILIES[gi % len(FAMILIES)]
        span = 360.0 * gtot / total
        ax.add_patch(Wedge((0, 0), r_mid, start - span, start, width=r_mid - r_in,
                           facecolor=base, edgecolor="white", lw=0.9, zorder=3))
        mid = start - span / 2
        if span >= min_deg_inner:
            rot = mid + 180 if 90 < mid % 360 < 270 else mid
            # A radial label is bounded by the band it sits in, not by the arc, so a
            # long name has to shrink or it prints across the ring outside it.
            gtext, fs = fit_radial(group.replace("\n", " "), r_mid - r_in, FS_IN)
            ax.text((r_in + r_mid) / 2 * math.cos(math.radians(mid)),
                    (r_in + r_mid) / 2 * math.sin(math.radians(mid)),
                    gtext, rotation=rot, rotation_mode="anchor", ha="center",
                    va="center", fontsize=fs, fontweight="bold", color=INK,
                    linespacing=0.92, zorder=6)
        else:
            a = math.radians(mid)
            outside.append({"angle": a, "r0": r_mid, "y": 1.09 * math.sin(a),
                            "side": 1 if math.cos(a) >= 0 else -1,
                            "text": group.replace("\n", " ")})

        lstart = start
        for li, (leaf, n) in enumerate(leaves):
            lspan = 360.0 * n / total
            ax.add_patch(Wedge((0, 0), r_out, lstart - lspan, lstart,
                               width=r_out - r_mid, facecolor=shade(base, 0.52 + 0.10 * (li % 3)),
                               edgecolor="white", lw=0.8, zorder=3))
            lmid = lstart - lspan / 2
            label = leaf if isinstance(leaf, str) else str(leaf)
            if len(leaves) == 1:
                # The inner ring already names it; repeating it prints one label on
                # top of the other at the same angle.
                lstart -= lspan
                continue
            if lspan >= min_deg_outer:
                rot = lmid + 180 if 90 < lmid % 360 < 270 else lmid
                ltext, lfs = fit_radial(label, r_out - r_mid, FS_OUT)
                ax.text((r_mid + r_out) / 2 * math.cos(math.radians(lmid)),
                        (r_mid + r_out) / 2 * math.sin(math.radians(lmid)),
                        ltext, rotation=rot, rotation_mode="anchor", ha="center",
                        va="center", fontsize=lfs, color=INK, linespacing=0.92,
                        zorder=6)
            else:
                a = math.radians(lmid)
                outside.append({"angle": a, "r0": r_out, "y": 1.09 * math.sin(a),
                                "side": 1 if math.cos(a) >= 0 else -1,
                                "text": f"{label}  {n}"})
            lstart -= lspan
        start -= span
    place_outside(ax, outside)


def donut(ax, counter: collections.Counter, total: int, r_in=0.46, r_out=1.0,
          min_deg=13.0) -> None:
    """Single ring, for a flat taxonomy. Grouping the nine QA types would mean
    inventing a level the benchmark does not define."""
    ax.set_xlim(-1.62, 1.62); ax.set_ylim(-1.42, 1.42)
    ax.set_aspect("equal"); ax.axis("off")
    outside = []
    start = 90.0
    for i, (k, n) in enumerate(counter.most_common()):
        span = 360.0 * n / total
        base = FAMILIES[i % len(FAMILIES)]
        ax.add_patch(Wedge((0, 0), r_out, start - span, start, width=r_out - r_in,
                           facecolor=shade(base, 0.80), edgecolor="white", lw=0.9, zorder=3))
        mid = start - span / 2
        if span >= min_deg:
            rot = mid + 180 if 90 < mid % 360 < 270 else mid
            ax.text((r_in + r_out) / 2 * math.cos(math.radians(mid)),
                    (r_in + r_out) / 2 * math.sin(math.radians(mid)),
                    f"{pretty(k)}\n{100 * n / total:.0f}%", rotation=rot,
                    rotation_mode="anchor", ha="center", va="center", fontsize=FS_OUT,
                    color=INK, linespacing=0.95, zorder=6)
        else:
            a = math.radians(mid)
            outside.append({"angle": a, "r0": r_out, "y": 1.09 * math.sin(a),
                            "side": 1 if math.cos(a) >= 0 else -1,
                            "text": f"{pretty(k)}  {100 * n / total:.1f}%"})
        start -= span
    place_outside(ax, outside)


def hist(ax, turns: list[int], sc: dict) -> None:
    BIN, BIN_MAX = 10, 90
    counts = collections.Counter(min(t // BIN * BIN, BIN_MAX) for t in turns)
    edges = list(range(0, BIN_MAX + 1, BIN))
    ax.bar(edges, [counts.get(e, 0) for e in edges], width=BIN * 0.86, align="edge",
           color="#b3cde3", edgecolor="#6a8fc0", lw=0.7, zorder=3)
    ax.axvline(sc["turns_median"], color="#c1553b", lw=1.3, ls="--", zorder=4)
    ax.text(sc["turns_median"] + 2, ax.get_ylim()[1] * 0.92,
            f"median {sc['turns_median']:.0f}", fontsize=FS_NOTE - 0.4, color="#c1553b",
            va="top")
    ax.set_xticks(edges)
    ax.set_xticklabels([str(e) for e in edges[:-1]] + [f"{BIN_MAX}+"], fontsize=FS_TICK)
    ax.tick_params(labelsize=FS_TICK, length=2)
    ax.set_xlabel("utterances per case", fontsize=FS_NOTE)
    ax.set_ylabel("cases", fontsize=FS_NOTE)
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def resolve_font(family: str, font_dir: Path | None) -> dict:
    registered = []
    if font_dir is not None:
        if not font_dir.is_dir():
            refuse(f"--font-dir {font_dir} is not a directory")
        for p in sorted(font_dir.glob("*")):
            if p.suffix.lower() in (".ttf", ".otf", ".ttc"):
                fm.fontManager.addfont(str(p)); registered.append(p.name)
    try:
        found = fm.findfont(fm.FontProperties(family=family), fallback_to_default=False)
    except ValueError:
        refuse(f"font '{family}' is not installed and no file for it was found")
    if fm.FontProperties(fname=found).get_name().lower() != family.lower():
        refuse(f"font '{family}' resolved to '{found}' - wrong typeface, refusing")
    return {"family": family, "file": found, "registered": registered}


def draw(d: dict, out_dir: Path, stem: str, family: str) -> list[Path]:
    style.apply()
    plt.rcParams.update({"text.color": INK})
    sc = d["scale"]
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    gs = fig.add_gridspec(2, 2, left=0.045, right=0.955, top=0.905, bottom=0.085,
                          wspace=0.02, hspace=0.02, height_ratios=[1.0, 0.90])

    ax = fig.add_subplot(gs[0, 0])
    sunburst(ax, nest(d["sites"], SITE_SYSTEM), sc["cases"])
    ax.set_title(f"Primary cancer site\n{sc['cases']} cases  ·  {len(d['sites'])} sites",
                 fontsize=FS_PANEL, fontweight="bold", pad=2)

    ax = fig.add_subplot(gs[0, 1])
    sunburst(ax, nest(d["roles"], ROLE_GROUP), sc["questions"])
    ax.set_title(f"Target specialist\n{sc['questions']:,} questions  ·  "
                 f"{len(d['roles'])} roles", fontsize=FS_PANEL, fontweight="bold", pad=2)

    ax = fig.add_subplot(gs[1, 0])
    donut(ax, d["qa_types"], sc["questions"])
    ax.set_title(f"Question type\n{len(d['qa_types'])} types",
                 fontsize=FS_PANEL, fontweight="bold", pad=2)

    ax = fig.add_subplot(gs[1, 1])
    hist(ax, d["turns"], sc)
    ax.set_title(f"Discussion length\n{sc['utterances']:,} utterances",
                 fontsize=FS_PANEL, fontweight="bold", pad=2)
    ax.set_position([0.60, 0.155, 0.33, 0.245])

    fig.suptitle("Benchmark statistics", fontsize=FS_TITLE, fontweight="bold",
                 x=0.045, y=0.972, ha="left")
    fig.text(0.045, 0.030,
             f"{sc['videos']} videos  ·  {sc['hours']:.0f} hours  ·  {sc['cases']} cases  ·  "
             f"{sc['slides']:,} slides (median {sc['slides_median']:.0f}/case)  ·  "
             f"{sc['questions']:,} questions",
             fontsize=FS_NOTE, color=MUTED, ha="left", va="bottom")

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
    ap.add_argument("--stem", default="fig2b_sunburst")
    ap.add_argument("--font-family", default="DejaVu Sans")
    ap.add_argument("--font-dir", type=Path, default=None)
    a = ap.parse_args()

    font = resolve_font(a.font_family, a.font_dir)
    d = gather(a.manifest_dir)
    written = draw(d, a.output_dir, a.stem, a.font_family)

    prov = {
        "figure": "2b (sunburst)", "sources": d["sources"], "font": font,
        "scale": d["scale"],
        "sites": dict(d["sites"].most_common()),
        "site_systems": {g: t for g, t, _ in nest(d["sites"], SITE_SYSTEM)},
        "roles": dict(d["roles"].most_common()),
        "role_groups": {g: t for g, t, _ in nest(d["roles"], ROLE_GROUP)},
        "qa_types": dict(d["qa_types"].most_common()),
        "hierarchy_note":
            "Inner rings regroup the published leaf labels; no case or question is "
            "reclassified, so counts match figure2b_statistics.py exactly.",
        "outputs": [str(p) for p in written],
    }
    (a.output_dir / f"{a.stem}.provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": [str(p) for p in written],
                      "site_systems": prov["site_systems"],
                      "role_groups": prov["role_groups"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
