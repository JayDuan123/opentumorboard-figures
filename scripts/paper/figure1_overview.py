#!/usr/bin/env python3
"""Figure 1 - what the benchmark is: one real tumor board, two tasks.

The teaser has to answer 'what does a model actually get, and what is it scored
against'. So it is built around ONE real case, read from the released manifests at
run time rather than typed in. Every string on the canvas - the diagnosis, the slide
captions, the question, the reference answer, the role sequence, the conclusion -
comes from the record identified by --video-uid / --case-id, and the script refuses
if that record is missing or if the QA it draws does not belong to that case. A
teaser that illustrates the benchmark with invented clinical text is a liability.

The left-to-right reading is source -> shared input -> two tasks. Both tasks receive
the SAME input block, which is the point of the middle column: what separates them is
what is demanded as output, not what they are told.

usage:
  python -m scripts.paper.figure1_overview \
      --manifest-dir .../model_evaluation/manifests --output-dir .../figures/fig1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

from scripts.paper import style  # noqa: E402

FIG_W_IN = style.A4_W
FIG_H_IN = 0.44 * style.A4_H
DPI = 600

FS_TITLE = 11.5
FS_COL = 6.6
FS_BODY = 5.3
FS_SMALL = 4.8
FS_CHIP = 4.9
FS_SCALE = 5.2

INK = "#1b1b1b"
MUTED = "#646464"
LINE = "#8c8c8c"
GRID = "#d8d8d8"
PANEL = "#f7f7f7"
WHITE = "#ffffff"

# Same ColorBrewer Paired cool pairs as Fig 2a. Purple = source data,
# blue = Task 2, green = Task 1. Colour identifies a task, never quality.
BLUE, BLUE_LIGHT = "#1f78b4", "#a6cee3"
GREEN, GREEN_LIGHT = "#33a02c", "#b2df8a"
PURPLE, PURPLE_LIGHT = "#6a3d9a", "#cab2d6"

TASK1 = ("tumor_board_simulation_test_video_split_60_10_30_v3_deleaked_repaired_"
         "xsfixed_reanchored_20260808.jsonl")
TASK2 = ("expert_qa_test_video_split_60_10_30_v3_qascreened_xsfixed_reanchored_"
         "rescreened_slidealigned_20260814.jsonl")
FULL1 = "tumor_board_simulation_full_video_split_60_10_30_v4_deleaked_xsfixed_reanchored_20260808.jsonl"
TRAINVAL2 = "expert_qa_trainval_video_split_60_10_30_v4_qascreened_xsfixed_20260809.jsonl"


def refuse(msg: str) -> None:
    raise SystemExit(f"figure1: {msg}")


def load(path: Path) -> list[dict]:
    if not path.exists():
        refuse(f"missing manifest {path}")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def section(summary: str, name: str) -> str:
    m = re.search(rf"\[\s*{name}\s*\]\s*:(.*?)(?=\n\s*\[|\Z)", summary, re.S | re.I)
    if not m:
        refuse(f"case_summary has no [ {name} ] section")
    return " ".join(m.group(1).split())


def clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip(" ,;.") + "…"


def gather(manifest_dir: Path, video_uid: str, case_id: str,
           qa_index: int | None, qa_id: str | None) -> dict:
    cases = load(manifest_dir / TASK1)
    qas = load(manifest_dir / TASK2)

    hit = [r for r in cases if r["video_uid"] == video_uid and r["case_id"] == case_id]
    if len(hit) != 1:
        refuse(f"{len(hit)} Task 1 records for {video_uid}/{case_id}; need exactly 1")
    case = hit[0]

    mine = [q for q in qas if q["video_uid"] == video_uid and q["case_id"] == case_id]
    if not mine:
        refuse(f"no Task 2 questions for {video_uid}/{case_id}; the two panels would "
               "describe different cases")
    if qa_id:
        pick = [q for q in mine if q["qa_id"] == qa_id]
        if not pick:
            refuse(f"--qa-id {qa_id} is not a question of {video_uid}/{case_id}")
        qa = pick[0]
    elif qa_index is None:
        # Shortest-answer wins picks vacuous pairs ("That is certainly possible"),
        # so aim for a substantive answer that still fits two lines.
        qa = min(mine, key=lambda q: abs(len(q["reference_answer"]) - 145)
                 + 2 * max(0, len(q["question"]) - 110))
    else:
        if not 0 <= qa_index < len(mine):
            refuse(f"--qa-index {qa_index} out of range (case has {len(mine)} questions)")
        qa = sorted(mine, key=lambda q: q["qa_index"])[qa_index]

    # The role sequence drawn under Task 1 must be the real one, in order.
    seq, seen = [], set()
    for u in case["reference_discussion"]:
        role = u["inferred_role"]
        if role not in seen:
            seen.add(role)
            seq.append(role)

    captions = [c.strip() for c in case["slides"].split("<image>") if c.strip()]
    if qa["case_summary"] != case["case_summary"]:
        refuse("the Task 1 and Task 2 records for this case carry different case_summary "
               "text; the shared-input column would be a fiction")

    # Corpus-level scale, so the teaser's numbers match Fig 2b's.
    allcases = load(manifest_dir / FULL1)
    allqa = load(manifest_dir / TRAINVAL2) + qas
    videos = {r["video_uid"] for r in allcases}
    hours = sum(r["case_end_sec"] - r["case_start_sec"] for r in allcases) / 3600.0

    return {
        "case": case,
        "qa": qa,
        "role_sequence": seq,
        "captions": captions,
        "demographics": section(case["case_summary"], "patient demographics"),
        "diagnosis": section(case["case_summary"], "diagnosis"),
        "complaint": section(case["case_summary"], "chief complaint"),
        "minutes": (case["case_end_sec"] - case["case_start_sec"]) / 60.0,
        "n_turns": len(case["reference_discussion"]),
        "n_qa_case": len(mine),
        "scale": {"videos": len(videos), "cases": len(allcases),
                  "questions": len(allqa), "hours": hours},
        "sources": {n: {"path": str((manifest_dir / n).resolve()),
                        "sha256": sha256(manifest_dir / n)}
                    for n in (TASK1, TASK2, FULL1, TRAINVAL2)},
    }


def resolve_font(family: str, font_dir: Path | None) -> dict:
    """matplotlib falls back silently on a missing family; a paper figure must not."""
    registered = []
    if font_dir is not None:
        if not font_dir.is_dir():
            refuse(f"--font-dir {font_dir} is not a directory")
        for p in sorted(font_dir.glob("*")):
            if p.suffix.lower() in (".ttf", ".otf", ".ttc"):
                fm.fontManager.addfont(str(p))
                registered.append(p.name)
    try:
        found = fm.findfont(fm.FontProperties(family=family), fallback_to_default=False)
    except ValueError:
        refuse(f"font '{family}' not installed and no file found"
               + (f" in {font_dir}" if font_dir else " (no --font-dir)")
               + f". Registered this run: {registered or 'none'}.")
    if fm.FontProperties(fname=found).get_name().lower() != family.lower():
        refuse(f"font '{family}' resolved to something else ({found}) - refusing")
    return {"family": family, "file": found, "registered": registered}


# ---------------------------------------------------------------- primitives
def box(ax, x, y, w, h, fc=WHITE, ec=LINE, lw=0.6, r=1.3, z=2, ls="solid"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, linestyle=ls))


def label(ax, x, y, s, fs=FS_BODY, color=INK, ha="left", va="top", weight="normal", z=4):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, fontweight=weight, zorder=z)


def wrap(ax, x, y, s, width, fs=FS_BODY, color=INK, lh=2.45, z=4, weight="normal",
         floor=None, what=""):
    """Draw wrapped text; return the y of the next free line.

    `floor` is the inside bottom of the enclosing box. Text that would cross it is a
    layout bug, and a teaser that silently spills clinical text out of its own panel
    is worse than one that refuses to build - so it refuses.
    """
    lines = textwrap.wrap(s, width)
    for i, line in enumerate(lines):
        ax.text(x, y - i * lh, line, fontsize=fs, color=color, ha="left", va="top",
                fontweight=weight, zorder=z)
    end = y - len(lines) * lh
    if floor is not None and end < floor:
        refuse(f"{what or 'a text block'} needs {len(lines)} lines and overflows its box "
               f"by {floor - end:.1f} units - shorten the clip or grow the box")
    return end


def chip_width(s, fs=FS_CHIP, pad=1.15):
    """1 x-unit = 0.07 in at this canvas; a lowercase advance is about 0.5 em."""
    return len(s) * fs * 0.101 + 2 * pad


def chip(ax, x, y, s, fc, ec, fs=FS_CHIP, pad=1.15, h=3.0):
    w = chip_width(s, fs, pad)
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=1.4",
                                fc=fc, ec=ec, lw=0.5, zorder=3))
    ax.text(x + w / 2, y, s, fontsize=fs, color=INK, ha="center", va="center", zorder=4)
    return x + w


def arrow(ax, x0, y0, x1, y1, color=LINE, lw=0.9, z=3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=6.5, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=z))


def draw(d: dict, out_dir: Path, stem: str, family: str) -> list[Path]:
    style.apply()
    plt.rcParams.update({"text.color": INK})
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    case, qa, sc = d["case"], d["qa"], d["scale"]

    ax.text(1.6, 97.2, "OpenTumorBoard", fontsize=FS_TITLE, fontweight="bold", va="top")
    ax.text(1.6, 91.6, "Real multidisciplinary tumor boards, scored as two tasks",
            fontsize=FS_COL - 0.7, color=MUTED, va="top")

    # ---------------------------------------------------------- 1. source
    L, LW = 1.6, 21.0
    label(ax, L, 85.0, "REAL TUMOR BOARD", FS_COL, PURPLE, weight="bold")
    box(ax, L, 43.5, LW, 38.0, fc=PANEL, ec=PURPLE_LIGHT)
    box(ax, L + 1.4, 71.0, LW - 2.8, 8.6, fc="#e8e2f0", ec=PURPLE_LIGHT, r=1.0)
    ax.add_patch(Polygon([[L + 8.9, 73.4], [L + 8.9, 77.2], [L + 12.2, 75.3]],
                         closed=True, fc=PURPLE, ec="none", zorder=4))
    ax.text(L + LW / 2, 68.4, f"{d['minutes']:.0f} min of recorded discussion",
            fontsize=FS_SMALL, color=MUTED, ha="center", va="top", zorder=4)
    y = wrap(ax, L + 1.4, 64.8, d["demographics"], 30, FS_BODY, INK, weight="bold")
    y = wrap(ax, L + 1.4, y - 0.6, clip(d["complaint"], 130), 33, FS_SMALL, MUTED,
             lh=2.25, floor=51.0, what="source-panel chief complaint")
    ax.text(L + 1.4, y - 1.4, f"{d['n_turns']} turns  ·  {len(d['role_sequence'])} specialists"
            f"  ·  {d['n_qa_case']} questions", fontsize=FS_SMALL - 0.3, color=PURPLE,
            va="top", zorder=4)

    ax.plot([L, L + LW], [39.4, 39.4], color=GRID, lw=0.6, zorder=2)
    for i, (v, k) in enumerate([(f"{sc['videos']}", "videos"), (f"{sc['hours']:.0f}", "hours"),
                                (f"{sc['cases']}", "cases"),
                                (f"{sc['questions']:,}", "questions")]):
        cx = L + 2.6 + i * 5.4
        ax.text(cx, 36.2, v, fontsize=FS_SCALE, fontweight="bold", ha="center", va="top")
        ax.text(cx, 32.5, k, fontsize=FS_SMALL - 0.3, color=MUTED, ha="center", va="top")

    label(ax, L, 25.4, "HOW IT IS BUILT", FS_COL - 0.9, MUTED, weight="bold")
    wrap(ax, L, 21.6, "ASR  ·  diarization  ·  case segmentation  ·  slide extraction  ·  "
         "role inference  ·  alignment  ·  QA generation", 34, FS_SMALL, MUTED, lh=2.3)
    ax.text(L, 10.6, "nine stages, detailed in Fig. 2a", fontsize=FS_SMALL - 0.3,
            color=MUTED, va="top", style="italic", zorder=4)

    # ---------------------------------------------------------- 2. shared input
    M, MW = 27.6, 21.0
    label(ax, M, 85.0, "WHAT THE MODEL IS GIVEN", FS_COL, INK, weight="bold")
    ax.text(M, 81.0, "identical for both tasks", fontsize=FS_SMALL, color=MUTED, va="top")
    box(ax, M, 62.6, MW, 16.0, fc=WHITE, ec=LINE)
    label(ax, M + 1.3, 76.6, "Case summary", FS_SMALL, MUTED, weight="bold")
    wrap(ax, M + 1.3, 73.2, clip(d["diagnosis"], 122), 36, FS_SMALL, INK, lh=2.25,
         floor=63.7, what="case-summary card")
    box(ax, M, 44.4, MW, 16.0, fc=WHITE, ec=LINE)
    label(ax, M + 1.3, 58.9, f"{len(d['captions'])} slide captions", FS_SMALL, MUTED, weight="bold")
    wrap(ax, M + 1.3, 55.0, clip(d["captions"][1], 122), 36, FS_SMALL, INK, lh=2.25,
         floor=45.5, what="slide-caption card")
    ax.text(M + MW / 2, 41.6, "no reference answer, no transcript, no tools",
            fontsize=FS_SMALL - 0.3, color=MUTED, ha="center", va="top", style="italic")

    arrow(ax, L + LW + 0.9, 62.5, M - 1.0, 70.6, PURPLE)
    arrow(ax, L + LW + 0.9, 62.5, M - 1.0, 52.4, PURPLE)
    # ---------------------------------------------------------- 3. the two tasks
    R, RW = 53.6, 44.8
    arrow(ax, M + MW + 0.9, 70.6, R - 1.2, 76.0, MUTED)
    arrow(ax, M + MW + 0.9, 52.4, R - 1.2, 32.0, MUTED)

    # --- Task 2: expert QA -------------------------------------------------
    box(ax, R, 57.5, RW, 31.5, fc=WHITE, ec=BLUE, lw=0.9)
    box(ax, R, 83.7, RW, 5.3, fc=BLUE, ec=BLUE, lw=0.9)
    ax.text(R + 1.5, 86.35, "TASK 2   Expert QA", fontsize=FS_COL, color=WHITE,
            fontweight="bold", va="center", zorder=5)
    ax.text(R + RW - 1.5, 86.35, "answer as the named specialist", fontsize=FS_SMALL,
            color="#dbeaf5", ha="right", va="center", zorder=5)

    x = chip(ax, R + 1.5, 79.9, f"role: {qa['target_specialist_role']}", BLUE_LIGHT, BLUE)
    chip(ax, x + 1.4, 79.9, qa["qa_type"].replace("_", " "), "#eef4f9", BLUE_LIGHT)

    y = wrap(ax, R + 1.5, 76.6, "Q  " + clip(qa["question"], 132), 84, FS_BODY, INK,
             lh=2.5, floor=70.0, what="Task 2 question")
    ax.plot([R + 1.5, R + RW - 1.5], [y - 0.9, y - 0.9], color=GRID, lw=0.5, zorder=3)
    y = wrap(ax, R + 1.5, y - 2.6,
             "Reference  " + clip(qa["reference_answer"], 186), 92, FS_SMALL, MUTED,
             lh=2.35, floor=62.4, what="Task 2 reference answer")
    ax.text(R + 1.5, y - 1.0,
            "verbatim from the real expert  ·  scored by clinical equivalence",
            fontsize=FS_SMALL - 0.4, color=BLUE, va="top", style="italic", zorder=4)

    # --- Task 1: simulation ------------------------------------------------
    box(ax, R, 5.0, RW, 48.5, fc=WHITE, ec=GREEN, lw=0.9)
    box(ax, R, 48.2, RW, 5.3, fc=GREEN, ec=GREEN, lw=0.9)
    ax.text(R + 1.5, 50.85, "TASK 1   Tumor Board Simulation", fontsize=FS_COL, color=WHITE,
            fontweight="bold", va="center", zorder=5)
    ax.text(R + RW - 1.5, 50.85, "run the whole board", fontsize=FS_SMALL,
            color="#dff0dc", ha="right", va="center", zorder=5)

    ax.text(R + 1.5, 45.1, "Generate the multi-specialist discussion, then the plan",
            fontsize=FS_SMALL, color=MUTED, va="top", zorder=4)
    x, yy = R + 1.5, 40.4
    for role in d["role_sequence"]:
        if x + chip_width(role) > R + RW - 1.5:
            x, yy = R + 1.5, yy - 4.1
        x = chip(ax, x, yy, role, GREEN_LIGHT, GREEN) + 1.2
    ax.text(R + RW - 1.5, 45.1,
            f"real board: {len(d['role_sequence'])} roles over {d['n_turns']} turns",
            fontsize=FS_SMALL - 0.4, color=MUTED, ha="right", va="top", zorder=4)
    if yy < 31.0:
        refuse("the role chips wrapped further than the panel allows")

    box(ax, R + 1.5, 9.2, RW - 3.0, 23.6, fc="#f4faf3", ec=GREEN_LIGHT)
    label(ax, R + 2.8, 31.2, "Reference conclusion", FS_SMALL, GREEN, weight="bold")
    wrap(ax, R + 2.8, 27.8, clip(case["reference_conclusion"], 600), 88, FS_SMALL, INK,
         lh=2.3, floor=10.4, what="Task 1 reference conclusion")
    ax.text(R + 1.5, 6.6, "distilled from the real board's decision  ·  scored on role "
            "coverage, clinical accuracy and conclusion alignment",
            fontsize=FS_SMALL - 0.4, color=GREEN, va="bottom", style="italic", zorder=4)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("pdf", "svg", "png"):
        p = out_dir / f"{stem}.{suffix}"
        fig.savefig(p, dpi=DPI, facecolor="white")
        written.append(p)
    plt.close(fig)
    return written


def caption(d: dict) -> str:
    sc, qa = d["scale"], d["qa"]
    return (
        f"Overview of OpenTumorBoard. {sc['videos']} real multidisciplinary tumor board "
        f"recordings ({sc['hours']:.0f} hours) are processed into {sc['cases']} cases and "
        f"{sc['questions']:,} atomic questions. Both tasks receive the same input - a "
        "structured case summary and the case-aligned slide captions - and differ only in "
        "what they must produce. Task 2 (Expert QA) asks the model to answer, in the first "
        "person and as a named specialist, a question that actually arose in the meeting; "
        "the reference is the expert's verbatim utterance. Task 1 (Tumor Board Simulation) "
        "asks for the whole role-tagged discussion and the resulting treatment plan, scored "
        "against the real discussion and its distilled conclusion. The case shown is real "
        f"({d['n_turns']} turns, {len(d['role_sequence'])} specialist roles, "
        f"{d['n_qa_case']} questions); all text is read from the released manifests."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--stem", default="fig1_overview")
    ap.add_argument("--video-uid", default="video_cc76d189e9e7967f")
    ap.add_argument("--case-id", default="case_001")
    ap.add_argument("--qa-index", type=int, default=None)
    ap.add_argument("--qa-id", default=None)
    ap.add_argument("--font-family", default="DejaVu Sans")
    ap.add_argument("--font-dir", type=Path, default=None)
    a = ap.parse_args()

    font = resolve_font(a.font_family, a.font_dir)
    d = gather(a.manifest_dir, a.video_uid, a.case_id, a.qa_index, a.qa_id)
    written = draw(d, a.output_dir, a.stem, a.font_family)

    prov = {
        "figure": "1",
        "sources": d["sources"],
        "font": font,
        "example_case": {
            "video_uid": a.video_uid, "case_id": a.case_id,
            "minutes": round(d["minutes"], 1), "turns": d["n_turns"],
            "role_sequence": d["role_sequence"], "questions_in_case": d["n_qa_case"],
            "slide_captions": len(d["captions"]),
            "qa_id": d["qa"]["qa_id"], "qa_type": d["qa"]["qa_type"],
            "target_specialist_role": d["qa"]["target_specialist_role"],
        },
        "scale": d["scale"],
        "caption": caption(d),
        "outputs": [str(p) for p in written],
    }
    (a.output_dir / f"{a.stem}.provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": prov["outputs"], "example_case": prov["example_case"],
                      "caption": prov["caption"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
