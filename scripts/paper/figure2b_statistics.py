#!/usr/bin/env python3
"""Figure 2b - benchmark statistics, computed from the manifests.

Three distributions and a scale strip. Everything is derived from the manifests at
run time; nothing is typed in. A statistics panel that disagrees with the data it
describes is worse than no panel, and the only way to keep them in step is to make
the figure a function of the manifests.

THE CANCER-TYPE PANEL. The dataset has no categorical site field - `case_summary`
carries a free-text `[ diagnosis ]:` section and nothing else - so the label is
derived, not read. `cancer_sites.py` strips the clauses describing where disease
SPREAD and then applies ordered regexes to what is left; without the stripping the
site of metastasis outranks the site of origin ('small cell lung carcinoma with brain
metastases' lands in CNS). It refuses rather than bucketing an unmatched string, so
the panel cannot quietly drift into a fiction. It is a keyword mapping over clinical
prose, not a clinician's label: the caption must say so, and the per-case assignments
ship in the provenance file for audit.

THE ASSERTIONS. The headline 16,215 is the union of two manifests, and it is only a
real total if they are disjoint. The split is video-level, so both the qa_id sets and
the video sets must not intersect; an overlap would mean the same question counted
twice and the panel silently inflated. The QA videos must also be exactly the Task 1
videos, or the two halves of this figure describe different corpora.

usage:
  python -m scripts.paper.figure2b_statistics \
      --manifest-dir .../model_evaluation/manifests \
      --output-dir .../figures/fig2b
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

from scripts.paper import cancer_sites  # noqa: E402

matplotlib.use("Agg")
from matplotlib import font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from scripts.paper import style  # noqa: E402

FIG_W_IN = style.A4_W
FIG_H_IN = 0.80 * style.A4_H
DPI = 600

FS_TICK = 10.0
FS_TITLE = 14.0
FS_PANEL = 11.0
FS_VALUE = 8.0
FS_SCALE = 9.0

INK = "#1b1b1b"
GRID = "#d8d8d8"
MUTED = "#5c5c5c"

# ColorBrewer Paired。三个面板讲的是同一个数据集的三种切法,所以同一个蓝;
# 中位数线用红,是唯一带语义的颜色(标记位置,不是"好坏")。
BAR = "#1f78b4"
BAR_SOFT = "#a6cee3"
ACCENT = "#e31a1c"

TRAINVAL = "expert_qa_trainval_video_split_60_10_30_v4_qascreened_xsfixed_20260809.jsonl"
TEST = ("expert_qa_test_video_split_60_10_30_v3_qascreened_xsfixed_reanchored_"
        "rescreened_slidealigned_20260814.jsonl")
TASK1 = ("tumor_board_simulation_full_video_split_60_10_30_v4_deleaked_xsfixed_"
         "reanchored_20260808.jsonl")

BIN = 10
BIN_MAX = 90


def refuse(message: str) -> None:
    raise SystemExit(f"figure2b: {message}")


def load(path: Path) -> list[dict]:
    if not path.exists():
        refuse(f"missing manifest {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pretty(name: str) -> str:
    return name.replace("_", " ").capitalize()


def gather(manifest_dir: Path) -> dict:
    trainval = load(manifest_dir / TRAINVAL)
    test = load(manifest_dir / TEST)
    cases = load(manifest_dir / TASK1)

    ids_tv = {r["qa_id"] for r in trainval}
    ids_te = {r["qa_id"] for r in test}
    if len(ids_tv) != len(trainval) or len(ids_te) != len(test):
        refuse("a QA manifest contains duplicate qa_id values")
    if ids_tv & ids_te:
        refuse(
            f"{len(ids_tv & ids_te)} qa_id values appear in BOTH QA manifests. The headline "
            "total is their union, so an overlap would count those questions twice."
        )
    videos_tv = {r["video_uid"] for r in trainval}
    videos_te = {r["video_uid"] for r in test}
    if videos_tv & videos_te:
        refuse(
            f"{len(videos_tv & videos_te)} videos appear in both QA splits. The split is "
            "video-level by construction; an overlap means the split is broken, not that "
            "this figure needs a different denominator."
        )

    qa = trainval + test
    videos_qa = videos_tv | videos_te
    videos_cases = {r["video_uid"] for r in cases}
    if videos_qa != videos_cases:
        refuse(
            f"QA covers {len(videos_qa)} videos and Task 1 covers {len(videos_cases)}; "
            f"symmetric difference {len(videos_qa ^ videos_cases)}. The two halves of this "
            "figure would describe different corpora."
        )

    qa_types = collections.Counter(r["qa_type"] for r in qa)
    roles = collections.Counter(r["target_specialist_role"] for r in qa)
    for name, counter in (("qa_type", qa_types), ("role", roles)):
        if sum(counter.values()) != len(qa):
            refuse(f"{name} counts do not sum to {len(qa)}")
        if any(not k for k in counter):
            refuse(f"{name} has an empty label; it would be plotted as a blank bar")

    sites, site_audit = cancer_sites.site_counts(
        [({"video_uid": r["video_uid"], "case_id": r["case_id"]}, r["case_summary"]) for r in cases])
    if sum(sites.values()) != len(cases):
        refuse(f"site counts do not sum to {len(cases)} cases")

    turns = [len(r["reference_discussion"]) for r in cases]
    roles_per_case = [len({u["inferred_role"] for u in r["reference_discussion"]}) for r in cases]
    slides = [str(r.get("slides") or "").count("<image>") for r in cases]
    durations = [r["case_end_sec"] - r["case_start_sec"] for r in cases]

    return {
        "qa": qa,
        "qa_types": qa_types,
        "roles": roles,
        "sites": sites,
        "site_audit": site_audit,
        "turns": turns,
        "roles_per_case": roles_per_case,
        "scale": {
            "videos": len(videos_qa),
            "cases": len(cases),
            "questions": len(qa),
            "qa_types": len(qa_types),
            "roles": len(roles),
            "sites": len(sites),
            "slides": sum(slides),
            "slides_median": statistics.median(slides),
            "slides_max": max(slides),
            "hours": sum(durations) / 3600.0,
            "case_minutes_median": statistics.median(durations) / 60.0,
            "utterances": sum(turns),
            "turns_median": statistics.median(turns),
            "turns_mean": statistics.mean(turns),
            "turns_min": min(turns),
            "turns_max": max(turns),
            "roles_per_case_median": statistics.median(roles_per_case),
        },
        "sources": {
            name: {"path": str((manifest_dir / name).resolve()),
                   "sha256": sha256(manifest_dir / name)}
            for name in (TRAINVAL, TEST, TASK1)
        },
    }


def resolve_font(family: str, font_dir: Path | None) -> dict:
    """matplotlib falls back silently on a missing family; a paper figure must not."""
    registered = []
    if font_dir is not None:
        if not font_dir.is_dir():
            refuse(f"--font-dir {font_dir} is not a directory")
        for path in sorted(font_dir.glob("*")):
            if path.suffix.lower() in (".ttf", ".otf", ".ttc"):
                fm.fontManager.addfont(str(path))
                registered.append(path.name)
    try:
        found = fm.findfont(fm.FontProperties(family=family), fallback_to_default=False)
    except ValueError:
        refuse(
            f"font '{family}' is not installed and no file for it was found"
            + (f" in {font_dir}" if font_dir else " (no --font-dir given)")
            + f". Registered this run: {registered or 'none'}. Regular AND bold are both needed."
        )
    resolved = fm.FontProperties(fname=found).get_name()
    if resolved.lower() != family.lower():
        refuse(f"font '{family}' resolved to '{resolved}' ({found}) - wrong typeface, refusing")
    return {"family": family, "file": found, "registered": registered}


def hbar_panel(ax, counter: collections.Counter, total: int, title: str,
               two_col: bool = False, prettify: bool = True) -> None:
    """two_col spreads a long tail across the full width so 17 sites stay legible."""
    items = counter.most_common()
    labels = [pretty(k) if prettify else k for k, _ in items]
    values = [v for _, v in items]
    ys = list(range(len(items)))
    ax.barh(ys, values, height=0.72, color=BAR, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.invert_yaxis()
    span = max(values)
    for y, v in zip(ys, values):
        ax.text(v + span * 0.02, y, f"{v:,}  ({100 * v / total:.1f}%)",
                va="center", ha="left", fontsize=FS_VALUE, color=MUTED)
    # 数值标签默认不裁剪,超出坐标轴就被画布边缘切掉,所以要靠 xlim 把它留在轴内。
    ax.set_xlim(0, span * (1.30 if two_col else 1.88))
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=FS_PANEL, fontweight="bold", loc="left", pad=6)


def hist_panel(ax, turns: list[int], scale: dict) -> None:
    counts = collections.Counter(min(t // BIN * BIN, BIN_MAX) for t in turns)
    edges = list(range(0, BIN_MAX + 1, BIN))
    values = [counts.get(e, 0) for e in edges]
    ax.bar(edges, values, width=BIN * 0.86, align="edge", color=BAR_SOFT,
           edgecolor=BAR, linewidth=0.7, zorder=3)
    median = scale["turns_median"]
    ax.axvline(median, color=ACCENT, linewidth=1.4, linestyle="--", zorder=4)
    ax.text(median + 1.5, max(values) * 0.94, f"median {median:.0f}",
            fontsize=FS_VALUE, color=ACCENT, va="top", ha="left")
    ax.set_xticks(edges)
    ax.set_xticklabels([str(e) for e in edges[:-1]] + [f"{BIN_MAX}+"], fontsize=FS_TICK)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlabel("utterances per case", fontsize=FS_TICK)
    ax.set_ylabel("cases", fontsize=FS_TICK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(
        f"Discussion length  ·  {scale['cases']} cases, {scale['utterances']:,} utterances",
        fontsize=FS_PANEL, fontweight="bold", loc="left", pad=6)
    # 多学科不是断言,是可以量的:每个 case 出现过几个不同专科角色。
    ax.text(0.985, 0.94,
            f"median {scale['roles_per_case_median']:.0f} distinct specialist roles per case",
            transform=ax.transAxes, ha="right", va="top", fontsize=FS_VALUE, color=MUTED)


def draw(data: dict, out_dir: Path, stem: str, family: str) -> list[Path]:
    style.apply()
    plt.rcParams.update({"text.color": INK, "axes.edgecolor": INK,
                         "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK})
    scale = data["scale"]
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.85, 0.80],
                          left=0.278, right=0.985, top=0.935, bottom=0.105,
                          wspace=0.80, hspace=0.34)

    hbar_panel(fig.add_subplot(gs[0, 0]), data["qa_types"], scale["questions"],
               f"Question types  \u00b7  {scale['qa_types']}")
    hbar_panel(fig.add_subplot(gs[0, 1]), data["roles"], scale["questions"],
               f"Target specialist  \u00b7  {scale['roles']}")
    hbar_panel(fig.add_subplot(gs[1, :]), data["sites"], scale["cases"],
               f"Primary cancer site  \u00b7  {scale['sites']}  (cases, derived)",
               two_col=True, prettify=False)
    hist_panel(fig.add_subplot(gs[2, :]), data["turns"], scale)

    fig.suptitle("OpenTumorBoard statistics", fontsize=FS_TITLE, fontweight="bold",
                 x=0.012, y=0.985, ha="left")
    fig.text(0.012, 0.028,
             f"{scale['videos']} videos  ·  {scale['hours']:.1f} hours  ·  "
             f"{scale['cases']} cases  ·  {scale['slides']:,} slides "
             f"(median {scale['slides_median']:.0f}/case, max {scale['slides_max']})  ·  "
             f"{scale['questions']:,} questions",
             fontsize=FS_SCALE, color=MUTED, ha="left", va="bottom")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("pdf", "svg", "png"):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=DPI, facecolor="white")
        written.append(path)
    plt.close(fig)
    return written


def caption(scale: dict) -> str:
    return (
        f"Benchmark statistics. {scale['videos']} tumor-board recordings "
        f"({scale['hours']:.1f} hours) yield {scale['cases']} cases and "
        f"{scale['questions']:,} atomic questions across {scale['qa_types']} question types and "
        f"{scale['roles']} specialist roles. Question and role distributions are over all "
        f"{scale['questions']:,} questions (train, validation and test; the splits are "
        "video-level and disjoint). Discussion length is the number of transcribed utterances "
        f"per case (median {scale['turns_median']:.0f}, mean {scale['turns_mean']:.1f}, range "
        f"{scale['turns_min']}-{scale['turns_max']}); the final bin is {BIN_MAX}+. Primary cancer "
        f"site spans {scale['sites']} categories and is counted over cases, not questions; the "
        "dataset carries no categorical site field, so the label is derived from the free-text "
        "diagnosis by an ordered keyword mapping that first removes clauses describing "
        "metastatic spread, and every per-case assignment is released with the figure."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--stem", default="fig2b_statistics")
    ap.add_argument("--font-family", default="DejaVu Sans")
    ap.add_argument("--font-dir", type=Path, default=None)
    args = ap.parse_args()

    font = resolve_font(args.font_family, args.font_dir)
    data = gather(args.manifest_dir)
    written = draw(data, args.output_dir, args.stem, args.font_family)

    provenance = {
        "figure": "2b",
        "sources": data["sources"],
        "font": font,
        "scale": data["scale"],
        "qa_types": dict(data["qa_types"].most_common()),
        "roles": dict(data["roles"].most_common()),
        "sites": dict(data["sites"].most_common()),
        "sites_method": {
            "source_field": "case_summary [ diagnosis ] free text",
            "method": "ordered keyword rules over the diagnosis with metastasis/extension "
                      "clauses removed first, so the site of origin is counted rather than "
                      "the site of spread; see scripts/paper/cancer_sites.py",
            "unmatched": 0,
            "caveat": "derived by keyword mapping, not labelled by a clinician; cases with "
                      "synchronous multiple primaries receive a single first-match label",
        },
        "caption": caption(data["scale"]),
        "outputs": [str(p) for p in written],
    }
    with (args.output_dir / f"{args.stem}.cancer_sites.jsonl").open("w", encoding="utf-8") as fh:
        for row in data["site_audit"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    (args.output_dir / f"{args.stem}.provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": [str(p) for p in written],
                      "scale": data["scale"],
                      "caption": provenance["caption"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
