#!/usr/bin/env python3
"""Figure 3 - expert QA (Task 2) by question type, per model.

Rows are the nine QA types, columns the board arms, cells the judge's mean clinical
equivalence (1-5). Everything is resolved from results_catalog.json at run time:
the arms come from active.models, and each arm's row is resolved through
aggregate_sources.judge under the catalog's own last-one-wins rule rather than by
naming a batch directory - batch names are not panels, and reading one as a panel is
how a superseded row gets published.

TWO THINGS THE FIGURE MUST NOT HIDE.

The closed-source block of the figure plan does not exist. No Claude, GPT, Gemini or
Grok row has ever been generated - the registry holds only open-weight vendors - so
the left block is drawn as an explicit gap rather than silently omitted, which would
let a reader take the remaining columns for the whole intended panel.

The columns do not share an input. Vision arms answer from the slide images and their
captions; text-only arms cannot take images and answer from captions alone. That is a
property of the models, not a defect, but it means a column-to-column difference
confounds model with input. Each column is therefore marked MM or cap, and the two
groups are never presented as a single ranking.

usage:
  python -m scripts.paper.figure3_task2_heatmap \
      --workspace .../OpenTumorBoard_workspace --output-dir .../figures/fig3
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import math
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

FIG_W_IN = 7.4
DPI = 600

FS_TITLE = 11.0
FS_BLOCK = 6.6
FS_COL = 5.9
FS_ROW = 6.4
FS_CELL = 5.5
FS_NOTE = 5.4

INK = "#1b1b1b"
MUTED = "#646464"
LINE = "#8c8c8c"
GAP = "#ededed"

# One continuous map for the whole grid. Colour encodes the score and nothing else;
# the model category is carried by the black group boxes instead. Encoding category
# in hue AND score in lightness made a cell's colour mean two things at once.
#
# ColorBrewer Blues, light (low) to dark (high), single hue and low saturation. The
# bottom of the ramp is truncated: unmodified Blues starts at #f7fbff, which against
# white gridlines reads as an empty cell rather than a weak score, and the weakest
# arm here is a result, not a gap.
CMAP = LinearSegmentedColormap.from_list(
    "blues_trunc", plt.get_cmap("Blues")(np.linspace(0.12, 1.0, 256)))

SHORT = {
    "deepseek_v4_pro": "DeepSeek-V4-Pro",
    "deepseek_v4_flash_api": "DeepSeek-V4-Flash",
    "qwen3_8_27b": "Qwen3.8-27B",
    "medgemma_27b_it": "MedGemma-27B",
    "huatuogpt_3_32b": "HuatuoGPT-3-32B",
    "meditron3_70b": "Meditron-3-70B",
    "medreason_8b": "MedReason-8B",
    "llama4_scout": "Llama 4 Scout",
    "gemma4_31b_it": "Gemma 4 31B",
    "ministral_3_14b_instruct_2512": "Ministral 3 14B",
    "nemotron_3_5_lightning": "Nemotron 3.5",
}

BLOCKS = [
    ("Closed-source frontier", []),
    ("Open-source frontier", ["deepseek_v4_pro", "deepseek_v4_flash_api", "qwen3_8_27b"]),
    ("Medical-purpose", ["medgemma_27b_it", "huatuogpt_3_32b", "meditron3_70b", "medreason_8b"]),
    ("Other open general", ["llama4_scout", "gemma4_31b_it", "ministral_3_14b_instruct_2512",
                            "nemotron_3_5_lightning"]),
]


def refuse(msg: str) -> None:
    raise SystemExit(f"figure3: {msg}")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gather(ws: Path) -> dict:
    cat_path = ws / "OpenTumorBoard/evaluation/results_catalog.json"
    reg_path = ws / "OpenTumorBoard/evaluation/model_registry.json"
    if not cat_path.exists():
        refuse(f"missing {cat_path}")
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    reg = json.loads(reg_path.read_text(encoding="utf-8"))["models"]

    # The catalog's own precedence rule, applied rather than restated.
    if "Later aggregate sources override earlier" not in cat["policy"]["source_priority"]:
        refuse("policy.source_priority is no longer last-one-wins; the resolution "
               "below encodes that rule and must be revisited")
    resolved, source_of = {}, {}
    for rel in cat["active"]["aggregate_sources"]["judge"]:
        p = ws / "model_evaluation" / rel
        if not p.exists():
            refuse(f"aggregate source missing on disk: {rel}")
        d = json.loads(p.read_text(encoding="utf-8"))
        for r in d.get("results", {}).get("task2", []) or []:
            resolved[r["model_key"]] = r
            source_of[r["model_key"]] = (rel, d.get("protocol_version", ""))

    arms = []
    for m in cat["active"]["models"]:
        if "task2" not in m["runs"]:
            continue
        key, cond = m["model_key"], m["runs"]["task2"]["condition"]
        # The condition decides the key, not whichever spelling happens to resolve
        # first: a multimodal arm publishes as mm_<key>, and several bare keys still
        # resolve to superseded v3 batches that are not board rows.
        want = ("mm_" + key) if cond == "multimodal" else key
        if want not in resolved:
            refuse(f"active arm {key} ({cond}) expects judged key {want}, which no "
                   "aggregate judge source provides")
        jk = want
        rel, proto = source_of[jk]
        if "task2_v4" not in proto:
            refuse(f"{key} resolves to {proto}, not task2_v4; the column would mix rubrics")
        row = resolved[jk]
        if not row.get("per_qa_type"):
            refuse(f"{key} has no per_qa_type; there is nothing to put in the rows")
        if SHORT.get(key.replace("_reasoning", "")) is None:
            refuse(f"no short column label for {key}; add one to SHORT rather than "
                   "letting a 30-character registry name run off the panel")
        arms.append({
            "model_key": key, "judged_key": jk, "condition": cond,
            "display": reg.get(key, {}).get("display_name", key),
            "short": SHORT.get(key.replace("_reasoning", "")),
            "overall": row["item_macro_clinical_equivalence"],
            "scored": row["scored_records"],
            "per_qa_type": {t: v["clinical_equivalence"] for t, v in row["per_qa_type"].items()},
            "judge_source": rel,
        })

    # Row order = corpus frequency, so Fig 3 and Fig 2b tell the same story in the
    # same order rather than two orderings of one distribution.
    man = ws / ("OpenTumorBoard_data/benchmark/task2/test/expert_qa_test_video_split_"
                "60_10_30_v3_qascreened_xsfixed_reanchored_rescreened_slidealigned_20260814.jsonl")
    counts = collections.Counter(
        json.loads(l)["qa_type"] for l in man.read_text(encoding="utf-8").splitlines() if l.strip())
    types = [t for t, _ in counts.most_common()]
    for a in arms:
        if set(a["per_qa_type"]) != set(types):
            refuse(f"{a['model_key']} covers {len(a['per_qa_type'])} QA types, manifest has {len(types)}")

    judges = {json.loads((ws / "model_evaluation" / a["judge_source"]).read_text(
        encoding="utf-8")).get("judge_model") for a in arms}
    if len(judges) != 1:
        refuse(f"columns were scored by different judges: {judges}")

    return {"arms": arms, "types": types, "type_counts": counts, "judge": judges.pop(),
            "catalog_sha256": sha256(cat_path),
            "n_records": {a["scored"] for a in arms}}


def resolve_font(family: str, font_dir: Path | None) -> dict:
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
               + (f" in {font_dir}" if font_dir else " (no --font-dir)"))
    if fm.FontProperties(fname=found).get_name().lower() != family.lower():
        refuse(f"font '{family}' resolved to something else ({found}) - refusing")
    return {"family": family, "file": found, "registered": registered}


def pretty(t: str) -> str:
    return t.replace("_", " ").capitalize()


def order(d: dict) -> list[dict]:
    """Block order from the figure plan; within a block, strongest arm first."""
    by_key = collections.defaultdict(list)
    for a in d["arms"]:
        by_key[a["model_key"].replace("_reasoning", "")].append(a)
    cols, placed = [], set()
    for title, keys in BLOCKS:
        group = []
        for k in keys:
            for a in sorted(by_key.get(k, []), key=lambda x: -x["overall"]):
                group.append(a)
                placed.add(a["model_key"])
        cols.append((title, group))
    left = [a for a in d["arms"] if a["model_key"] not in placed]
    if left:
        refuse(f"{len(left)} arms are in no block: {[a['model_key'] for a in left]}. "
               "Every board arm must be placed, or the figure silently drops a model.")
    return cols


def _ink_on(rgba) -> str:
    """Cell text must survive both ends of the ramp: near-white at the low end and
    near-black at the high. Pick by relative luminance, not by a value threshold, so
    the choice stays correct if the colour map is swapped or reversed."""
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if lum < 0.50 else "#101010"


def draw(d: dict, out_dir: Path, stem: str, family: str) -> list[Path]:
    """Question types on rows, model arms on columns, groups drawn as black boxes.

    This is the orientation the figure plan asks for. It also puts the nine types on
    the axis a reader scans first, which is what the panel is about: the spread across
    question types is larger than the spread across most models.
    """
    plt.rcParams.update({"font.family": family, "text.color": INK,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    blocks = [(t, g) for t, g in order(d) if g]
    types = d["types"]
    cols = [(t, a) for t, g in blocks for a in g]
    nrow, ncol = len(types), len(cols)

    GAP = 0.34                       # gap above the all-questions row
    height_units = nrow + GAP + 1
    cell_w, cell_h = 0.385, 0.335
    label_in, right_in = 1.52, 0.92
    grid_w, grid_h = ncol * cell_w, height_units * cell_h
    longest = max(len(a["short"]) + (3 if a["model_key"].endswith("_reasoning") else 0)
                  for _, a in cols)
    rise_in = longest * FS_COL * 0.52 * math.sin(math.radians(45)) / 72.0
    fig_w = label_in + grid_w + right_in
    fig_h = 0.72 + grid_h + rise_in + 0.26

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([label_in / fig_w, (rise_in + 0.22) / fig_h,
                       grid_w / fig_w, grid_h / fig_h])
    ax.set_xlim(0, ncol); ax.set_ylim(height_units, 0); ax.axis("off")

    vals = [a["per_qa_type"][t] for _, a in cols for t in types]
    vals += [a["overall"] for _, a in cols]
    norm = Normalize(vmin=min(vals), vmax=max(vals))

    for c, (_, a) in enumerate(cols):
        for r, t in enumerate(types):
            v = a["per_qa_type"][t]
            col = CMAP(norm(v))
            ax.add_patch(Rectangle((c, r), 1, 1, fc=col, ec="white", lw=0.4, zorder=2))
            ax.text(c + 0.5, r + 0.5, f"{v:.2f}", fontsize=FS_CELL, ha="center",
                    va="center", color=_ink_on(col), zorder=3)
        col = CMAP(norm(a["overall"]))
        ax.add_patch(Rectangle((c, nrow + GAP), 1, 1, fc=col, ec="white", lw=0.4, zorder=2))
        ax.text(c + 0.5, nrow + GAP + 0.5, f"{a['overall']:.2f}", fontsize=FS_CELL,
                ha="center", va="center", color=_ink_on(col), fontweight="bold", zorder=3)
        lab = a["short"] + ("  \u00b7R" if a["model_key"].endswith("_reasoning") else "")
        ax.text(c + 0.5, height_units + 0.30, lab, fontsize=FS_COL, rotation=45,
                ha="right", va="top", rotation_mode="anchor", clip_on=False)
        if a["condition"] == "multimodal":
            ax.plot([c + 0.5], [height_units + 0.12], marker="o", ms=2.4, mfc=INK,
                    mec="none", zorder=4, clip_on=False)

    left = 0
    for title, group in blocks:
        ax.add_patch(Rectangle((left, 0), len(group), nrow, fill=False, ec="black",
                               lw=1.15, zorder=5))
        ax.add_patch(Rectangle((left, nrow + GAP), len(group), 1, fill=False, ec="black",
                               lw=1.15, zorder=5))
        ax.text(left + len(group) / 2, -0.30, title, fontsize=FS_BLOCK, color=INK,
                ha="center", va="bottom", fontweight="bold", clip_on=False)
        left += len(group)

    for r, t in enumerate(types):
        ax.text(-0.22, r + 0.5, pretty(t), fontsize=FS_ROW, ha="right", va="center")
    ax.text(-0.22, nrow + GAP + 0.5, "All questions", fontsize=FS_ROW, ha="right",
            va="center", fontweight="bold")

    cax = fig.add_axes([(label_in + grid_w + 0.16) / fig_w, (rise_in + 0.22) / fig_h,
                        0.15 / fig_w, grid_h / fig_h])
    cax.imshow([[i] for i in range(255, -1, -1)], aspect="auto", cmap=CMAP,
               extent=(0, 1, norm.vmin, norm.vmax))
    cax.set_xticks([]); cax.yaxis.tick_right(); cax.yaxis.set_label_position("right")
    cax.tick_params(labelsize=FS_NOTE, length=2, pad=1.5)
    cax.set_ylabel("Clinical equivalence  (1\u20135)", fontsize=FS_NOTE + 0.3, labelpad=3)
    for sp in cax.spines.values():
        sp.set_linewidth(0.5); sp.set_color(INK)

    fig.suptitle("Expert QA by question type", fontsize=FS_TITLE, fontweight="bold",
                 x=(label_in + grid_w / 2) / fig_w, y=1 - 0.10 / fig_h, ha="center",
                 va="top")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("pdf", "svg", "png"):
        q = out_dir / f"{stem}.{suffix}"
        fig.savefig(q, dpi=DPI, facecolor="white", bbox_inches="tight")
        written.append(q)
    plt.close(fig)
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--stem", default="fig3_task2_heatmap")
    ap.add_argument("--font-family", default="DejaVu Sans")
    ap.add_argument("--font-dir", type=Path, default=None)
    a = ap.parse_args()

    font = resolve_font(a.font_family, a.font_dir)
    d = gather(a.workspace)
    written = draw(d, a.output_dir, a.stem, a.font_family)

    prov = {
        "figure": "3",
        "task": "task2",
        "metric": "llm_judge clinical_equivalence (1-5), task2_judge_v4",
        "judge_model": d["judge"],
        "records_per_arm": sorted(d["n_records"]),
        "results_catalog_sha256": d["catalog_sha256"],
        "row_order": "QA-type frequency in the test manifest (same order as Fig 2b)",
        "qa_type_counts": dict(d["type_counts"].most_common()),
        "arms": [{k: v for k, v in a_.items() if k != "per_qa_type"} for a_ in d["arms"]],
        "matrix": {a_["model_key"]: a_["per_qa_type"] for a_ in d["arms"]},
        "absent": {
            "closed_source_frontier": "No Claude, GPT, Gemini or Grok row exists. The model "
                                      "registry holds only open-weight vendors; the block is "
                                      "drawn as an explicit gap.",
        },
        "caption": (
            "Expert QA (Task 2) by question type. Cells are the LLM judge's mean clinical "
            "equivalence on a 1-5 scale over all {n:,} test questions, under task2_judge_v4 "
            "with {j} as judge; the colour scale is shared across all three panels. Rows are "
            "ordered by question-type frequency in the corpus. A blue column label marks an "
            "arm answering from the slide images and their captions; a black label marks a "
            "text-only arm answering from captions alone, so columns do not share an input "
            "and MM and caption-only arms are not one ranking. '.R' denotes reasoning mode. "
            "Closed-source frontier models have not been run."
        ).format(n=max(d["n_records"]), j=d["judge"]),
        "caption_note": "The figure carries no legend text; this caption holds what the "
                        "panels no longer state on the canvas, and must be used with them.",
        "font": font,
        "outputs": [str(p) for p in written],
    }
    (a.output_dir / f"{a.stem}.provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": prov["outputs"], "arms": len(d["arms"]),
                      "types": len(d["types"]), "judge": d["judge"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
