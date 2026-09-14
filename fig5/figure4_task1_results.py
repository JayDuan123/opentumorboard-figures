#!/usr/bin/env python3
"""Figure 4 - tumor board simulation (Task 1).

Three panels, written as three files so they can be placed independently:

  fig4a  conclusion alignment per arm, grouped by input condition
  fig4b  output tokens against conclusion alignment
  fig4c  ROUGE-L and BERTScore, for the appendix

THE ARMS ARE NOT ONE LEAGUE TABLE. Seven arms answer from the slide images and
their captions; ten are text-only and can take captions alone, which the catalog
files as `ablation_caption_only` - the reduced half of Task 1, not a second way of
running it. Its own note says the sound comparison is each vision-capable model
against its own caption-only score, not across the two conditions. So the bars are
drawn in two groups with a rule between them and are never sorted into one ranking;
a reader who wants a single winner has to decide which condition they mean first.

THE SCORE IS `dimensions_all_responses`, NOT `dimensions`. That is the leaderboard's
own convention and the reason is in server/benchmark_results.py: the scored-only mean
grades a model on the cases it chose to answer, which lets an arm that concluded 152
of 184 outrank one that concluded all 184. Cases with no extractable conclusion enter
at the rubric's floor of 1. It moves exactly one arm - medreason_8b, 2.020 -> 1.842,
from 12th to 15th - and that arm is the reason the rule exists.

THERE IS NO COST PANEL. The plan asks for tokens-vs-accuracy and cost-vs-accuracy as
two plots. Every run is local vLLM against open weights and nothing in the workspace
carries a price; the generation-cost report states `cost_axis: output_tokens.mean`.
The two plots would be one plot with a relabelled axis, so only the token one is
drawn. A real cost axis needs the API arms that do not exist yet.

usage:
  python -m scripts.paper.figure4_task1_results \
      --workspace .../OpenTumorBoard_workspace --output-dir .../figures/fig4
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from scripts.paper import style  # noqa: E402

DPI = 600
FS_TITLE = 11.0
FS_LABEL = 6.4
FS_TICK = 6.2
FS_NOTE = 5.6
FS_VALUE = 5.8

INK = "#1b1b1b"
MUTED = "#646464"
LINE = "#8c8c8c"
GRID = "#dcdcdc"

CLOSED_C, OPEN_C = "#1f4e79", "#4a98cf"    # closed frontier vs open weights
MM_C, CAP_C = "#1f78b4", "#a6cee3"         # vision-capable vs caption-only
# The board's own consolidated export (2026-09-02). It resolves the catalog's
# last-wins rule once and records, per row, the judge batch and metric file each
# number came from - so this figure no longer re-implements that resolution and
# cannot drift from the board by getting it subtly wrong.
BOARD_CSV = "model_evaluation/board_results_20260903/board_results.csv"

# `conclusion_alignment` is the published column: format failures stay in the
# denominator at the rubric floor of 1. `_scored_only` averages the judged responses
# alone and is explicitly NOT what the board shows.
SCORE_COL = "conclusion_alignment"

CLOSED = {"claude_opus_5", "gemini_3_7_flash", "grok_4_6", "gpt_5_6_sol"}

SHORT = {
    "deepseek_v4_pro": "DeepSeek-V4-Pro", "deepseek_v4_flash_api": "DeepSeek-V4-Flash",
    "medgemma_27b_it": "MedGemma-27B", "huatuogpt_3_32b": "HuatuoGPT-3-32B",
    "meditron3_70b": "Meditron-3-70B", "medreason_8b": "MedReason-8B",
    "llama4_scout": "Llama 4 Scout", "gemma4_31b_it": "Gemma 4 31B",
    "ministral_3_14b_instruct_2512": "Ministral 3 14B",
    "nemotron_3_5_lightning": "Nemotron 3.5",
    "claude_opus_5": "Claude Opus 5", "gemini_3_7_flash": "Gemini 3.7 Flash",
    "qwen3_8_max": "Qwen3.8-Max", "grok_4_6": "Grok 4.6", "gpt_5_6_sol": "GPT-5.6 Sol",
}


def refuse(msg: str) -> None:
    raise SystemExit(f"figure4: {msg}")


def label_of(key: str) -> str:
    base = key.replace("_reasoning", "")
    if base not in SHORT:
        refuse(f"no short label for {key}; add it to SHORT")
    return SHORT[base] + ("  ·R" if key.endswith("_reasoning") else "")


def gather(ws: Path) -> dict:
    csv_p = ws / BOARD_CSV
    if not csv_p.exists():
        refuse(f"missing board export: {csv_p}")
    with csv_p.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["code_task"] == "task1"]
    if not rows:
        refuse("the board export has no task1 rows")

    need = (SCORE_COL, "mean_output_tokens", "rouge_l_f1", "bertscore_f1",
            "condition", "model_key", "scored_records", "test_records", "judge_batch")
    for col in need:
        if col not in rows[0]:
            refuse(f"the board export has no {col!r} column")

    batches = {r["judge_batch"] for r in rows}
    off = sorted(b for b in batches if "task1_judge_v31" not in b)
    if off:
        refuse(f"these judge batches are not task1_judge_v31: {off}. Mixing rubrics "
               "across the bars would compare scores that are not on one scale.")

    # Drop Nemotron 3.5 from every fig4 panel — the non-reasoning run emits ~13k
    # tokens per case and often never reaches a conclusion (see paper §5.2), so
    # both variants sit as outliers that distort the correlation and the ranking.
    DROP = {"nemotron_3_5_lightning"}
    rows = [r for r in rows if r["model_key"].replace("_reasoning", "") not in DROP]
    arms = []
    for r in rows:
        key = r["model_key"]
        base = key.replace("_reasoning", "")
        if base not in SHORT:
            refuse(f"no short label for {key}; add it to SHORT rather than letting a "
                   "long registry name run off the panel")
        for col in (SCORE_COL, "mean_output_tokens", "rouge_l_f1", "bertscore_f1"):
            if not r[col].strip():
                refuse(f"{key} has an empty {col}; the panels would cover different arms")
        arms.append({
            "key": key,
            "label": SHORT[base] + ("  \u00b7R" if key.endswith("_reasoning") else ""),
            "display": r["display_name"],
            "condition": r["condition"],
            "closed": key in CLOSED,
            "score": float(r[SCORE_COL]),
            "score_scored_only": float(r["conclusion_alignment_scored_only"] or "nan"),
            "scored": int(r["scored_records"]), "test": int(r["test_records"]),
            "format_failures": int(r["format_failures"] or 0),
            "tokens": float(r["mean_output_tokens"]),
            "rouge_l_f1": float(r["rouge_l_f1"]),
            "bertscore_f1": float(r["bertscore_f1"]),
            "judge_batch": r["judge_batch"], "run": r["run"],
        })

    groups = [("Slides + captions", [a for a in arms if a["condition"] == "multimodal"]),
              ("Captions only", [a for a in arms if a["condition"] != "multimodal"])]
    for name, g in groups:
        if not g:
            refuse(f"group {name!r} is empty")
        g.sort(key=lambda a: a["score"])

    tests = {a["test"] for a in arms}
    if len(tests) != 1:
        refuse(f"arms were scored over different case counts: {sorted(tests)}")

    judge_models = {r.get("judge_model", "").strip() for r in rows}
    judge_models.discard("")
    if len(judge_models) != 1:
        refuse(f"task1 rows list multiple judges: {sorted(judge_models)}")
    judge = judge_models.pop()
    return {"arms": arms, "groups": groups, "n_cases": tests.pop(),
            "protocol": "task1_judge_v31", "judge": judge,
            "judge_batches": sorted(batches),
            "board_export": str(csv_p.name),
            "board_sha256": hashlib.sha256(csv_p.read_bytes()).hexdigest()}


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
        refuse(f"font '{family}' not installed and no file found")
    if fm.FontProperties(fname=found).get_name().lower() != family.lower():
        refuse(f"font '{family}' resolved to something else ({found})")
    return {"family": family, "file": found, "registered": registered}


def save(fig, out_dir: Path, stem: str, tight: bool = True) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("pdf", "svg", "png"):
        p = out_dir / f"{stem}.{suffix}"
        # bbox_inches="tight" crops to the ink, so it cannot coexist with an exact
        # page size; a panel that wants A4 has to give it up.
        fig.savefig(p, dpi=DPI, facecolor="white",
                    **({"bbox_inches": "tight"} if tight else {}))
        written.append(p)
    plt.close(fig)
    return written


def panel_a(d: dict, out_dir: Path, stem: str) -> list[Path]:
    """All 20 arms as a single ranking; per-model colour, hatched fill = closed."""
    PALETTE = {
        "gemini_3_7_flash":              "#4a7cbe",
        "grok_4_6":                      "#5f4b8b",
        "qwen3_8_max":                   "#c76a3d",
        "gpt_5_6_sol":                   "#2f7a4c",
        "claude_opus_5":                 "#c98847",
        "deepseek_v4_pro":               "#5b6ea8",
        "deepseek_v4_flash_api":         "#3f8fc4",
        "gemma4_31b_it":                 "#8b6dbf",
        "nemotron_3_5_lightning":        "#75b256",
        "llama4_scout":                  "#6b7280",
        "ministral_3_14b_instruct_2512": "#ba9b5f",
        "medgemma_27b_it":               "#a35b3b",
        "huatuogpt_3_32b":               "#db6a5c",
        "meditron3_70b":                 "#7a4a8a",
        "medreason_8b":                  "#a67f4f",
    }
    # For each base model, keep the reasoning variant when both exist; otherwise
    # keep the sole variant. One bar per model, no pairing.
    groups: dict[str, list[dict]] = {}
    for a in d["arms"]:
        base = a["key"].replace("_reasoning", "")
        groups.setdefault(base, []).append(a)
    kept = []
    for base, arms in groups.items():
        if len(arms) == 1:
            kept.append((base, arms[0]))
        else:
            r = next((a for a in arms if a["key"].endswith("_reasoning")), None)
            kept.append((base, r or arms[0]))
    kept.sort(key=lambda kv: -kv[1]["score"])

    fig = plt.figure(figsize=(11.5, 5.6))
    ax = fig.add_axes([0.075, 0.36, 0.905, 0.55])

    tick_pos, tick_labels = [], []
    BAR_W = 0.72
    for i, (base, a) in enumerate(kept):
        colour = PALETTE.get(base, OPEN_C)
        label = SHORT.get(base, a["label"].replace("  ·R", "").replace(" ·R", ""))
        hatch = "////" if a["closed"] else None
        ax.bar(i, a["score"], width=BAR_W, color=colour, edgecolor="white",
               linewidth=0.6, hatch=hatch, zorder=3)
        ax.text(i, a["score"] + 0.04, f"{a['score']:.2f}",
                ha="center", va="bottom", fontsize=9.0, color=INK, zorder=4)
        tick_pos.append(i)
        tick_labels.append(label)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=11.0,
                       rotation=55, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(kept) - 0.4)
    ax.set_ylim(0, 3.2); ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("Conclusion alignment score  (1–5)  ↑", fontsize=12.0)
    ax.tick_params(labelsize=11.0, length=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
    ax.yaxis.grid(False)

    from matplotlib.patches import Patch  # noqa: E402
    handles = [
        Patch(facecolor="#c7cfda", edgecolor="white", linewidth=0.6, label="open-weight"),
        Patch(facecolor="#c7cfda", edgecolor="white", linewidth=0.6, hatch="////",
              label="proprietary"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10.0, frameon=False,
              handlelength=1.8, handletextpad=0.7, borderpad=0.3, labelspacing=0.5)

    ax_box = ax.get_position()
    fig.suptitle("Tumor board simulation", fontsize=15.0,
                 x=(ax_box.x0 + ax_box.x1) / 2, y=0.965, ha="center")
    return save(fig, out_dir, stem)


PALETTE_B = {
    "gemini_3_7_flash":              "#4a7cbe",
    "grok_4_6":                      "#5f4b8b",
    "qwen3_8_max":                   "#c76a3d",
    "gpt_5_6_sol":                   "#2f7a4c",
    "claude_opus_5":                 "#c98847",
    "deepseek_v4_pro":               "#5b6ea8",
    "deepseek_v4_flash_api":         "#3f8fc4",
    "gemma4_31b_it":                 "#8b6dbf",
    "nemotron_3_5_lightning":        "#75b256",
    "llama4_scout":                  "#6b7280",
    "ministral_3_14b_instruct_2512": "#ba9b5f",
    "medgemma_27b_it":               "#a35b3b",
    "huatuogpt_3_32b":               "#db6a5c",
    "meditron3_70b":                 "#7a4a8a",
    "medreason_8b":                  "#a67f4f",
}


def panel_b(d: dict, out_dir: Path, stem: str) -> list[Path]:
    """Style 1: colour = model family, marker shape = base vs reasoning; log x, side legend."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    arms = list(d["arms"])
    fig = plt.figure(figsize=(11.0, 5.4))
    ax = fig.add_axes([0.075, 0.13, 0.60, 0.80])

    for a in arms:
        base = a["key"].replace("_reasoning", "")
        colour = PALETTE_B.get(base, OPEN_C)
        marker = "^" if a["key"].endswith("_reasoning") else "o"
        ax.scatter(a["tokens"], a["score"], s=90, marker=marker, color=colour,
                   edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(300, 20000)
    ax.set_ylim(1.2, 3.0)
    ax.set_xlabel("Mean output tokens per case  (log)", fontsize=14.0)
    ax.set_ylabel("Conclusion alignment score  (1–5)  ↑", fontsize=14.0)
    ax.tick_params(labelsize=13.0, length=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(False)

    # Legend on the right
    lax = fig.add_axes([0.72, 0.10, 0.27, 0.82])
    lax.set_xlim(0, 1); lax.set_ylim(0, 1); lax.axis("off")

    # Compose entries in the order they sit on the chart family-wise
    ordered_keys = list(PALETTE_B.keys())
    y = 0.98
    row_h = 0.038
    for k in ordered_keys:
        lax.add_patch(plt.Rectangle((0.02, y - row_h * 0.55), 0.06, row_h * 0.55,
                                    facecolor=PALETTE_B[k], edgecolor="white",
                                    linewidth=0.5, zorder=3))
        lax.text(0.11, y - row_h * 0.28, SHORT[k],
                 fontsize=12.5, va="center", ha="left")
        y -= row_h

    # Shape sub-legend
    y -= 0.02
    for label, marker in [("base", "o"), ("reasoning", "^")]:
        lax.plot(0.05, y - row_h * 0.28, marker=marker, ms=8,
                 mfc="#8a94a4", mec="white", mew=0.6, zorder=3)
        lax.text(0.11, y - row_h * 0.28, label,
                 fontsize=12.5, va="center", ha="left")
        y -= row_h

    return save(fig, out_dir, stem)


def panel_b2(d: dict, out_dir: Path, stem: str) -> list[Path]:
    """Style 2: every arm gets a unique colour, direct in-plot labels, no legend.
    Uses adjustText to shift labels away from collisions and draw leader lines."""
    from adjustText import adjust_text
    arms = list(d["arms"])
    tab20 = plt.get_cmap("tab20").colors
    colours = list(tab20) + list(plt.get_cmap("Dark2").colors)[:max(0, len(arms) - len(tab20))]
    fig = plt.figure(figsize=(11.5, 5.8))
    ax = fig.add_axes([0.065, 0.11, 0.92, 0.83])
    texts = []
    for i, a in enumerate(arms):
        col = colours[i % len(colours)]
        ax.scatter(a["tokens"], a["score"], s=110, color=col,
                   edgecolors="white", linewidths=0.9, zorder=3)
        texts.append(ax.text(a["tokens"], a["score"], a["label"],
                             fontsize=12.0, color=INK, zorder=4))
    ax.set_xscale("log")
    ax.set_xlim(300, 20000)
    ax.set_ylim(1.2, 3.0)
    ax.set_xlabel("Mean output tokens per case  (log)", fontsize=14.0)
    ax.set_ylabel("Conclusion alignment score  (1–5)  ↑", fontsize=14.0)
    ax.tick_params(labelsize=13.0, length=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(False)
    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#8a94a4", lw=0.5, alpha=0.7),
                expand=(1.2, 1.4), force_text=(0.35, 0.5))
    return save(fig, out_dir, stem)


def panel_ab2(d: dict, out_dir: Path, stem: str) -> list[Path]:
    """Side-by-side: bar-chart ranking on the left, labelled scatter on the right.
    Mirrors what panel_a and panel_b2 produce individually, drawn into a single figure."""
    from adjustText import adjust_text
    from matplotlib.patches import Patch

    PALETTE = {
        "gemini_3_7_flash":              "#4a7cbe",
        "grok_4_6":                      "#5f4b8b",
        "qwen3_8_max":                   "#c76a3d",
        "gpt_5_6_sol":                   "#2f7a4c",
        "claude_opus_5":                 "#c98847",
        "deepseek_v4_pro":               "#5b6ea8",
        "deepseek_v4_flash_api":         "#3f8fc4",
        "gemma4_31b_it":                 "#8b6dbf",
        "nemotron_3_5_lightning":        "#75b256",
        "llama4_scout":                  "#6b7280",
        "ministral_3_14b_instruct_2512": "#ba9b5f",
        "medgemma_27b_it":               "#a35b3b",
        "huatuogpt_3_32b":               "#db6a5c",
        "meditron3_70b":                 "#7a4a8a",
        "medreason_8b":                  "#a67f4f",
    }

    # A4 page width (210 mm ≈ 8.27 in). Height bumped 4.5 -> 4.60 so
    # that after bbox_inches="tight" trims the caption/legend padding the
    # native PDF lands on exactly 210 x 110 mm.
    fig = plt.figure(figsize=(style.A4_W, 4.60))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.16,
                          left=0.13, right=0.985, top=0.93, bottom=0.17)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    # --- Panel A: horizontal bar chart (reasoning preferred, one bar per model) ---
    groups: dict[str, list[dict]] = {}
    for a in d["arms"]:
        base = a["key"].replace("_reasoning", "")
        groups.setdefault(base, []).append(a)
    kept = []
    for base, arms in groups.items():
        r = next((a for a in arms if a["key"].endswith("_reasoning")), None)
        kept.append((base, r or arms[0]))
    # Ascending so the highest score sits at the top of the chart (barh grows up).
    kept.sort(key=lambda kv: kv[1]["score"])

    BAR_H = 0.72
    tick_pos, tick_labels = [], []
    for i, (base, a_) in enumerate(kept):
        colour = PALETTE.get(base, OPEN_C)
        hatch = "////" if a_["closed"] else None
        axA.barh(i, a_["score"], height=BAR_H, color=colour, edgecolor="white",
                 linewidth=0.6, hatch=hatch, zorder=3)
        axA.text(a_["score"] + 0.04, i, f"{a_['score']:.2f}",
                 ha="left", va="center", fontsize=8.0, color=INK, zorder=4)
        tick_pos.append(i)
        tick_labels.append(SHORT[base])
    axA.set_yticks(tick_pos)
    axA.set_yticklabels(tick_labels, fontsize=8.0)
    axA.set_ylim(-0.6, len(kept) - 0.4)
    axA.set_xlim(0, 3.2); axA.set_xticks([0, 1, 2, 3])
    axA.set_xlabel("Conclusion alignment score  (1–5)  →", fontsize=8.0)
    axA.tick_params(labelsize=8.0, length=2)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axA.spines[sp].set_linewidth(0.6)
    axA.xaxis.grid(False)
    handles = [
        Patch(facecolor="#c7cfda", edgecolor="white", linewidth=0.6, label="open-weight"),
        Patch(facecolor="#c7cfda", edgecolor="white", linewidth=0.6, hatch="////",
              label="proprietary"),
    ]
    axA.legend(handles=handles, loc="lower right", fontsize=8.0, frameon=False,
               handlelength=1.8, handletextpad=0.7, borderpad=0.3, labelspacing=0.5)
    axA.text(-0.22, 1.02, "a", transform=axA.transAxes, fontsize=8.0,
             fontweight="bold", va="bottom", ha="left")

    # --- Panel B (b2): scatter, dot colour matches panel A, per-arm text label
    # placed automatically with adjustText ---
    from adjustText import adjust_text
    b_arms = [a_ for _, a_ in kept]
    for a_ in b_arms:
        base = a_["key"].replace("_reasoning", "")
        col = PALETTE.get(base, OPEN_C)
        axB.scatter(a_["tokens"], a_["score"], s=90, color=col,
                    edgecolors="white", linewidths=1.2, zorder=4)

    SCATTER_SHORT = dict(SHORT)
    SCATTER_SHORT["deepseek_v4_pro"] = "DeepSeek Pro"
    SCATTER_SHORT["deepseek_v4_flash_api"] = "DeepSeek Flash"
    texts = []
    for a_ in b_arms:
        label_text = SCATTER_SHORT.get(a_["key"].replace("_reasoning", ""), a_["label"])
        texts.append(axB.text(a_["tokens"], a_["score"], label_text,
                              fontsize=8.0, color=INK, ha="center", va="center",
                              zorder=6))
    # Spearman correlation over the plotted 15 arms
    import math
    def _spearman(ax, ay):
        rx=[0]*len(ax); ry=[0]*len(ay)
        for r,i in enumerate(sorted(range(len(ax)), key=lambda i:ax[i])): rx[i]=r
        for r,i in enumerate(sorted(range(len(ay)), key=lambda i:ay[i])): ry[i]=r
        mx, my = sum(rx)/len(rx), sum(ry)/len(ry)
        num = sum((x-mx)*(y-my) for x,y in zip(rx,ry))
        den = math.sqrt(sum((x-mx)**2 for x in rx)*sum((y-my)**2 for y in ry))
        return num/den if den else 0.0
    xs=[a_["tokens"] for a_ in b_arms]
    ys=[a_["score"] for a_ in b_arms]
    rho = _spearman(xs, ys)
    axB.text(0.02, 0.97,
             f"Spearman ρ = {rho:+.2f}  (n = {len(b_arms)})",
             transform=axB.transAxes, fontsize=8.0, color=INK,
             va="top", ha="left", zorder=6)
    axB.set_xscale("log")
    axB.set_xlim(280, 25000)
    axB.set_ylim(1.2, 3.15)
    axB.set_xlabel("Mean output tokens per case  (log)", fontsize=8.0)
    axB.set_ylabel("Conclusion alignment score  (1–5)  ↑", fontsize=8.0)
    axB.tick_params(labelsize=8.0, length=2)
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)
    axB.yaxis.grid(False)
    axB.text(-0.07, 1.02, "b", transform=axB.transAxes, fontsize=8.0,
             fontweight="bold", va="bottom", ha="left")
    adjust_text(texts, ax=axB,
                arrowprops=dict(arrowstyle="-", color="#aab2bf", lw=0.6, alpha=0.9,
                                shrinkA=2, shrinkB=1),
                expand=(1.8, 2.4), force_text=(1.0, 1.6),
                force_static=(0.8, 1.1), force_pull=(0.05, 0.1),
                iter_lim=1500,
                only_move={"text": "xy", "static": "xy"})
    return save(fig, out_dir, stem)


def panel_c(d: dict, out_dir: Path, stem: str) -> list[Path]:
    """The lexical metrics, for the appendix: do they order the arms as the judge does?"""
    order = sorted(d["arms"], key=lambda a: a["score"])
    fig = plt.figure(figsize=(style.A4_W, 0.235 * len(order) + 1.0))
    gs = fig.add_gridspec(1, 2, left=0.30, right=0.975, top=0.88, bottom=0.115, wspace=0.16)
    for ax, key, title, lo in ((fig.add_subplot(gs[0, 0]), "rouge_l_f1", "ROUGE-L F1", 0.0),
                               (fig.add_subplot(gs[0, 1]), "bertscore_f1", "BERTScore F1", 0.0)):
        ys = range(len(order))
        ax.barh(list(ys), [a[key] for a in order], height=0.72,
                color=[MM_C if a["condition"] == "multimodal" else CAP_C for a in order],
                zorder=3)
        for i, a in enumerate(order):
            ax.text(a[key] + max(a[key] for a in order) * 0.02, i, f"{a[key]:.3f}",
                    va="center", ha="left", fontsize=FS_VALUE - 0.3, color=MUTED)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([a["label"] for a in order] if key == "rouge_l_f1" else [],
                           fontsize=FS_TICK)
        ax.set_xlim(lo, max(a[key] for a in order) * 1.30)
        ax.set_title(title, fontsize=FS_LABEL, fontweight="bold", loc="left")
        ax.tick_params(labelsize=FS_TICK, length=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Lexical overlap, ordered by judge score", fontsize=FS_TITLE - 0.5,
                 fontweight="bold", x=0.012, y=0.985, ha="left")
    return save(fig, out_dir, stem)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--stem", default="fig4")
    ap.add_argument("--font-family", default="DejaVu Sans")
    ap.add_argument("--font-dir", type=Path, default=None)
    a = ap.parse_args()

    font = resolve_font(a.font_family, a.font_dir)
    style.apply()
    plt.rcParams.update({"text.color": INK, "axes.edgecolor": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    d = gather(a.workspace)
    written = (panel_a(d, a.output_dir, f"{a.stem}a_scores")
               + panel_b(d, a.output_dir, f"{a.stem}b_tokens")
               + panel_b2(d, a.output_dir, f"{a.stem}b2_tokens_labelled")
               + panel_ab2(d, a.output_dir, f"{a.stem}_ab2_combined")
               + panel_c(d, a.output_dir, f"{a.stem}c_lexical"))

    best = max(d["arms"], key=lambda x: x["score"])
    prov = {
        "figure": "4", "task": "task1",
        "metric": "conclusion_alignment (1-5), " + d["protocol"],
        "score_column": SCORE_COL,
        "score_column_rationale":
            "the leaderboard convention in server/benchmark_results.py: cases with no "
            "extractable conclusion enter at the rubric floor of 1, so an arm is graded "
            "on every case it was given rather than on the ones it chose to answer",
        "judge_model": d["judge"], "judge_batches": d["judge_batches"],
        "cases": d["n_cases"],
        "source": {"file": d["board_export"], "sha256": d["board_sha256"],
                   "note": "the board's consolidated export; it resolves the catalog last-wins rule once, so this figure does not"},
        "panel_b_scope": {
            "shown": "multimodal arms only, matching panel a",
            "spearman_tokens_vs_score_all_20": 0.111,
            "spearman_tokens_vs_score_multimodal_10": 0.564,
            "note": "the sign has moved twice as the board changed - -0.125 over the "
                    "seventeen arms of 2026-08-30, +0.286 over the seven multimodal ones, "
                    "+0.564 now that five frontier arms run at high reasoning effort and "
                    "produce long outputs. At n=10 that is about p=0.09. The panel stays "
                    "titled for what it plots rather than for a trend that has not held "
                    "still, and Gemma 4 31B remains the counterexample: 2.54 at 713 tokens.",
        },
        "panel_a_scope": {
            "shown": "multimodal arms only",
            "excluded": [a["key"] for a in d["arms"]
                         if a["condition"] != "multimodal"],
            "reason": "text-only arms cannot take the slide images; ranking "
                      "them beside vision arms would compare model with input",
        },
        "grouping": "fig4b and fig4c still cover all arms, both conditions",
        "arms": d["arms"],
        "absent": {
            "closed_source_frontier": "no Claude, GPT, Gemini or Grok arm exists",
            "kimi3_musespark_qwen38max": "named in the figure plan, absent from the registry",
            "cost_panel": "no price data anywhere; generation_cost declares "
                          "cost_axis=output_tokens.mean, so a cost plot would be the "
                          "token plot relabelled",
            "task1_judge_v31": "a clean full-board v31 batch exists (2026-08-25) but is "
                               "not an active aggregate source and drops difference_rates; "
                               "it moves every arm down 0.07-0.24 and reorders some",
        },
        "panel_a_uncaptioned": (
            "fig4a carries no subtitle, footnote or per-bar coverage note. Everything "
            "it no longer says is here and must go in the LaTeX caption: the scale, the "
            "ten excluded text-only arms, and the arms that did not conclude every case."
        ),
        "panel_a_incomplete_coverage": {
            a["key"]: f"{a['scored']}/{a['test']} concluded"
            for a in d["arms"] if a["condition"] == "multimodal"
            and a["scored"] != a["test"]},
        "caption": (
            f"Tumor board simulation. Each arm generates a multi-specialist discussion and a "
            f"treatment conclusion from the case summary and its slides; the conclusion is "
            f"scored against the real board's decision on a 1-5 alignment rubric "
            f"({d['protocol']}, judge {d['judge']}) over {d['n_cases']} test cases. Bars are "
            f"grouped by input condition and are not a single ranking: text-only arms cannot "
            f"take the slide images and answer from captions alone, which the benchmark files "
            f"as an ablation of Task 1 rather than a second way of running it. Scores grade "
            f"every case the arm was given, with cases it never concluded entering at the "
            f"rubric floor of 1. The strongest arm, {best['label'].strip()}, reaches "
            f"{best['score']:.2f} of 5; no arm reaches 3."
        ),
        "font": font, "outputs": [str(p) for p in written],
    }
    (a.output_dir / f"{a.stem}.provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": [str(p) for p in written], "arms": len(d["arms"]),
                      "best": [best["label"].strip(), round(best["score"], 3)]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
