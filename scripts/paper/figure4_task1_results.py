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

MM_C, CAP_C = "#1f78b4", "#a6cee3"        # vision-capable vs caption-only
SCORE_FIELD = "dimensions_all_responses"

SHORT = {
    "deepseek_v4_pro": "DeepSeek-V4-Pro", "deepseek_v4_flash_api": "DeepSeek-V4-Flash",
    "qwen3_8_27b": "Qwen3.8-27B", "medgemma_27b_it": "MedGemma-27B",
    "huatuogpt_3_32b": "HuatuoGPT-3-32B", "meditron3_70b": "Meditron-3-70B",
    "medreason_8b": "MedReason-8B", "llama4_scout": "Llama 4 Scout",
    "gemma4_31b_it": "Gemma 4 31B", "ministral_3_14b_instruct_2512": "Ministral 3 14B",
    "nemotron_3_5_lightning": "Nemotron 3.5",
}


def refuse(msg: str) -> None:
    raise SystemExit(f"figure4: {msg}")


def label_of(key: str) -> str:
    base = key.replace("_reasoning", "")
    if base not in SHORT:
        refuse(f"no short label for {key}; add it to SHORT")
    return SHORT[base] + ("  ·R" if key.endswith("_reasoning") else "")


def gather(ws: Path) -> dict:
    cat_p = ws / "OpenTumorBoard/evaluation/results_catalog.json"
    cat = json.loads(cat_p.read_text(encoding="utf-8"))
    cond = {m["model_key"]: m["runs"]["task1"]["condition"]
            for m in cat["active"]["models"] if "task1" in m["runs"]}

    # Resolve through the catalog's own last-one-wins rule rather than naming a
    # batch. Naming one is how this figure first got written, and it pointed at v25
    # a week after v25 left aggregate_sources; the guard below caught it, but the
    # guard should not have been the only thing standing between a stale rubric and
    # a published figure.
    if "Later aggregate sources override earlier" not in cat["policy"]["source_priority"]:
        refuse("policy.source_priority is no longer last-one-wins; this resolution "
               "encodes that rule and must be revisited")
    rows, batch_of, proto_of = {}, {}, {}
    for rel in cat["active"]["aggregate_sources"]["judge"]:
        p = ws / "model_evaluation" / rel
        if not p.exists():
            refuse(f"aggregate judge source missing on disk: {rel}")
        jd = json.loads(p.read_text(encoding="utf-8"))
        for r in jd.get("results", {}).get("task1", []) or []:
            rows[r["model_key"]] = r
            batch_of[r["model_key"]] = rel
            proto_of[r["model_key"]] = jd.get("protocol_version", "")

    board = {k: v for k, v in batch_of.items() if k in cond}
    if len(set(board.values())) != 1:
        refuse(f"the {len(cond)} board arms resolve to {len(set(board.values()))} different "
               "judge batches; a bar chart across them would mix protocols")
    judge_batch = next(iter(set(board.values())))
    protos = {proto_of[k] for k in cond if k in proto_of}
    if len(protos) != 1:
        refuse(f"arms span protocols {protos}")
    protocol = protos.pop()

    jb = json.loads((ws / "model_evaluation" / judge_batch.replace(
        "results/summary.json", "build_config.json")).read_text(encoding="utf-8"))
    judged_run = {k.split(":", 1)[1]: os.path.basename(os.path.dirname(v["responses"]))
                  for k, v in jb["candidate_runs"].items() if k.startswith("task1:")}

    def metric(kind, task, pick):
        out = {}
        for rel in cat["active"]["aggregate_sources"][kind]:
            p = ws / "model_evaluation" / rel
            if not p.exists():
                continue
            for e in (json.loads(p.read_text(encoding="utf-8")).get("results") or []):
                if e.get("task") != task:
                    continue
                key = e["model"]["key"] if "model" in e else e["model_key"]
                src = e.get("responses_path") or e.get("response_file") or e.get("responses") or ""
                out[key] = (pick(e), os.path.basename(os.path.dirname(src)))
        return out

    tokens = metric("generation_cost", "task1", lambda e: e["output_tokens"]["mean"])
    rouge = metric("rouge", "task1", lambda e: e["item_macro"]["rouge_l"]["f1"]["mean"])
    bert = metric("bertscore", "task1", lambda e: e["item_macro"]["raw"]["f1"]["mean"])

    arms = []
    for k_, r in rows.items():
        k = r["model_key"]
        if k not in cond:
            continue
        for src, name in ((tokens, "output tokens"), (rouge, "ROUGE-L"), (bert, "BERTScore")):
            if k not in src:
                refuse(f"{k} has no {name}; the panels would cover different arms")
            # The output contract changed on 2026-08-30 and every arm was regenerated.
            # A metric still computed over the superseded run would pair a new score
            # with an old response, which no panel would reveal.
            if k in judged_run and src[k][1] != judged_run[k]:
                refuse(f"{k}: {name} was computed over run {src[k][1]!r} but the judge "
                       f"scored {judged_run[k]!r}; the panels would mix generations")
        arms.append({
            "key": k, "label": label_of(k), "condition": cond[k],
            "score": r[SCORE_FIELD]["conclusion_alignment"],
            "score_scored_only": r["dimensions"]["conclusion_alignment"],
            "scored": r["scored_records"], "test": r["test_records"],
            "format_failures": r["candidate_format_failures"],
            "tokens": tokens[k][0], "rouge_l_f1": rouge[k][0], "bertscore_f1": bert[k][0],
            "run": judged_run.get(k),
            "difference_rates": r.get("difference_rates"),
        })
    if len(arms) != len(cond):
        refuse(f"judge batch covers {len(arms)} of the {len(cond)} active Task 1 arms")

    groups = [("Slides + captions", [a for a in arms if a["condition"] == "multimodal"]),
              ("Captions only", [a for a in arms if a["condition"] != "multimodal"])]
    for name, g in groups:
        if not g:
            refuse(f"group {name!r} is empty")
        g.sort(key=lambda a: a["score"])

    return {"arms": arms, "groups": groups, "n_cases": arms[0]["test"],
            "protocol": protocol, "judge_batch": judge_batch,
            "judge": json.loads((ws / "model_evaluation" / judge_batch).read_text(
                encoding="utf-8")).get("judge_model"),
            "catalog_sha256": hashlib.sha256(cat_p.read_bytes()).hexdigest()}


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


def save(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("pdf", "svg", "png"):
        p = out_dir / f"{stem}.{suffix}"
        fig.savefig(p, dpi=DPI, facecolor="white", bbox_inches="tight")
        written.append(p)
    plt.close(fig)
    return written


def panel_a(d: dict, out_dir: Path, stem: str) -> list[Path]:
    """Bars in two groups. No global sort: the groups are not one league table."""
    nrow = len(d["arms"]) + 1                      # +1 for the gap between groups
    fig_h = 0.235 * nrow + 0.95
    fig = plt.figure(figsize=(style.A4_W, fig_h))
    ax = fig.add_axes([0.335, 0.115, 0.625, 0.80])

    y, ticks, labels = 0.0, [], []
    for gi, (name, group) in enumerate(reversed(d["groups"])):
        colour = MM_C if name.startswith("Slides") else CAP_C
        for a in group:
            ax.barh(y, a["score"], height=0.72, color=colour, zorder=3)
            note = ""
            if a["scored"] != a["test"]:
                note = f"   {a['scored']}/{a['test']} concluded"
            ax.text(a["score"] + 0.03, y, f"{a['score']:.2f}{note}", va="center",
                    ha="left", fontsize=FS_VALUE, color=MUTED, zorder=4)
            ticks.append(y); labels.append(a["label"]); y += 1
        ax.text(-0.335, y - len(group) / 2 - 0.5, name, fontsize=FS_LABEL, color=INK,
                rotation=90, va="center", ha="center", fontweight="bold",
                transform=ax.get_yaxis_transform(which="grid"), clip_on=False)
        if gi == 0:
            ax.axhline(y - 0.5 + 0.0, color=INK, lw=0.8, zorder=5)
            y += 1

    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.set_ylim(-0.8, y - 0.2)
    ax.set_xlim(0, 3.0)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Conclusion alignment  (1–5)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK, length=2)
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)

    fig.suptitle("Tumor board simulation", fontsize=FS_TITLE, fontweight="bold",
                 x=0.012, y=0.995, ha="left")
    fig.text(0.012, 0.955,
             f"{d['n_cases']} cases  ·  no arm reaches 3 of 5",
             fontsize=FS_NOTE, color=MUTED, ha="left", va="top")
    return save(fig, out_dir, stem)


def panel_b(d: dict, out_dir: Path, stem: str) -> list[Path]:
    """Tokens against score. Log x because the arms span an order of magnitude."""
    fig = plt.figure(figsize=(0.66 * style.A4_W, 0.30 * style.A4_H))
    ax = fig.add_axes([0.135, 0.145, 0.835, 0.755])
    for name, group in d["groups"]:
        colour = MM_C if name.startswith("Slides") else CAP_C
        ax.scatter([a["tokens"] for a in group], [a["score"] for a in group], s=26,
                   c=colour, edgecolors="white", linewidths=0.6, zorder=3, label=name)
    for a in d["arms"]:
        ax.annotate(a["label"], (a["tokens"], a["score"]), textcoords="offset points",
                    xytext=(4.5, 3.0), fontsize=FS_NOTE - 0.6, color=MUTED, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Mean output tokens per case  (log)", fontsize=FS_LABEL)
    ax.set_ylabel("Conclusion alignment  (1–5)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK, length=2)
    ax.grid(True, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=FS_NOTE, frameon=False, loc="lower right")
    fig.suptitle("More output is not better", fontsize=FS_TITLE - 0.5,
                 fontweight="bold", x=0.012, y=0.985, ha="left")
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
        ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
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
               + panel_c(d, a.output_dir, f"{a.stem}c_lexical"))

    best = max(d["arms"], key=lambda x: x["score"])
    prov = {
        "figure": "4", "task": "task1",
        "metric": "conclusion_alignment (1-5), " + d["protocol"],
        "score_field": SCORE_FIELD,
        "score_field_rationale":
            "the leaderboard convention in server/benchmark_results.py: cases with no "
            "extractable conclusion enter at the rubric floor of 1, so an arm is graded "
            "on every case it was given rather than on the ones it chose to answer",
        "judge_model": d["judge"], "judge_batch": d["judge_batch"],
        "cases": d["n_cases"], "results_catalog_sha256": d["catalog_sha256"],
        "grouping": "input condition; the two groups are not one ranking",
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
