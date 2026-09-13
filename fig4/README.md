# fig4

Everything needed to reproduce Figure 4 (SPECIALIST TURN by question type)
in one place — the plot's numbers, the script that regenerates them, the
metadata that dates and cites them, and the raw per-arm judge outputs
they came from.

## Files

| file | what it is |
|---|---|
| `fig4_data.csv` | 9 QA types × 9 arm clinical-equivalence grid + an `All questions` overall row + a trailing `_arm_block` row that names each arm's block. Column headers carry `[cap]` (answered from captions only) or `[mm]` (read the slide images). |
| `dump_fig4_data.py` | Regenerates `fig4_data.csv` by importing the constants from `../standalone/fig4_heatmap.py`, so the CSV cannot drift from the plot. Run: `python fig4/dump_fig4_data.py`. |
| `fig4_heatmap.provenance.json` | Judge model, batch manifests, and the 4,844-question case set that back the grid. |
| `source/` | Per-arm raw judge outputs (nine subdirectories, one per column of the heatmap). Each holds `summary.json` (aggregate stats) and `task2.per_item.jsonl` (4,844 rows, one per test question, with the candidate's answer, the judge's 1–5 verdict, and full provenance sha256s). See `source/README.md`. |

## The plotting code

The heatmap script itself lives one level up at
`../standalone/fig4_heatmap.py` (the `standalone/` directory groups every
paper figure's script). The older provenance snapshot from the manifest-
driven pipeline sits at `../provenance/fig4.provenance.json`.

## Reproducing the numbers end-to-end

1. **Grid** — `python fig4/dump_fig4_data.py` rewrites `fig4_data.csv` from
   the constants in the plotting script.
2. **Grid from raw** — each `source/<arm>/task2.per_item.jsonl` has 4,844
   rows. Group by `qa_type`, treat `candidate_format_failure=true` as
   1.0, take the mean per QA type per arm, and you land on the same
   values that `fig4_data.csv` and the heatmap show.
3. **Plot** — `python ../standalone/fig4_heatmap.py --output-dir out
   --stem fig4 --font-family Helvetica --font-dir <path with Helvetica>`
   writes `out/fig4.{pdf,png,svg}`.

Judge is **Qwen3.8-27B** running the `llm_judge_task2_v4` protocol
(2026-09-03). Every arm here was run against the same test set of
4,844 SPECIALIST TURN questions, 184 cases, 66 recordings.
