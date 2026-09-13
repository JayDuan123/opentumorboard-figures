# fig5

The Board Simulation leaderboard (left panel) and its scatter of
conclusion alignment against mean output tokens per case (right panel),
plus every input the plotting script reads.

## Files

| file | what it is |
|---|---|
| `figure4_task1_results.py` | Plotting code. Retained under its original repo name (paper section rename was **fig4→fig5** without a source-file rename) — the "Task 1" in its name is the codebase's Board Simulation. Emits `panel_a` (bars), `panel_b2` (scatter), `panel_c` (ROUGE / BERTScore appendix table), and a `dual` composite that fuses `panel_a` and `panel_b2` into the single figure the paper uses. |
| `fig5.pdf` / `fig5.png` | Current rendered figure as shipped in the paper. Two panels: A (leaderboard) + B (tokens vs alignment, Spearman ρ = +0.63). |
| `fig5_dual.provenance.json` | Snapshot: 15 shared arms and 5 Task-1-only arms fed into the composite, the score column convention, and every judge batch that contributed. |
| `fig5_bars_arrows.provenance.json` | Same provenance for the sibling `bars_arrows` layout kept in the repo for the ranking-shift discussion, not currently used in the paper. |
| `source/board_results.csv` | 35-row aggregate table: one row per model configuration × condition, with conclusion alignment, mean output tokens, format failures, clinical equivalence, and companion metrics. All numbers in the two panels come out of this file. |

## Dependencies

The script imports `scripts.paper.style` (already in the parent repo at
`../scripts/paper/style.py`) for the shared 8/10-pt Helvetica style, the
A4 canvas width `A4_W`, and the palette. Run it from the repo root so
that import resolves:

```bash
python -m scripts.paper.figure4_task1_results \
    --workspace <workspace root that holds board_results.csv> \
    --output-dir <where to write panels>
```

`board_results.csv` in this folder can serve as the workspace's
`model_evaluation/board_results_20260903/board_results.csv` — copy it
under that path or point `--workspace` at wherever it already lives.

## Notes on the current fig5

- Native page dimensions are **606 × 301 pt (8.42 × 4.17 in)**. Rendered
  under `\includegraphics[width=\linewidth]` this scales by roughly 0.77
  and the 8pt source text prints at ~6.2pt — smaller than the 7.8pt that
  fig6 (`figsize=(7.6, 2.7)` → 6.63 × 2.86 in native) reaches at the same
  `\linewidth`. If the two figures need to match on paper, either enlarge
  the two-panel dual figsize to something like `A4_W × 3.2 in` **or**
  regenerate fig6 at the same native ratio; see the fig6 README for its
  reproduction path.
- The paper's fig5 uses `pdf.use14corefonts=True`, so the PDF references
  Helvetica without embedding it — every viewer supplies its own copy.
  Compare with fig4/fig6, which embed a subsetted Type3 Helvetica from
  the local `Helvetica.ttc`. Both are visually Helvetica, but a PDF
  hex-dump of fig5 shows `/BaseFont /Helvetica` where fig4/fig6 show
  `EDQXKT+Helvetica`.
