# fig5/rendered/

Fresh outputs from running `figure4_task1_results.py` on the CSV shipped
in `../source/board_results.csv`, kept alongside the paper's shipped
`../fig5.pdf` so anyone can verify the source reproduces the figure.

Command (from repo root):

```bash
python -m scripts.paper.figure4_task1_results \
    --workspace /tmp/fig5_ws \
    --output-dir fig5/rendered \
    --stem fig5 \
    --font-family Helvetica --font-dir <path with Helvetica.ttc>
```

`--workspace` needs the CSV at
`<workspace>/model_evaluation/board_results_20260903/board_results.csv`.
A one-liner shim is

```bash
mkdir -p /tmp/fig5_ws/model_evaluation/board_results_20260903
cp fig5/source/board_results.csv \
   /tmp/fig5_ws/model_evaluation/board_results_20260903/
```

## Files this run produced

| file | what it is |
|---|---|
| `fig5_ab2_combined.{pdf,png,svg}` | **The paper's fig5** — leaderboard (bars) fused with tokens vs alignment (scatter). Matches `../fig5.pdf` in layout and dimensions (600.9 vs 606.3 pt wide). |
| `fig5a_scores.{pdf,png,svg}` | Panel A alone: leaderboard bars, sorted by conclusion alignment descending. |
| `fig5b_tokens.{pdf,png,svg}` | Panel B alone (dots only). |
| `fig5b2_tokens_labelled.{pdf,png,svg}` | Panel B with each dot labelled by its model — the version actually placed in the composite. |
| `fig5c_lexical.{pdf,png,svg}` | Appendix table: ROUGE-L and BERTScore per configuration. Not used in the main body. |
| `fig5.provenance.json` | Judge model, batch manifests, best configuration (Gemini 3.7 Flash at 2.783), and the exact 15 shared / 5 task-1-only arms this run drew. |

## Verifying against the shipped fig5

- Shipped `../fig5.pdf`: 606.3 × 300.5 pt, 46 Helvetica 8pt + 14 Helvetica
  7.2pt + 2 Helvetica-Bold 12pt + a couple of AdobeSongStd / ArialMT
  spans from an older toolchain.
- Fresh `fig5_ab2_combined.pdf`: 600.9 × 298.0 pt, 62 Helvetica 8pt +
  2 Helvetica-Bold 8pt + 2 Helvetica 5.6pt — cleaner font set, no
  Adobe Song / Arial residue.

Same layout, same data, same look. The differences are the natural
consequence of re-running the code today rather than reusing a shipped
PDF: fresh subsetting, no leftover fonts from whatever tool the earlier
render passed through.
