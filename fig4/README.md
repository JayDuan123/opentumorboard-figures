# fig4

Everything needed to reproduce Figure 4 (SPECIALIST TURN by question type)
in one place.

| file | what it is |
|---|---|
| `fig4_data.csv` | 9 QA types × 9 arm clinical-equivalence grid + `All questions` overall row + `_arm_block` block-membership row. Column headers carry `[cap]` (captions only) or `[mm]` (read slide images) so the two are never conflated. |
| `dump_fig4_data.py` | Regenerates `fig4_data.csv` by importing the constants from `../standalone/fig4_heatmap.py`, so the CSV cannot drift from the plot. Run: `python fig4/dump_fig4_data.py`. |
| `fig4_heatmap.provenance.json` | Judge model, batch manifests, and case set that back the numbers in the grid. |

The plotting script itself lives at `../standalone/fig4_heatmap.py` and the
older provenance snapshot at `../provenance/fig4.provenance.json`.
