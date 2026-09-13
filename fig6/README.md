# fig6

Two-panel training figure and its raw inputs. Both panels are Helvetica
8pt; combined they produce Figure 6 in the paper.

## Files

| file | what it is |
|---|---|
| `fig6a.{pdf,png,svg}` | Bar chart: Qwen2.5-VL-3B vs `+SFT` vs `+SFT+RL` on Conclusion alignment + the four Board Simulation decisions (Therapy, Surgery, Next action, Clinical trial), scored 0–5. All 184 test cases. |
| `fig6b.{pdf,png,svg}` | RL validation-reward curve, truncated at the peak step (step 400 / epoch 4), with a dashed horizontal line marking the maximum val reward 0.391 that selects the shipped checkpoint. |
| `source/v25_training_curves_final.csv` | 451 optimizer-step rows × 26 columns from the RL run: per-step train loss/reward, per-eval-step val reward, and reward decomposition into match / format / plan / consistency. |

## Regenerating

Both panels come out of the same routines used by the combined fig6:

```bash
python standalone/fig6_split.py \
    --output-dir fig6 \
    --font-family Helvetica --font-dir <path with Helvetica.ttc> \
    --curves-csv fig6/source/v25_training_curves_final.csv
```

To rebuild the two-panel version instead, use `standalone/fig6_combined.py`
with the same arguments.

## Numbers behind panel (a)

Hard-coded in `standalone/fig6_training.py` / `standalone/fig6_combined.py`
(same values under `ARMS`):

| arm            | Conclusion align. | Therapy | Surgery | Next action | Clinical trial |
|----------------|:-----------------:|:-------:|:-------:|:-----------:|:--------------:|
| Qwen2.5-VL-3B  | 1.58              | 1.06    | 1.23    | 1.34        | 0.98           |
| + SFT          | 1.67              | 1.21    | 1.99    | 2.03        | 1.22           |
| + SFT + RL     | 1.86              | 1.39    | 2.27    | 2.07        | 1.13           |

## Numbers behind panel (b)

Read from `source/v25_training_curves_final.csv`.  Peak validation reward
0.391 at step 400 (epoch 4.00); the curve is trimmed to step 400 because
val reward degrades after that while train reward keeps climbing —
consistent with early-stopping the RL checkpoint at that step.
