# standalone figure scripts

Three drop-in scripts that reproduce **Fig. 3**, **Fig. 4** and **Fig. 6** from
the OpenTumorBoard paper without any data files. All numbers are hard-coded
inside each script, so `python fig3_wheel.py`, `python fig4_heatmap.py`,
`python fig6_training.py` each write `pdf`, `png` and `svg` in `--output-dir`
(default `.`) using only `matplotlib`.

## quick start

```bash
pip install matplotlib
python fig3_wheel.py    --output-dir out
python fig4_heatmap.py  --output-dir out
python fig6_training.py --output-dir out
```

## fonts

Use Helvetica or an Arimo / Nimbus Sans clone by pointing at a directory of
`.ttf` / `.otf` files:

```bash
python fig3_wheel.py --font-family Arimo --font-dir ~/.fonts
```

Without `--font-family` matplotlib falls back to DejaVu Sans, which is fine.

## what's in each file

| Script | Overleaf id | Data | Rendered size |
|---|---|---|---|
| `fig3_wheel.py`    | Fig. 3 | 611 cases: cancer sites, target specialists (16,215 QA), specialist turn types (16,215 QA), slide-based modality | 3.6 × 3.6 in |
| `fig4_heatmap.py`  | Fig. 4 | 9 arms × 9 QA types, Qwen3.8-27B judge (`task2_judge_v4`), 4,844 test questions | 9.5 × 5.6 in |
| `fig6_training.py` | Fig. 6 | Qwen2.5-VL-3B base / +SFT / +SFT+RL on conclusion alignment + 4 rubric decisions | 3.6 × 2.6 in |

Fig. 5 (Task 1 leaderboard + tokens vs. score) is a hand-tuned PDF stored
directly in `overleaf/figures/fig5.pdf`; there is no scripted version.

To update the hard-coded numbers, re-run the paper's data pipeline in the
main repo and drop the resulting counts into the `SITES` / `ROLES` /
`QTYPES` / `MODALITY` dicts at the top of `fig3_wheel.py`, the `GENERAL` /
`MEDICAL` arm lists in `fig4_heatmap.py`, and the `ARMS` list in
`fig6_training.py`.
