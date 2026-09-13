# fig4/source/

Raw per-arm judge outputs that back the numbers in `../fig4_data.csv`.
Nine directories, one per arm in Figure 4 (five general-purpose models
plus four medical models). Every arm here was judged by **Qwen3.8-27B**
(`judge_v4`) on the SPECIALIST TURN test set of 4,844 questions.

```
fig4/source/
├── deepseek_v4_pro/
│   ├── summary.json         aggregate stats: overall mean CE, per-QA-type
│   │                        means, malformed counts, cases seen, etc.
│   └── task2.per_item.jsonl one JSON line per test question with the
│                            candidate's answer, the judge's 1-5 clinical
│                            equivalence score, and full provenance sha256s
├── deepseek_v4_flash/
├── llama4_scout/
├── gemma4_31b_it/
├── ministral_3_14b/
├── medgemma_27b/
├── huatuogpt_3_32b/
├── meditron3_70b/
└── medreason_8b/
```

Every `per_item.jsonl` has 4,844 rows (one per test question). Fields
include `qa_type` (the row in the heatmap), `candidate_format_failure`
(malformed answer → scored 1 by rule), and `judge_status` / the judge's
verdict object under `judge_output`.

The `[cap]` / `[mm]` tag in `../fig4_data.csv` records which arms read
slide images and which answered from captions only. The `mm_` prefix on
some source directory names in the upstream release corresponds to the
same distinction.

To re-derive the 9×9 grid in `../fig4_data.csv` from these files:
aggregate each arm's `per_item.jsonl` by `qa_type`, treat every
`candidate_format_failure=true` row as 1.0, and take the mean; the
`_arm_block` row groups arms into the two blocks the heatmap draws.
