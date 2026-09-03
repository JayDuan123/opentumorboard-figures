# Where each board row's results live (2026-09-02)

Authority: OpenTumorBoard/evaluation/results_catalog.json (active.models + active.aggregate_sources).
Judge batch = last catalog judge source, in order, whose protocol is task2 v4 / task1 v31 and whose results[task] lists the model (the catalog's last-wins merge).
Multimodal Specialist Turn rows are keyed with the mm_ prefix inside the judge summary (catalog tasks.task2.result_key_prefixes). Per-item judge outputs: <batch>/task{1,2}.output.jsonl. Runs hold the generations. ROUGE-L / BERTScore / cost: model_evaluation/results/candidate_v1/video_split_60_10_30_v3/.

### Specialist Turn (code task2)

| model | condition | run (model_evaluation/runs/…) | judge batch (model_evaluation/judge/…) | scored |
|---|---|---|---|---|
| llama4_scout | multimodal | mm_llama4_scout_task2_test_v3_xsfixed_multimodal_20260813_max16384 | board13_task2_judge_v4_images_20260814 (keyed mm_llama4_scout) | 4844 |
| huatuogpt_3_32b | caption_only | huatuogpt_3_32b_task2_test_v3_xsfixed_20260813_max8192_slidealigned | board13_task2_judge_v4_images_20260814 | 4844 |
| huatuogpt_3_32b_reasoning | caption_only | huatuogpt_3_32b_reasoning_task2_test_v3_xsfixed_20260809_max8192_slidealigned | board13_task2_judge_v4_images_20260814 | 4844 |
| ministral_3_14b_instruct_2512 | multimodal | mm_ministral_3_14b_instruct_2512_task2_test_v3_xsfixed_multimodal_20260816_max16384 | board13_task2_judge_v4_images_20260814 | 4844 |
| medreason_8b | caption_only | medreason_8b_task2_test_v3_xsfixed_20260809_max8192_slidealigned | board13_task2_judge_v4_images_20260814 | 4843 |
| gemma4_31b_it | multimodal | mm_gemma4_31b_it_task2_test_v3_xsfixed_multimodal_20260813_max16384 | board13_task2_judge_v4_images_20260814 (keyed mm_gemma4_31b_it) | 4844 |
| gemma4_31b_it_reasoning | multimodal | mm_gemma4_31b_it_reasoning_task2_test_v3_xsfixed_multimodal_20260813_max16384 | board13_task2_judge_v4_images_20260814 (keyed mm_gemma4_31b_it_reasoning) | 4844 |
| medgemma_27b_it | multimodal | mm_medgemma_27b_it_task2_test_v3_xsfixed_multimodal_20260813_max16384 | board13_task2_judge_v4_images_20260814 (keyed mm_medgemma_27b_it) | 4844 |
| nemotron_3_5_lightning | caption_only | nemotron_3_5_lightning_task2_test_v3_xsfixed_20260813_max8192_slidealigned | board13_task2_judge_v4_images_20260814 | 4844 |
| nemotron_3_5_lightning_reasoning | caption_only | nemotron_3_5_lightning_reasoning_task2_test_v3_xsfixed_20260812_max8192_slidealigned | board13_task2_judge_v4_images_20260814 | 4842 |
| meditron3_70b | caption_only | meditron3_70b_task2_test_v3_xsfixed_20260817_max8192_slidealigned | meditron3_70b_task2_judge_v4_images_20260817_max8192 | 4844 |
| deepseek_v4_pro | caption_only | deepseek_v4_pro_task2_test_v3_xsfixed_20260815_max8192 | deepseek_v4_pro_task2_judge_v4_images_20260815 | 4844 |
| deepseek_v4_pro_reasoning | caption_only | deepseek_v4_pro_reasoning_task2_test_v3_xsfixed_20260815_max8192 | deepseek_v4_pro_task2_judge_v4_images_20260815 | 4841 |
| deepseek_v4_flash_api | caption_only | deepseek_v4_flash_api_task2_test_v3_xsfixed_20260814_max8192_slidealigned | deepseek_v4_flash_api_task2_judge_v4_images_20260815 | 4844 |
| deepseek_v4_flash_api_reasoning | caption_only | deepseek_v4_flash_api_reasoning_task2_test_v3_xsfixed_20260814_max8192_slidealigned | deepseek_v4_flash_api_task2_judge_v4_images_20260815 | 4839 |

### Board Simulation (code task1)

| model | condition | run (model_evaluation/runs/…) | judge batch (model_evaluation/judge/…) | scored |
|---|---|---|---|---|
| llama4_scout | multimodal | llama4_scout_task1_test_v3_xsfixed_multimodal_v3_20260830_max65536 | board17_transcript3_task1_judge_v31_images_20260830 | 184 |
| huatuogpt_3_32b | ablation_caption_only | huatuogpt_3_32b_task1_test_v3_xsfixed_transcript_captiononly_v1_20260830_max32768 | board17_transcript3_task1_judge_v31_images_20260830 | 184 |
| huatuogpt_3_32b_reasoning | ablation_caption_only | huatuogpt_3_32b_reasoning_task1_test_v3_xsfixed_transcript_captiononly_v1_20260830_max32768 | board17_transcript3_task1_judge_v31_images_20260830 | 184 |
| ministral_3_14b_instruct_2512 | multimodal | ministral_3_14b_instruct_2512_task1_test_v3_xsfixed_multimodal_v3_20260830_max65536 | board17_transcript3_task1_judge_v31_images_20260830 | 184 |
| medreason_8b | ablation_caption_only | medreason_8b_task1_test_v3_xsfixed_transcript_captiononly_v1_20260830_max65536 | board17_transcript3_task1_judge_v31_images_20260830 | 175 |
| gemma4_31b_it | multimodal | gemma4_31b_it_task1_test_v3_xsfixed_multimodal_v3_20260830_max65536 | board17_transcript3_task1_judge_v31_images_20260830 | 184 |
| gemma4_31b_it_reasoning | multimodal | gemma4_31b_it_reasoning_task1_test_v3_xsfixed_multimodal_v3_20260830_max65536 | board17_transcript3_task1_judge_v31_images_20260830 | 184 |
| medgemma_27b_it | multimodal | medgemma_27b_it_task1_test_v3_xsfixed_multimodal_v3_20260830_max65536 | board17_transcript3_task1_judge_v31_images_20260830 | 184 |
| nemotron_3_5_lightning | ablation_caption_only | nemotron_3_5_lightning_task1_test_v3_xsfixed_transcript_captiononly_v1_20260830_max65536 | board17_transcript3_task1_judge_v31_images_20260830 | 184 |
| nemotron_3_5_lightning_reasoning | ablation_caption_only | nemotron_3_5_lightning_reasoning_task1_test_v3_xsfixed_transcript_captiononly_v1_20260830_max65536 | board17_transcript3_task1_judge_v31_images_20260830 | 184 |
| meditron3_70b | ablation_caption_only | meditron3_70b_task1_test_v3_xsfixed_transcript_captiononly_v1_20260830_max65536 | board17_transcript3_task1_judge_v31_images_20260830 | 180 |
| deepseek_v4_pro | ablation_caption_only | deepseek_v4_pro_task1_test_v3_xsfixed_transcript_captiononly_v1_20260830_max65536 | board17_transcript3_task1_judge_v31_images_20260830 | 184 |
| deepseek_v4_pro_reasoning | ablation_caption_only | deepseek_v4_pro_reasoning_task1_test_v3_xsfixed_transcript_captiononly_v1_20260830_max65536 | board17_transcript3_task1_judge_v31_images_20260830 | 184 |
| deepseek_v4_flash_api | ablation_caption_only | deepseek_v4_flash_api_task1_test_v3_xsfixed_transcript_captiononly_v1_20260830_max65536 | board17_transcript3_task1_judge_v31_images_20260830 | 184 |
| deepseek_v4_flash_api_reasoning | ablation_caption_only | deepseek_v4_flash_api_reasoning_task1_test_v3_xsfixed_transcript_captiononly_v1_20260830_max65536 | board17_transcript3_task1_judge_v31_images_20260830 | 184 |
| claude_opus_5 | multimodal | opus5_task1_test_v3_xsfixed_multimodal_20260901_max128000 | api4_task1_judge_v31_images_20260901 | 184 |
| gemini_3_7_flash | multimodal | gemini37flash_task1_test_v3_xsfixed_multimodal_20260901_max65536_effortxhigh | api4_task1_judge_v31_images_20260901 | 184 |
| qwen3_8_max | multimodal | qwen38max_task1_test_v3_xsfixed_multimodal_20260901_max131072_effortxhigh | api4_task1_judge_v31_images_20260901 | 184 |
| grok_4_6 | multimodal | grok46_task1_test_v3_xsfixed_multimodal_20260901_max450000_effortxhigh | api4_task1_judge_v31_images_20260901 | 184 |
| gpt_5_6_sol | multimodal | gpt56sol_task1_test_v3_xsfixed_multimodal_or_20260901_max128000_effortxhigh | gpt56sol_task1_judge_v31_images_20260901 | 181 |



## This package

- `board_results.csv` / `.json`: one row per (model, task) with the published headline numbers, resolved by the catalog's last-wins rule, and the source file each number came from.
- `sources/`: byte copies of every summary the numbers were read from, at their original relative paths under model_evaluation/, plus each judge batch's build_config, run_config and per-item `task{1,2}.output.jsonl`.
- Regenerate: the script is in the session log of 2026-09-02; inputs are results_catalog.json and the files under sources/.

`conclusion_alignment` / `clinical_equivalence` are the published numbers: format failures stay in the denominator at the rubric floor of 1. The `_scored_only` columns average the judged responses alone and are NOT what the board shows.
