#!/usr/bin/env python3
"""Copy the inputs the four figure scripts read into a portable tree.

The full workspace is over a gigabyte, almost all of it slide images and raw
video the figures never touch. This walks results_catalog.json and pulls only
what the scripts actually open - about 150 MB - preserving relative paths so the
copy can be passed to --workspace unchanged.

It resolves the file set from the catalog rather than from a checked-in list.
The catalog moved under this project once already (Task 1's published rubric went
v25 -> v31 mid-week, and the judge batch, the candidate runs and every auxiliary
metric moved with it), and a static list would have quietly shipped the old set.

usage:
  python scripts/collect_inputs.py --workspace <src> --dest <dir> [--tar out.tgz]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path

MANIFESTS = [
    "tumor_board_simulation_test_video_split_60_10_30_v3_deleaked_repaired_xsfixed_reanchored_20260808.jsonl",
    "expert_qa_test_video_split_60_10_30_v3_qascreened_xsfixed_reanchored_rescreened_slidealigned_20260814.jsonl",
    "tumor_board_simulation_full_video_split_60_10_30_v4_deleaked_xsfixed_reanchored_20260808.jsonl",
    "expert_qa_trainval_video_split_60_10_30_v4_qascreened_xsfixed_20260809.jsonl",
]
BENCH_TASK2 = ("OpenTumorBoard_data/benchmark/task2/test/expert_qa_test_video_split_"
               "60_10_30_v3_qascreened_xsfixed_reanchored_rescreened_slidealigned_20260814.jsonl")


def needed(ws: Path) -> tuple[set[str], list[str]]:
    missing, want = [], {
        "OpenTumorBoard/evaluation/results_catalog.json",
        "OpenTumorBoard/evaluation/model_registry.json",
        BENCH_TASK2,
    }
    want |= {f"model_evaluation/manifests/{m}" for m in MANIFESTS}

    cat_p = ws / "OpenTumorBoard/evaluation/results_catalog.json"
    if not cat_p.exists():
        raise SystemExit(f"collect_inputs: no results_catalog.json under {ws}")
    agg = json.loads(cat_p.read_text(encoding="utf-8"))["active"]["aggregate_sources"]
    for kind in ("judge", "generation_cost", "rouge", "bertscore", "task2_by_role"):
        for rel in agg.get(kind, []):
            want.add(f"model_evaluation/{rel}")
            if kind == "judge":
                # Figure 4 reads the build config to check that each auxiliary metric
                # was computed over the same run the judge scored.
                want.add(f"model_evaluation/{rel}".replace(
                    "results/summary.json", "build_config.json"))

    present = set()
    for rel in sorted(want):
        if (ws / rel).exists():
            present.add(rel)
        else:
            missing.append(rel)
    return present, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--tar", type=Path, default=None)
    a = ap.parse_args()

    files, missing = needed(a.workspace)
    total = 0
    for rel in sorted(files):
        src, dst = a.workspace / rel, a.dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        total += os.path.getsize(src)

    print(f"copied {len(files)} files, {total / 1e6:.1f} MB -> {a.dest}")
    if missing:
        # Not fatal: an aggregate source can be listed and not yet synced. The figure
        # scripts refuse on their own if one they need is absent, so say it here and
        # let them be the gate.
        print(f"NOT FOUND ({len(missing)}) - the figure scripts will refuse if they "
              f"need any of these:")
        for rel in missing:
            print(f"  {rel}")

    if a.tar:
        with tarfile.open(a.tar, "w:gz") as tf:
            tf.add(a.dest, arcname=a.dest.name)
        print(f"wrote {a.tar} ({os.path.getsize(a.tar) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
