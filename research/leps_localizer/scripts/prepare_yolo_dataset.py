"""Materialize a YOLO-format dataset directory from the merged FG
train/val index.jsonl.

Source:
    research/leps_localizer/data/butterflies-fg-train-2026-05/index.jsonl

Output (default):
    research/leps_localizer/data/yolo_butterflies-fg-train-2026-05/
        ├── data.yaml
        ├── images/{train,val}/<stem>.jpg  (symlinks)
        └── labels/{train,val}/<stem>.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.localization.leps_data import read_index_entries  # noqa: E402
from src.localization.prepare_yolo import prepare_yolo_dataset  # noqa: E402

DEFAULT_DETECTOR_DATASET_ROOT = (
    "/home/michael/Projects/Fieldguide/chroma-backend/"
    ".claude/worktrees/detector-training/detector_dataset"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--index",
        type=Path,
        default=Path(
            "research/leps_localizer/data/butterflies-fg-train-2026-05/index.jsonl"
        ),
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("research/leps_localizer/data/yolo_butterflies-fg-train-2026-05"),
    )
    p.add_argument(
        "--detector-dataset-root",
        type=Path,
        default=Path(
            os.environ.get("DETECTOR_DATASET_ROOT", DEFAULT_DETECTOR_DATASET_ROOT)
        ),
    )
    p.add_argument(
        "--leeds-root",
        type=Path,
        default=None,
        help="optional: include leeds eval entries by passing the leeds image dir",
    )
    args = p.parse_args()

    train = read_index_entries(args.index, split="train")
    val = read_index_entries(args.index, split="val")
    counts = prepare_yolo_dataset(
        train_entries=train,
        val_entries=val,
        out_root=args.out_root,
        fg_root=args.detector_dataset_root,
        leeds_root=args.leeds_root,
    )
    print(json.dumps(counts.__dict__, indent=2))
    print(f"Wrote {args.out_root}/data.yaml")


if __name__ == "__main__":
    main()
