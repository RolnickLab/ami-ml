"""Re-partition the 3 FG butterfly eval-locked index.jsonl files into a
single train/val index for localizer training.

The source index files were emitted by detector_dataset (chroma-backend,
private) and are mirrored locally on the workstation. Update SOURCES
below if you've cloned them somewhere else.

Usage:
    poetry run python research/leps_localizer/scripts/repartition_fg.py \
        --out research/leps_localizer/data/butterflies-fg-train-2026-05/index.jsonl
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

from src.localization.repartition import repartition_indexes  # noqa: E402

DEFAULT_SOURCES = [
    "datasets/arthropoda/leps-butterflies-500/index.jsonl",
    "datasets/arthropoda/leps-butterflies-medsmall-1000/index.jsonl",
    "datasets/arthropoda/leps-butterflies-small-1000/index.jsonl",
]

DEFAULT_DETECTOR_DATASET_ROOT = (
    "/home/michael/Projects/Fieldguide/chroma-backend/"
    ".claude/worktrees/detector-training/detector_dataset"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--detector-dataset-root",
        default=os.environ.get("DETECTOR_DATASET_ROOT", DEFAULT_DETECTOR_DATASET_ROOT),
        help="root of the chroma-backend detector_dataset checkout",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "research/leps_localizer/data/butterflies-fg-train-2026-05/index.jsonl"
        ),
    )
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260505)
    args = p.parse_args()

    root = Path(args.detector_dataset_root)
    sources = [root / s for s in DEFAULT_SOURCES]
    missing = [str(s) for s in sources if not s.exists()]
    if missing:
        raise SystemExit("Missing source index files:\n  " + "\n  ".join(missing))

    counts = repartition_indexes(
        sources=sources,
        out_path=args.out,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(json.dumps(counts, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
