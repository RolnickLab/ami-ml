"""Re-partition the 3 FG butterfly eval-locked index.jsonl files into a
single train/val index for localizer training.

The source index files come from a separate dataset export of Fieldguide
photos with user-drawn crop boxes. Pass the directory that holds them with
--detector-dataset-root (or the DETECTOR_DATASET_ROOT environment variable).

Usage:
    uv run python research/leps_localizer/scripts/repartition_fg.py \
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--detector-dataset-root",
        default=os.environ.get("DETECTOR_DATASET_ROOT"),
        help="root of the source dataset export (or set DETECTOR_DATASET_ROOT)",
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
    if not args.detector_dataset_root:
        p.error("--detector-dataset-root or DETECTOR_DATASET_ROOT is required")

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
