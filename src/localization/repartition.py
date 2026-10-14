"""Re-partition FG eval-locked index.jsonl files into a single train/val
training index.

The 3 FG butterfly eval sets (`leps-butterflies-500`,
`leps-butterflies-medsmall-1000`, `leps-butterflies-small-1000`) were
originally locked at `split=eval` for evaluation. For the
`leps-localizer-training` work we repurpose them as training data and use
the Leeds dataset (different distribution) as held-out eval.

Split assignment is deterministic via `md5(seed:photo_id)` so that the
same photo always lands in the same split across re-runs and across
machines. De-duplication on `photo_id` keeps a single record per photo
even if a photo appears in two source indexes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def assign_split_train_val(
    photo_id: str, *, val_fraction: float = 0.1, seed: int = 42
) -> str:
    """Deterministic train/val assignment for a photo.

    Hashes `<seed>:<photo_id>` with md5, takes the first 16 hex chars as
    a 64-bit int, normalizes to [0, 1), and returns "val" if below
    `val_fraction` else "train".
    """
    key = str(seed) + ":" + photo_id
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    bucket = int(digest[0:16], 16) / float(1 << 64)
    return "val" if bucket < val_fraction else "train"


def repartition_indexes(
    *,
    sources: Iterable[Path],
    out_path: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    """Merge N source index.jsonl files into one with train/val splits.

    Records are de-duplicated on `photo_id` (first occurrence wins).
    Each surviving record's `split` field is overwritten by
    `assign_split_train_val`.

    Returns a counts dict: total / train / val / duplicates_dropped.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    train = 0
    val = 0
    dups = 0

    with out_path.open("w") as fout:
        for src in sources:
            src = Path(src)
            with src.open() as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    pid = rec["photo_id"]
                    if pid in seen:
                        dups += 1
                        continue
                    seen.add(pid)
                    split = assign_split_train_val(
                        pid, val_fraction=val_fraction, seed=seed
                    )
                    rec["split"] = split
                    if split == "train":
                        train += 1
                    else:
                        val += 1
                    fout.write(json.dumps(rec) + "\n")

    return {
        "total": train + val,
        "train": train,
        "val": val,
        "duplicates_dropped": dups,
    }
