"""Lepidoptera localizer dataset adapter — reads index.jsonl and produces
(image, target) pairs for torchvision detection or YOLO-style labels.

The canonical training-data format is `index.jsonl` (see
`detector_dataset/src/detector_dataset/index_jsonl.py` for the schema).
This module is intentionally tolerant of optional fields — the eval-set
backfill leaves some FG-only fields blank.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from PIL import Image

SINGLE_CLASS_ID = 0
SINGLE_CLASS_NAME = "arthropod"


@dataclass(frozen=True)
class IndexEntry:
    photo_id: str
    image_key: str
    width: int
    height: int
    bbox_xyxy: tuple[float, float, float, float]
    bbox_area_fraction: float
    bbox_is_square: bool
    split: str
    dataset_name: str

    @classmethod
    def from_jsonl_line(cls, line: str) -> "IndexEntry":
        d = json.loads(line)
        x1, y1, x2, y2 = d["bbox_xyxy"]
        return cls(
            photo_id=d["photo_id"],
            image_key=d["image_key"],
            width=int(d["width"]),
            height=int(d["height"]),
            bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
            bbox_area_fraction=float(d["bbox_area_fraction"]),
            bbox_is_square=bool(d["bbox_is_square"]),
            split=d["split"],
            dataset_name=d["dataset_name"],
        )


def read_index_entries(path: Path, split: str | None = None) -> list[IndexEntry]:
    entries: list[IndexEntry] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = IndexEntry.from_jsonl_line(line)
            if split is not None and entry.split != split:
                continue
            entries.append(entry)
    return entries


class LepsLocalizerDataset(torch.utils.data.Dataset):
    """Single-class arthropod localization dataset, reading from index.jsonl.

    Args:
        index_path: path to `index.jsonl`.
        image_root: directory `image_key` is resolved relative to.
        split: filter to "train" / "val" / "eval" (None = all).
        transform: optional callable applied to (PIL image, target_dict).
    """

    def __init__(
        self,
        index_path: Path,
        image_root: Path,
        *,
        split: str | None = None,
        transform: Callable | None = None,
    ):
        super().__init__()
        self.index_path = Path(index_path)
        self.image_root = Path(image_root)
        self.entries = read_index_entries(self.index_path, split=split)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple:
        entry = self.entries[idx]
        img_path = self.image_root / entry.image_key
        if not img_path.exists():
            # eval-set backfill stored bare basenames; fall back to stem search
            cand = self.image_root / Path(entry.image_key).name
            if cand.exists():
                img_path = cand
            else:
                raise FileNotFoundError(img_path)
        img = Image.open(img_path).convert("RGB")
        boxes = torch.tensor([list(entry.bbox_xyxy)], dtype=torch.float32)
        labels = torch.tensor([SINGLE_CLASS_ID], dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": torch.tensor(
                [
                    (entry.bbox_xyxy[2] - entry.bbox_xyxy[0])
                    * (entry.bbox_xyxy[3] - entry.bbox_xyxy[1])
                ],
                dtype=torch.float32,
            ),
            "iscrowd": torch.zeros((1,), dtype=torch.int64),
        }
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target


def xyxy_to_yolo_normalized(
    bbox_xyxy: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return cx, cy, w, h


def materialize_yolo_labels(
    entries: Iterable[IndexEntry],
    *,
    out_dir: Path,
    class_id: int = SINGLE_CLASS_ID,
) -> int:
    """Write one `.txt` file per entry next to its image (YOLO format).

    YOLO trainer expects `<labels_dir>/<image_stem>.txt` with one line per
    box: `<class> <cx> <cy> <w> <h>` all normalized to [0, 1].
    Returns count of files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for entry in entries:
        cx, cy, w, h = xyxy_to_yolo_normalized(
            entry.bbox_xyxy, entry.width, entry.height
        )
        stem = Path(entry.image_key).stem
        out = out_dir / f"{stem}.txt"
        line = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(class_id, cx, cy, w, h)
        out.write_text(line)
        count += 1
    return count
