"""Unit tests for the leps localizer dataset adapter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.localization.leps_data import (
    IndexEntry,
    LepsLocalizerDataset,
    materialize_yolo_labels,
    read_index_entries,
    xyxy_to_yolo_normalized,
)


def _make_record(
    *,
    photo_id: str,
    width: int = 1000,
    height: int = 800,
    bbox: tuple[float, float, float, float] = (100.0, 100.0, 300.0, 300.0),
    split: str = "train",
    image_key: str | None = None,
) -> dict:
    return {
        "dataset_name": "test",
        "dataset_version": "v1.0.0",
        "data_source": "test",
        "extracted_at": "2026-05-05T00:00:00Z",
        "extract_commit_sha": "x",
        "image_key": image_key or f"images/{photo_id[:2]}/{photo_id}.jpg",
        "sha256": photo_id,
        "photo_id": photo_id,
        "category_id": "c",
        "parents": [],
        "user_id": "u",
        "width": width,
        "height": height,
        "bbox_xyxy": list(bbox),
        "bbox_area_fraction": (bbox[2] - bbox[0])
        * (bbox[3] - bbox[1])
        / (width * height),
        "bbox_is_square": (bbox[2] - bbox[0]) == (bbox[3] - bbox[1]),
        "created_at": "",
        "split": split,
    }


def _write_index(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "index.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


def _write_jpg(path: Path, w: int, h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), color=(128, 64, 32)).save(path)


def test_index_entry_from_jsonl_line():
    rec = _make_record(photo_id="p1")
    entry = IndexEntry.from_jsonl_line(json.dumps(rec))
    assert entry.photo_id == "p1"
    assert entry.bbox_xyxy == (100.0, 100.0, 300.0, 300.0)
    assert entry.width == 1000


def test_read_index_entries_filter_split(tmp_path: Path):
    records = [
        _make_record(photo_id="p1", split="train"),
        _make_record(photo_id="p2", split="val"),
        _make_record(photo_id="p3", split="train"),
    ]
    path = _write_index(tmp_path, records)
    train = read_index_entries(path, split="train")
    val = read_index_entries(path, split="val")
    assert {e.photo_id for e in train} == {"p1", "p3"}
    assert {e.photo_id for e in val} == {"p2"}
    assert len(read_index_entries(path)) == 3


def test_xyxy_to_yolo_normalized():
    cx, cy, w, h = xyxy_to_yolo_normalized((0.0, 0.0, 200.0, 100.0), 1000, 500)
    assert cx == pytest.approx(0.1)
    assert cy == pytest.approx(0.1)
    assert w == pytest.approx(0.2)
    assert h == pytest.approx(0.2)


def test_dataset_returns_image_target(tmp_path: Path):
    records = [_make_record(photo_id="p1", image_key="p1.jpg")]
    index_path = _write_index(tmp_path, records)
    image_root = tmp_path / "images"
    _write_jpg(image_root / "p1.jpg", 1000, 800)
    ds = LepsLocalizerDataset(index_path, image_root)
    assert len(ds) == 1
    img, target = ds[0]
    assert img.size == (1000, 800)
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [0]
    assert target["boxes"][0].tolist() == [100.0, 100.0, 300.0, 300.0]


def test_dataset_falls_back_to_basename(tmp_path: Path):
    """Eval-backfill records use bare basenames in image_key (no `images/aa/` prefix)."""
    records = [_make_record(photo_id="p1", image_key="images/aa/p1.jpg")]
    index_path = _write_index(tmp_path, records)
    image_root = tmp_path / "img-flat"
    _write_jpg(image_root / "p1.jpg", 100, 100)
    ds = LepsLocalizerDataset(index_path, image_root)
    img, _ = ds[0]
    assert img.size == (100, 100)


def test_materialize_yolo_labels(tmp_path: Path):
    records = [
        _make_record(
            photo_id="p1",
            width=1000,
            height=500,
            bbox=(0.0, 0.0, 200.0, 100.0),
            image_key="p1.jpg",
        ),
        _make_record(
            photo_id="p2",
            width=500,
            height=500,
            bbox=(100.0, 100.0, 200.0, 200.0),
            image_key="p2.jpg",
        ),
    ]
    index_path = _write_index(tmp_path, records)
    entries = read_index_entries(index_path)
    out_dir = tmp_path / "labels"
    n = materialize_yolo_labels(entries, out_dir=out_dir)
    assert n == 2
    p1_lbl = (out_dir / "p1.txt").read_text().strip()
    parts = p1_lbl.split()
    assert parts[0] == "0"
    assert float(parts[1]) == pytest.approx(0.1)
    assert float(parts[2]) == pytest.approx(0.1)
    assert float(parts[3]) == pytest.approx(0.2)
    assert float(parts[4]) == pytest.approx(0.2)
