"""Unit tests for prepare_yolo dataset materializer."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.localization.leps_data import IndexEntry
from src.localization.prepare_yolo import (
    PrepareCounts,
    prepare_split,
    prepare_yolo_dataset,
    resolve_image_path,
    write_data_yaml,
)


def _entry(
    photo_id: str,
    *,
    dataset_name: str = "leps-butterflies-500",
    image_key: str | None = None,
    bbox=(10.0, 10.0, 90.0, 90.0),
    width: int = 100,
    height: int = 100,
    split: str = "train",
) -> IndexEntry:
    if image_key is None:
        image_key = f"{photo_id}.jpg"
    return IndexEntry(
        photo_id=photo_id,
        image_key=image_key,
        width=width,
        height=height,
        bbox_xyxy=tuple(bbox),
        bbox_area_fraction=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (width * height),
        bbox_is_square=True,
        split=split,
        dataset_name=dataset_name,
    )


def _make_fg_image(fg_root: Path, dataset_name: str, image_key: str) -> Path:
    p = fg_root / "datasets" / "arthropoda" / dataset_name / "images" / image_key
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color=(0, 0, 255)).save(p)
    return p


def test_resolve_fg_image_path(tmp_path: Path):
    e = _entry("p1", dataset_name="leps-butterflies-500")
    p = resolve_image_path(e, fg_root=tmp_path)
    assert p is not None
    assert "leps-butterflies-500" in str(p)
    assert p.name == "p1.jpg"


def test_resolve_leeds_image_path(tmp_path: Path):
    e = _entry(
        "0010001",
        dataset_name="leeds-butterflies",
        image_key="images/0010001.png",
    )
    p = resolve_image_path(e, fg_root=tmp_path, leeds_root=tmp_path / "leeds")
    assert p is not None
    assert p.name == "0010001.png"


def test_resolve_unknown_dataset(tmp_path: Path):
    e = _entry("p1", dataset_name="unknown")
    assert resolve_image_path(e, fg_root=tmp_path) is None


def test_prepare_split_creates_symlinks_and_labels(tmp_path: Path):
    fg = tmp_path / "fg"
    _make_fg_image(fg, "leps-butterflies-500", "p1.jpg")
    _make_fg_image(fg, "leps-butterflies-500", "p2.jpg")
    entries = [_entry("p1"), _entry("p2")]
    images_dir = tmp_path / "out" / "images" / "train"
    labels_dir = tmp_path / "out" / "labels" / "train"
    linked, missing = prepare_split(
        entries,
        images_dir=images_dir,
        labels_dir=labels_dir,
        fg_root=fg,
    )
    assert linked == 2
    assert missing == 0
    assert (images_dir / "p1.jpg").is_symlink()
    assert (images_dir / "p2.jpg").is_symlink()
    label_text = (labels_dir / "p1.txt").read_text().strip()
    parts = label_text.split()
    assert parts[0] == "0"
    assert len(parts) == 5


def test_prepare_split_skips_missing_image(tmp_path: Path):
    fg = tmp_path / "fg"
    # Only p1 image exists; p2 is missing
    _make_fg_image(fg, "leps-butterflies-500", "p1.jpg")
    entries = [_entry("p1"), _entry("p2")]
    images_dir = tmp_path / "out" / "images" / "train"
    labels_dir = tmp_path / "out" / "labels" / "train"
    linked, missing = prepare_split(
        entries,
        images_dir=images_dir,
        labels_dir=labels_dir,
        fg_root=fg,
    )
    assert linked == 1
    assert missing == 1
    assert (labels_dir / "p1.txt").exists()
    assert not (labels_dir / "p2.txt").exists()


def test_write_data_yaml(tmp_path: Path):
    out = tmp_path / "data.yaml"
    write_data_yaml(out, root=tmp_path)
    body = out.read_text()
    assert "nc: 1" in body
    assert "arthropod" in body
    assert "images/train" in body


def test_prepare_yolo_dataset_full(tmp_path: Path):
    fg = tmp_path / "fg"
    _make_fg_image(fg, "leps-butterflies-500", "p1.jpg")
    _make_fg_image(fg, "leps-butterflies-500", "p2.jpg")
    out_root = tmp_path / "yolo_ds"
    counts = prepare_yolo_dataset(
        train_entries=[_entry("p1", split="train")],
        val_entries=[_entry("p2", split="val")],
        out_root=out_root,
        fg_root=fg,
    )
    assert isinstance(counts, PrepareCounts)
    assert counts.train == 1
    assert counts.val == 1
    assert (out_root / "data.yaml").exists()
    assert (out_root / "images" / "train" / "p1.jpg").is_symlink()
    assert (out_root / "images" / "val" / "p2.jpg").is_symlink()
    assert (out_root / "labels" / "train" / "p1.txt").exists()
    assert (out_root / "labels" / "val" / "p2.txt").exists()
