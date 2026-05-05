"""Tests for FG index.jsonl re-partitioning into train/val."""
from __future__ import annotations

import json
from pathlib import Path

from src.localization.repartition import assign_split_train_val, repartition_indexes


def _make_record(photo_id: str, split: str = "eval") -> dict:
    return {
        "dataset_name": "src",
        "dataset_version": "v1.0.0",
        "data_source": "fieldguide-prod",
        "extracted_at": "2026-05-05T00:00:00Z",
        "extract_commit_sha": "x",
        "image_key": f"{photo_id}.jpg",
        "sha256": "",
        "photo_id": photo_id,
        "category_id": "c",
        "parents": [],
        "user_id": "u",
        "width": 100,
        "height": 100,
        "bbox_xyxy": [0.0, 0.0, 50.0, 50.0],
        "bbox_area_fraction": 0.25,
        "bbox_is_square": True,
        "created_at": "",
        "split": split,
    }


def test_assign_split_deterministic():
    a = assign_split_train_val("abc123", val_fraction=0.1, seed=42)
    b = assign_split_train_val("abc123", val_fraction=0.1, seed=42)
    assert a == b
    assert a in ("train", "val")


def test_assign_split_seed_changes_assignment():
    """Different seeds should produce different splits for at least some IDs."""
    ids = [f"id{i}" for i in range(200)]
    s1 = [assign_split_train_val(i, val_fraction=0.1, seed=1) for i in ids]
    s2 = [assign_split_train_val(i, val_fraction=0.1, seed=2) for i in ids]
    assert s1 != s2


def test_assign_split_distribution_close_to_target():
    ids = [f"id{i}" for i in range(2000)]
    val_count = sum(
        1 for i in ids if assign_split_train_val(i, val_fraction=0.1, seed=42) == "val"
    )
    frac = val_count / len(ids)
    assert 0.07 < frac < 0.13, f"val fraction {frac} far from 0.1"


def test_repartition_writes_merged_index(tmp_path: Path):
    src1 = tmp_path / "a.jsonl"
    src2 = tmp_path / "b.jsonl"
    recs1 = [_make_record(f"a{i}") for i in range(50)]
    recs2 = [_make_record(f"b{i}") for i in range(50)]
    src1.write_text("\n".join(json.dumps(r) for r in recs1) + "\n")
    src2.write_text("\n".join(json.dumps(r) for r in recs2) + "\n")

    out = tmp_path / "merged.jsonl"
    counts = repartition_indexes(
        sources=[src1, src2],
        out_path=out,
        val_fraction=0.1,
        seed=42,
    )
    assert counts["total"] == 100
    assert counts["train"] + counts["val"] == 100
    assert counts["val"] > 0
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 100
    splits = {json.loads(line)["split"] for line in lines}
    assert splits == {"train", "val"}


def test_repartition_preserves_record_fields(tmp_path: Path):
    src = tmp_path / "a.jsonl"
    rec = _make_record("p1")
    src.write_text(json.dumps(rec) + "\n")
    out = tmp_path / "merged.jsonl"
    repartition_indexes(sources=[src], out_path=out, val_fraction=0.1, seed=42)
    merged = json.loads(out.read_text().strip())
    assert merged["photo_id"] == "p1"
    assert merged["bbox_xyxy"] == [0.0, 0.0, 50.0, 50.0]
    assert merged["split"] in ("train", "val")
    assert merged["dataset_name"] == "src"


def test_repartition_dedup_by_photo_id(tmp_path: Path):
    """If same photo_id appears in multiple source files, keep only one."""
    src1 = tmp_path / "a.jsonl"
    src2 = tmp_path / "b.jsonl"
    src1.write_text(json.dumps(_make_record("dup")) + "\n")
    src2.write_text(json.dumps(_make_record("dup")) + "\n")
    out = tmp_path / "merged.jsonl"
    counts = repartition_indexes(
        sources=[src1, src2], out_path=out, val_fraction=0.1, seed=42
    )
    assert counts["total"] == 1
    assert counts["duplicates_dropped"] == 1
