"""Unit tests for eval_locked aggregation + markdown formatter."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.localization.eval_locked import (
    EvalSampleResult,
    aggregate,
    evaluate_entries,
    format_markdown_report,
    write_per_sample_jsonl,
)
from src.localization.leps_data import IndexEntry


def _make_entry(
    photo_id: str,
    *,
    image_key: str | None = None,
    bbox=(10.0, 10.0, 90.0, 90.0),
    width: int = 100,
    height: int = 100,
    area_frac: float = 0.64,
    is_square: bool = True,
) -> IndexEntry:
    return IndexEntry(
        photo_id=photo_id,
        image_key=image_key or f"{photo_id}.jpg",
        width=width,
        height=height,
        bbox_xyxy=tuple(bbox),
        bbox_area_fraction=area_frac,
        bbox_is_square=is_square,
        split="eval",
        dataset_name="test",
    )


def _write_jpg(path: Path, w: int = 100, h: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), color=(0, 128, 255)).save(path)


def test_evaluate_entries_perfect_predictor(tmp_path: Path):
    entries = [_make_entry("p1"), _make_entry("p2")]
    image_root = tmp_path / "imgs"
    for e in entries:
        _write_jpg(image_root / e.image_key)

    def perfect(img):
        return [[10.0, 10.0, 90.0, 90.0]]

    results = evaluate_entries(
        entries, image_root=image_root, predict_fn=perfect, iou_threshold=0.5
    )
    assert len(results) == 2
    assert all(r.hit_iou for r in results)
    assert all(r.best_iou == 1.0 for r in results)


def test_evaluate_entries_empty_preds(tmp_path: Path):
    entries = [_make_entry("p1")]
    image_root = tmp_path / "imgs"
    _write_jpg(image_root / entries[0].image_key)

    def none(img):
        return []

    results = evaluate_entries(
        entries, image_root=image_root, predict_fn=none, iou_threshold=0.5
    )
    r = results[0]
    assert r.n_preds == 0
    assert r.best_iou == 0.0
    assert r.hit_iou is False


def test_aggregate_recall_calculations():
    results = [
        EvalSampleResult(
            photo_id="p1",
            dataset_name="t",
            width=100,
            height=100,
            gt_bbox=[0, 0, 10, 10],
            bbox_area_fraction=0.01,
            bbox_is_square=True,
            n_preds=1,
            best_iou=0.9,
            best_containment=0.9,
            hit_iou=True,
            hit_containment=True,
        ),
        EvalSampleResult(
            photo_id="p2",
            dataset_name="t",
            width=100,
            height=100,
            gt_bbox=[0, 0, 10, 10],
            bbox_area_fraction=0.30,
            bbox_is_square=True,
            n_preds=1,
            best_iou=0.2,
            best_containment=0.7,
            hit_iou=False,
            hit_containment=True,
        ),
        EvalSampleResult(
            photo_id="p3",
            dataset_name="t",
            width=100,
            height=100,
            gt_bbox=[0, 0, 10, 10],
            bbox_area_fraction=0.60,
            bbox_is_square=True,
            n_preds=0,
            best_iou=0.0,
            best_containment=0.0,
            hit_iou=False,
            hit_containment=False,
        ),
    ]
    agg = aggregate(results)
    assert agg["n_samples"] == 3
    assert agg["recall_iou"] == 1 / 3
    assert agg["recall_containment"] == 2 / 3
    assert agg["missed_completely"] == 1
    assert len(agg["by_bucket_iou"]) == 5


def test_aggregate_empty_returns_zeros():
    agg = aggregate([])
    assert agg["n_samples"] == 0
    assert agg["recall_iou"] == 0.0


def test_write_per_sample_jsonl(tmp_path: Path):
    results = [
        EvalSampleResult(
            photo_id="p1",
            dataset_name="t",
            width=100,
            height=100,
            gt_bbox=[0, 0, 10, 10],
            bbox_area_fraction=0.5,
            bbox_is_square=True,
            n_preds=1,
            best_iou=0.8,
            best_containment=0.9,
            hit_iou=True,
            hit_containment=True,
        )
    ]
    out = tmp_path / "out" / "samples.jsonl"
    write_per_sample_jsonl(results, out)
    assert out.exists()
    rec = json.loads(out.read_text().strip())
    assert rec["photo_id"] == "p1"
    assert rec["best_iou"] == 0.8


def test_format_markdown_report_contains_metrics():
    agg = {
        "n_samples": 100,
        "recall_iou": 0.42,
        "recall_containment": 0.71,
        "mean_best_iou": 0.55,
        "mean_best_containment": 0.80,
        "missed_completely": 3,
        "by_bucket_iou": [{"lo": 0.0, "hi": 0.05, "n": 20, "hits": 5, "recall": 0.25}],
        "by_bucket_containment": [
            {"lo": 0.0, "hi": 0.05, "n": 20, "hits": 12, "recall": 0.60}
        ],
    }
    md = format_markdown_report(agg, title="YOLO eval", iou_threshold=0.5)
    assert "YOLO eval" in md
    assert "0.4200" in md
    assert "0.7100" in md
    assert "0.05" in md
