"""Evaluate a single-subject arthropod detector against a locked
index.jsonl eval set.

Model-agnostic by design: callers pass a `predict_fn(image) -> list[xyxy]`
so we can plug Ultralytics YOLO, RT-DETR, or torchvision FRCNN with the
same eval loop.

Outputs:
    - per-sample JSONL (photo_id, gt, best pred score, hit flags, ...)
    - aggregate metrics dict (recall@iou, recall@containment, per-bucket)
    - markdown report (human-readable summary)

Single-subject reduction: each FG / Leeds image has exactly one
ground-truth bbox. `best_of_n_match` picks the model's best pred and
flags hit if its IoU (or containment for square-gt FG sets) >= threshold.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from PIL import Image

from src.localization.leps_data import IndexEntry
from src.localization.metrics import (
    DEFAULT_AREA_FRAC_BUCKETS,
    best_of_n_match,
    recall_by_bucket,
)

PredictFn = Callable[[Image.Image], Sequence[Sequence[float]]]


@dataclass
class EvalSampleResult:
    photo_id: str
    dataset_name: str
    width: int
    height: int
    gt_bbox: list[float]
    bbox_area_fraction: float
    bbox_is_square: bool
    n_preds: int
    best_iou: float
    best_containment: float
    hit_iou: bool
    hit_containment: bool


def evaluate_entries(
    entries: Iterable[IndexEntry],
    *,
    image_root: Path,
    predict_fn: PredictFn,
    iou_threshold: float = 0.5,
) -> list[EvalSampleResult]:
    """Run `predict_fn` against each entry and collect per-sample results."""
    image_root = Path(image_root)
    results: list[EvalSampleResult] = []
    for entry in entries:
        img_path = image_root / entry.image_key
        if not img_path.exists():
            cand = image_root / Path(entry.image_key).name
            if cand.exists():
                img_path = cand
            else:
                raise FileNotFoundError(img_path)
        with Image.open(img_path) as raw:
            img = raw.convert("RGB")
            preds = list(predict_fn(img))
        gt = list(entry.bbox_xyxy)
        hit_iou, best_iou, _ = best_of_n_match(
            preds, gt, iou_threshold=iou_threshold, use_containment=False
        )
        hit_cont, best_cont, _ = best_of_n_match(
            preds, gt, iou_threshold=iou_threshold, use_containment=True
        )
        results.append(
            EvalSampleResult(
                photo_id=entry.photo_id,
                dataset_name=entry.dataset_name,
                width=entry.width,
                height=entry.height,
                gt_bbox=gt,
                bbox_area_fraction=entry.bbox_area_fraction,
                bbox_is_square=entry.bbox_is_square,
                n_preds=len(preds),
                best_iou=best_iou,
                best_containment=best_cont,
                hit_iou=hit_iou,
                hit_containment=hit_cont,
            )
        )
    return results


def aggregate(results: Sequence[EvalSampleResult]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {
            "n_samples": 0,
            "recall_iou": 0.0,
            "recall_containment": 0.0,
            "mean_best_iou": 0.0,
            "mean_best_containment": 0.0,
            "missed_completely": 0,
            "by_bucket_iou": [],
            "by_bucket_containment": [],
        }
    recall_iou = sum(1 for r in results if r.hit_iou) / n
    recall_cont = sum(1 for r in results if r.hit_containment) / n
    missed = sum(1 for r in results if r.n_preds == 0)
    mean_iou = sum(r.best_iou for r in results) / n
    mean_cont = sum(r.best_containment for r in results) / n
    samples_iou = [(r.bbox_area_fraction, r.hit_iou) for r in results]
    samples_cont = [(r.bbox_area_fraction, r.hit_containment) for r in results]
    return {
        "n_samples": n,
        "recall_iou": recall_iou,
        "recall_containment": recall_cont,
        "mean_best_iou": mean_iou,
        "mean_best_containment": mean_cont,
        "missed_completely": missed,
        "by_bucket_iou": recall_by_bucket(
            samples_iou, buckets=DEFAULT_AREA_FRAC_BUCKETS
        ),
        "by_bucket_containment": recall_by_bucket(
            samples_cont, buckets=DEFAULT_AREA_FRAC_BUCKETS
        ),
    }


def write_per_sample_jsonl(results: Sequence[EvalSampleResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")


def format_markdown_report(
    aggregates: dict[str, Any],
    *,
    title: str,
    iou_threshold: float = 0.5,
    extra: dict[str, str] | None = None,
) -> str:
    lines: list[str] = [f"# {title}", ""]
    if extra:
        for k, v in extra.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    lines.extend(
        [
            f"- **n_samples**: {aggregates['n_samples']}",
            f"- **iou_threshold**: {iou_threshold}",
            "",
            "## Headline",
            "",
            "| metric | value |",
            "|---|---|",
            "| recall (IoU) | {:.4f} |".format(aggregates["recall_iou"]),
            "| recall (containment) | {:.4f} |".format(
                aggregates["recall_containment"]
            ),
            "| mean best IoU | {:.4f} |".format(aggregates["mean_best_iou"]),
            "| mean best containment | {:.4f} |".format(
                aggregates["mean_best_containment"]
            ),
            "| missed (no preds) | {} |".format(aggregates["missed_completely"]),
            "",
            "## Recall by bbox-area-fraction bucket (IoU)",
            "",
            "| lo | hi | n | hits | recall |",
            "|---|---|---|---|---|",
        ]
    )
    for b in aggregates["by_bucket_iou"]:
        lines.append(
            "| {:.2f} | {:.2f} | {} | {} | {:.4f} |".format(
                b["lo"], b["hi"], b["n"], b["hits"], b["recall"]
            )
        )
    lines.extend(
        [
            "",
            "## Recall by bbox-area-fraction bucket (containment)",
            "",
            "| lo | hi | n | hits | recall |",
            "|---|---|---|---|---|",
        ]
    )
    for b in aggregates["by_bucket_containment"]:
        lines.append(
            "| {:.2f} | {:.2f} | {} | {} | {:.4f} |".format(
                b["lo"], b["hi"], b["n"], b["hits"], b["recall"]
            )
        )
    lines.append("")
    return "\n".join(lines)
