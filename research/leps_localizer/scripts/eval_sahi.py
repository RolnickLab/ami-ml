"""Evaluate a YOLO checkpoint with SAHI tiled inference against a locked
index.jsonl eval set.

Mirrors `eval_on_locked.py` but wraps the model in SAHI's
`get_sliced_prediction`. Lets us compare untiled vs tiled inference on the
same checkpoint.

Usage:
    python research/leps_localizer/scripts/eval_sahi.py \\
        --model /mnt/butterflies-fg-2026-05/runs/yolov11s-fg-2026-05-r2/weights/best.pt \\
        --index /mnt/butterflies-fg-2026-05/metadata/leeds-index.jsonl \\
        --image-root /mnt/butterflies-fg-2026-05/datasets/leeds-butterflies/images \\
        --out-dir /mnt/butterflies-fg-2026-05/eval/yolov11s_r2_leeds_sahi768 \\
        --slice 768 --overlap 0.2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.localization.eval_locked import (  # noqa: E402
    aggregate,
    evaluate_entries,
    format_markdown_report,
    write_per_sample_jsonl,
)
from src.localization.leps_data import read_index_entries  # noqa: E402


def _build_sahi_predictor(
    model_path: Path,
    *,
    conf: float,
    device: str,
    slice_size: int,
    overlap: float,
):
    """Return predict_fn(PIL.Image) -> list[xyxy] using SAHI tiled inference."""
    from sahi import AutoDetectionModel  # noqa: WPS433
    from sahi.predict import get_sliced_prediction  # noqa: WPS433

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(model_path),
        confidence_threshold=conf,
        device=("cuda:" + device) if device.isdigit() else device,
    )

    def predict(img):
        result = get_sliced_prediction(
            img,
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            verbose=0,
        )
        boxes: list[list[float]] = []
        for op in result.object_prediction_list:
            b = op.bbox
            boxes.append([b.minx, b.miny, b.maxx, b.maxy])
        return boxes

    return predict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--slice", dest="slice_size", type=int, default=768)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--title", type=str, default="SAHI tiled eval")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = read_index_entries(args.index, split=args.split)
    if not entries:
        raise SystemExit(f"No entries in {args.index} (split={args.split!r})")

    predict_fn = _build_sahi_predictor(
        args.model,
        conf=args.conf,
        device=args.device,
        slice_size=args.slice_size,
        overlap=args.overlap,
    )

    results = evaluate_entries(
        entries,
        image_root=args.image_root,
        predict_fn=predict_fn,
        iou_threshold=args.iou_threshold,
    )
    agg = aggregate(results)

    samples_path = args.out_dir / "per_sample.jsonl"
    metrics_path = args.out_dir / "metrics.json"
    report_path = args.out_dir / "report.md"

    write_per_sample_jsonl(results, samples_path)
    metrics_path.write_text(json.dumps(agg, indent=2) + "\n")
    extra = {
        "model": str(args.model),
        "index": str(args.index),
        "image_root": str(args.image_root),
        "split": args.split or "all",
        "arch": "sahi-yolo",
        "conf": str(args.conf),
        "slice": str(args.slice_size),
        "overlap": str(args.overlap),
    }
    report_path.write_text(
        format_markdown_report(
            agg, title=args.title, iou_threshold=args.iou_threshold, extra=extra
        )
    )

    print(f"Wrote {samples_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {report_path}")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
