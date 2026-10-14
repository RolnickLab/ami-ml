"""Evaluate a trained YOLO/RT-DETR/FRCNN checkpoint against a locked
index.jsonl eval set.

Default eval set: Leeds butterflies (held-out, different distribution).
Override with --index / --image-root for FG val splits or other indexes.

Usage:
    poetry run python research/leps_localizer/scripts/eval_on_locked.py \
        --model runs/yolov11s-fg-2026-05/weights/best.pt \
        --index <path-to-leeds-index.jsonl> \
        --image-root <path-to-leeds-images> \
        --out-dir research/leps_localizer/eval_outputs/yolov11s_leeds
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


def _build_yolo_predictor(model_path: Path, *, conf: float, device: str):
    """Lazy-import ultralytics so test environments without it still work."""
    from ultralytics import YOLO  # noqa: WPS433

    model = YOLO(str(model_path))

    def predict(img):
        results = model.predict(img, conf=conf, device=device, verbose=False)
        boxes: list[list[float]] = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy
            arr = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else xyxy
            for box in arr.tolist():
                boxes.append(box)
        return boxes

    return predict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="path to .pt checkpoint")
    p.add_argument(
        "--index",
        type=Path,
        required=True,
        help="path to eval index.jsonl",
    )
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--split", type=str, default=None, help="filter to split (default: all)"
    )
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--device", type=str, default="0", help="cuda index or 'cpu'")
    p.add_argument("--title", type=str, default="Localizer eval")
    p.add_argument(
        "--arch",
        type=str,
        default="yolo",
        choices=["yolo"],
        help="prediction backend (only yolo for now; RT-DETR/FRCNN later)",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = read_index_entries(args.index, split=args.split)
    if not entries:
        raise SystemExit(f"No entries in {args.index} (split={args.split!r})")

    if args.arch == "yolo":
        predict_fn = _build_yolo_predictor(
            args.model, conf=args.conf, device=args.device
        )
    else:
        raise SystemExit(f"Unsupported arch: {args.arch}")

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
        "arch": args.arch,
        "conf": str(args.conf),
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
