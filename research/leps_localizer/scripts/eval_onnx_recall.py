"""Evaluate a YOLO26 NMS-free ONNX export on the FG val split, reporting
recall@IoU=0.5 globally and per size bin.

Bins by ratio of GT box area to image area (matches METHODS_AND_RESULTS):
    small  : ratio < 0.077
    mid    : 0.077 <= ratio < 0.328
    large  : ratio >= 0.328

Assumes:
    - ONNX input is [1, 3, imgsz, imgsz], float32, RGB 0-1, letterboxed
      with gray (114) padding (Ultralytics convention).
    - Output is [1, max_dets, 6] = (x1, y1, x2, y2, conf, cls) in pixel
      coords relative to the letterboxed input.

Usage:
    uv run python research/leps_localizer/scripts/eval_onnx_recall.py \\
        --onnx path/to/best.onnx \\
        --images /mnt/butterflies-fg-2026-05/yolo/images/val \\
        --labels /mnt/butterflies-fg-2026-05/yolo/labels/val \\
        --imgsz 640 --conf 0.25 \\
        --out research/leps_localizer/eval_outputs/yolo26s_fp32_recall.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def letterbox(
    img: np.ndarray, sz: int, pad: int = 114
) -> tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    r = min(sz / w, sz / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    px, py = (sz - nw) // 2, (sz - nh) // 2
    canvas = np.full((sz, sz, 3), pad, dtype=np.uint8)
    resized = np.asarray(
        Image.fromarray(img).resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.uint8
    )
    canvas[py : py + nh, px : px + nw] = resized
    return canvas, r, px, py


def preprocess(path: Path, sz: int) -> tuple[np.ndarray, dict, tuple[int, int]]:
    img = np.asarray(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    lb, r, px, py = letterbox(img, sz)
    tensor = lb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return tensor, {"scale": r, "padX": px, "padY": py, "srcW": w, "srcH": h}, (h, w)


def read_yolo_label(path: Path) -> list[tuple[float, float, float, float]]:
    """Read YOLO-format label (class cx cy w h, all normalized 0-1). Returns
    [(x1, y1, x2, y2)] in normalized image coords."""
    boxes: list[tuple[float, float, float, float]] = []
    if not path.exists():
        return boxes
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = (float(x) for x in parts[:5])
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU. a:[N,4], b:[M,4]. Returns [N,M]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + bb[None, :] - inter + 1e-9
    return inter / union


def postprocess(out: np.ndarray, meta: dict, conf_th: float) -> np.ndarray:
    """Decode [1, N, 6] → [K, 4] xyxy in original-image coords (filtered by conf)."""
    rows = out[0]
    keep = rows[:, 4] >= conf_th
    rows = rows[keep]
    if len(rows) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = rows[:, :4].copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - meta["padX"]) / meta["scale"]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - meta["padY"]) / meta["scale"]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, meta["srcW"])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, meta["srcH"])
    return boxes.astype(np.float32)


def size_bin(ratio: float, small: float, mid: float) -> str:
    if ratio < small:
        return "small"
    if ratio < mid:
        return "mid"
    return "large"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True, help="val images dir")
    p.add_argument("--labels", type=Path, required=True, help="val labels dir")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5, help="match IoU threshold")
    p.add_argument("--small-th", type=float, default=0.077)
    p.add_argument("--mid-th", type=float, default=0.328)
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    args = p.parse_args()

    img_paths = sorted(
        list(args.images.glob("*.jpg"))
        + list(args.images.glob("*.jpeg"))
        + list(args.images.glob("*.png"))
    )
    if args.limit:
        img_paths = img_paths[: args.limit]
    if not img_paths:
        raise SystemExit(f"no images in {args.images}")

    sess = ort.InferenceSession(str(args.onnx), providers=args.providers)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    bins = {"small": [0, 0], "mid": [0, 0], "large": [0, 0]}  # [matched, total]
    n_gt = 0
    n_matched = 0
    n_pred = 0
    t_inf = 0.0
    t0 = time.monotonic()

    for i, ip in enumerate(img_paths):
        lp = args.labels / (ip.stem + ".txt")
        gt_norm = read_yolo_label(lp)
        tensor, meta, (h, w) = preprocess(ip, args.imgsz)

        ti = time.monotonic()
        out = sess.run([out_name], {in_name: tensor})[0]
        t_inf += time.monotonic() - ti
        out_arr = np.asarray(out, dtype=np.float32)

        preds = postprocess(out_arr, meta, args.conf)
        n_pred += len(preds)

        if not gt_norm:
            continue
        gt_pix = np.array(
            [(b[0] * w, b[1] * h, b[2] * w, b[3] * h) for b in gt_norm],
            dtype=np.float32,
        )
        ious = iou_xyxy(gt_pix, preds) if len(preds) else np.zeros((len(gt_pix), 0))
        matched_per_gt = (
            ious.max(axis=1) >= args.iou if ious.size else np.zeros(len(gt_pix), bool)
        )

        for j, gt in enumerate(gt_pix):
            n_gt += 1
            ratio = ((gt[2] - gt[0]) * (gt[3] - gt[1])) / (w * h)
            b = size_bin(ratio, args.small_th, args.mid_th)
            bins[b][1] += 1
            if matched_per_gt[j]:
                bins[b][0] += 1
                n_matched += 1

        if (i + 1) % 200 == 0:
            print(
                f"  {i + 1}/{len(img_paths)}  "
                f"recall_so_far={n_matched / max(n_gt, 1):.4f}"
            )

    elapsed = time.monotonic() - t0
    onnx_size_mb = args.onnx.stat().st_size / (1024 * 1024)

    metrics = {
        "onnx": str(args.onnx),
        "onnx_size_mb": round(onnx_size_mb, 2),
        "imgsz": args.imgsz,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "n_images": len(img_paths),
        "n_gt": n_gt,
        "n_predictions": n_pred,
        "recall_overall": round(n_matched / max(n_gt, 1), 4),
        "recall_by_size": {
            k: {
                "matched": v[0],
                "total": v[1],
                "recall": round(v[0] / v[1], 4) if v[1] else None,
            }
            for k, v in bins.items()
        },
        "size_thresholds": {"small_lt": args.small_th, "mid_lt": args.mid_th},
        "inference_ms_per_image": round(1000 * t_inf / len(img_paths), 2),
        "wall_seconds": round(elapsed, 1),
        "providers": args.providers,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
