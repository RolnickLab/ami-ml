# flake8: noqa: E221, E231, E702
"""Bulk localizer inference over a parquet worklist shard.

Reads parquet rows (one row = one training_images record), resolves each to a
JPEG on /mnt/squash-0, runs a localizer (YOLO26-s or DEIMv2-S), and writes one
NDJSON row per input image to --out. Designed to be the workhorse for the
10.7M training_images BQ table.

Output NDJSON schema matches the planned `localizer_eval_results` BQ table
(see .claude/skills/bigquery-leps/procedures/merge-eval-into-training.md):

    {
      "dataset_source_uuid": str,
      "model_id": str,
      "run_id": str,
      "run_at": ISO8601 UTC,
      "arch": "yolo26s" | "deimv2-s",
      "imgsz": int,
      "predicted_bbox_xyxy": [x1,y1,x2,y2] or [],
      "score": float or null,
      "image_width": int or null,
      "image_height": int or null,
      "n_detections_above_thresh": int
    }

Every input row produces exactly one output row. Missing-file or
inference-failure rows still emit a zero-detection row so the downstream
MERGE can distinguish "model found nothing" from "not yet processed".

Resume: if --out already exists and its line count equals the parquet row
count, the script exits 0 without re-running.

Usage (YOLO26-s):

    .venv/bin/python research/leps_localizer/scripts/batch_localize.py \\
        --shard shard-0000.parquet \\
        --out   shard-0000.ndjson \\
        --weights /mnt/butterflies-fg-2026-05/runs/yolo26s-fg-2026-05-v2-2/weights/best.pt \\
        --arch yolo26s \\
        --imgsz 1280 --batch 32 \\
        --image-root /mnt/squash-0 \\
        --run-id 2026-05-12-yolo26s-v2-canary \\
        --model-id yolo26s_v2
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import traceback
from pathlib import Path


def _now_iso_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_parquet_rows(path: Path) -> list[dict]:
    """Read parquet shard to a list of dicts. Worklist shards are small (<100MB)."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    cols = table.column_names
    required = {"dataset_source_uuid", "relative_local_path"}
    missing = required - set(cols)
    if missing:
        raise SystemExit(
            f"Parquet {path} missing required cols: {missing}. Got: {cols}"
        )
    return table.to_pylist()


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as fh:
        for _ in fh:
            n += 1
    return n


def _emit_row(
    fh,
    *,
    dataset_source_uuid: str,
    model_id: str,
    run_id: str,
    run_at: str,
    arch: str,
    imgsz: int,
    bbox: list[float],
    score: float | None,
    image_width: int | None,
    image_height: int | None,
    n_detections: int,
) -> None:
    row = {
        "dataset_source_uuid": dataset_source_uuid,
        "model_id": model_id,
        "run_id": run_id,
        "run_at": run_at,
        "arch": arch,
        "imgsz": imgsz,
        "predicted_bbox_xyxy": bbox,
        "score": score,
        "image_width": image_width,
        "image_height": image_height,
        "n_detections_above_thresh": n_detections,
    }
    fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# YOLO predictor
# ---------------------------------------------------------------------------


def _build_yolo_predictor(weights: Path, device: str, imgsz: int, score_thresh: float):
    """Returns a callable: list[str|np.ndarray] -> list[(bbox|None, score|None, n_kept)]."""
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def predict_batch(image_paths: list[str]):
        # ultralytics handles file paths directly; passing list gets batched.
        results = model.predict(
            image_paths,
            conf=score_thresh,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        out = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                out.append((None, None, 0))
                continue
            xyxy = r.boxes.xyxy
            conf = r.boxes.conf
            xyxy_np = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else xyxy
            conf_np = conf.cpu().numpy() if hasattr(conf, "cpu") else conf
            # top-1 by score
            top_idx = int(conf_np.argmax())
            top_box = xyxy_np[top_idx].tolist()
            top_conf = float(conf_np[top_idx])
            out.append((top_box, top_conf, int(len(conf_np))))
        return out

    return predict_batch


# ---------------------------------------------------------------------------
# DEIMv2 predictor (built when needed; requires DEIMv2 venv + repo)
# ---------------------------------------------------------------------------


def _build_deimv2_predictor(
    weights: Path,
    device: str,
    imgsz: int,
    score_thresh: float,
    deimv2_root: Path,
    deimv2_config: Path,
):
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image

    if str(deimv2_root) not in sys.path:
        sys.path.insert(0, str(deimv2_root))
    from engine.core import YAMLConfig  # noqa: WPS433

    cfg = YAMLConfig(str(deimv2_config), resume=str(weights))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    eval_size = cfg.yaml_cfg["eval_spatial_size"]
    if isinstance(eval_size, int):
        eval_size = (eval_size, eval_size)
    else:
        eval_size = tuple(eval_size)
    vit_backbone = bool(cfg.yaml_cfg.get("DINOv3STAs", False))

    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = DeployModel().to(device).eval()
    if vit_backbone:
        transforms = T.Compose(
            [
                T.Resize(eval_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transforms = T.Compose([T.Resize(eval_size), T.ToTensor()])

    @torch.inference_mode()
    def predict_batch(image_paths: list[str]):
        out = []
        # DEIMv2 currently single-image-forward in eval_deimv2_on_locked.py.
        # Kept the same here for simplicity; can batch later.
        for ip in image_paths:
            try:
                img = Image.open(ip).convert("RGB")
            except Exception:
                out.append((None, None, 0))
                continue
            w, h = img.size
            orig_size = torch.tensor([[w, h]], device=device)
            im = transforms(img).unsqueeze(0).to(device)
            _labels, boxes, scores = model(im, orig_size)
            scr = scores[0].detach().cpu().numpy()
            box = boxes[0].detach().cpu().numpy()
            keep = scr >= score_thresh
            n_kept = int(keep.sum())
            if n_kept == 0:
                out.append((None, None, 0))
                continue
            top_idx = int(scr[keep].argmax())
            kept_boxes = box[keep]
            kept_scores = scr[keep]
            out.append(
                (kept_boxes[top_idx].tolist(), float(kept_scores[top_idx]), n_kept)
            )
        return out

    return predict_batch


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard", type=Path, required=True, help="input parquet shard")
    p.add_argument("--out", type=Path, required=True, help="output NDJSON path")
    p.add_argument("--weights", type=Path, required=True, help="model weights .pt/.pth")
    p.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["yolo26s", "deimv2-s"],
    )
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument(
        "--image-root",
        type=Path,
        default=Path("/mnt/squash-0"),
        help=(
            "Base path for relative_local_path. Used as-is when --multi-squash "
            "is not set."
        ),
    )
    p.add_argument(
        "--multi-squash",
        action="store_true",
        help=(
            "If set, ignore --image-root and pick /mnt/squash-{photo_id %% 10} "
            "per row. Requires all 10 ami-squashfs@N units mounted; see the "
            "session benchmark doc for setup."
        ),
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--model-id", type=str, required=True)
    p.add_argument("--score-thresh", type=float, default=0.25)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=None, help="cap rows processed (debug)")
    p.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="print rolling throughput every N images",
    )
    # DEIMv2-only knobs
    p.add_argument("--deimv2-root", type=Path, default=None)
    p.add_argument("--deimv2-config", type=Path, default=None)
    args = p.parse_args()

    if not args.shard.exists():
        raise SystemExit(f"--shard not found: {args.shard}")
    if not args.weights.exists():
        raise SystemExit(f"--weights not found: {args.weights}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_parquet_rows(args.shard)
    if args.limit:
        rows = rows[: args.limit]
    n_rows = len(rows)
    print(f"[batch_localize] shard rows: {n_rows}", flush=True)

    # Resume: skip if output already complete.
    if args.out.exists():
        existing = _count_lines(args.out)
        if existing == n_rows:
            print(
                f"[batch_localize] {args.out} already has {existing} lines; skipping",
                flush=True,
            )
            return
        else:
            print(
                f"[batch_localize] {args.out} has {existing} lines (expected {n_rows});"
                " overwriting",
                flush=True,
            )

    # Build predictor.
    if args.arch == "yolo26s":
        predict_batch = _build_yolo_predictor(
            args.weights, args.device, args.imgsz, args.score_thresh
        )
    else:  # deimv2-s
        if not args.deimv2_root or not args.deimv2_config:
            raise SystemExit(
                "--arch deimv2-s requires --deimv2-root and --deimv2-config"
            )
        predict_batch = _build_deimv2_predictor(
            args.weights,
            args.device,
            args.imgsz,
            args.score_thresh,
            args.deimv2_root,
            args.deimv2_config,
        )

    def resolve_disk_path(r: dict) -> Path | None:
        rel = r.get("relative_local_path")
        if not rel:
            return None
        if args.multi_squash:
            pid = r.get("photo_id")
            if pid is None:
                return None
            root = Path(f"/mnt/squash-{int(pid) % 10}")
        else:
            root = args.image_root
        return root / rel

    # Warm up (1 small batch). Don't count this in throughput.
    print(
        f"[batch_localize] warming up arch={args.arch} imgsz={args.imgsz}", flush=True
    )
    warmup_paths: list[str] = []
    for r in rows[: min(2, n_rows)]:
        p_on_disk = resolve_disk_path(r)
        if p_on_disk is not None and p_on_disk.exists():
            warmup_paths.append(str(p_on_disk))
    if warmup_paths:
        try:
            predict_batch(warmup_paths)
        except Exception as e:
            print(f"[batch_localize] warmup failed: {e!r}", flush=True)

    run_at = _now_iso_utc()
    n_missing = 0
    n_errors = 0
    n_detected = 0
    n_zero_det = 0
    t0 = time.perf_counter()
    last_log_t = t0
    last_log_i = 0
    processed = 0

    with args.out.open("w") as fh:
        # Walk rows in batches.
        for batch_start in range(0, n_rows, args.batch):
            batch_rows = rows[batch_start : batch_start + args.batch]
            # Resolve paths; bucket missing-file rows out of the prediction batch.
            paths_to_predict: list[str] = []
            path_idxs: list[int] = []
            for i, r in enumerate(batch_rows):
                disk = resolve_disk_path(r)
                if disk is None or not disk.exists():
                    continue
                paths_to_predict.append(str(disk))
                path_idxs.append(i)

            preds_by_idx: dict[int, tuple] = {}
            if paths_to_predict:
                try:
                    preds = predict_batch(paths_to_predict)
                    for idx, pred in zip(path_idxs, preds):
                        preds_by_idx[idx] = pred
                except Exception as e:
                    n_errors += len(paths_to_predict)
                    print(
                        f"[batch_localize] batch error: {e!r}; emitting zero-det rows",
                        file=sys.stderr,
                        flush=True,
                    )
                    traceback.print_exc(file=sys.stderr)

            for i, r in enumerate(batch_rows):
                disk = resolve_disk_path(r)
                img_w = r.get("image_width")
                img_h = r.get("image_height")
                if disk is None or not disk.exists():
                    n_missing += 1
                    _emit_row(
                        fh,
                        dataset_source_uuid=r["dataset_source_uuid"],
                        model_id=args.model_id,
                        run_id=args.run_id,
                        run_at=run_at,
                        arch=args.arch,
                        imgsz=args.imgsz,
                        bbox=[],
                        score=None,
                        image_width=img_w,
                        image_height=img_h,
                        n_detections=0,
                    )
                    continue
                if i in preds_by_idx:
                    bbox, score, n_kept = preds_by_idx[i]
                    if bbox is None:
                        n_zero_det += 1
                        _emit_row(
                            fh,
                            dataset_source_uuid=r["dataset_source_uuid"],
                            model_id=args.model_id,
                            run_id=args.run_id,
                            run_at=run_at,
                            arch=args.arch,
                            imgsz=args.imgsz,
                            bbox=[],
                            score=None,
                            image_width=img_w,
                            image_height=img_h,
                            n_detections=0,
                        )
                    else:
                        n_detected += 1
                        _emit_row(
                            fh,
                            dataset_source_uuid=r["dataset_source_uuid"],
                            model_id=args.model_id,
                            run_id=args.run_id,
                            run_at=run_at,
                            arch=args.arch,
                            imgsz=args.imgsz,
                            bbox=bbox,
                            score=score,
                            image_width=img_w,
                            image_height=img_h,
                            n_detections=n_kept,
                        )
                else:
                    # Path existed but batch errored.
                    _emit_row(
                        fh,
                        dataset_source_uuid=r["dataset_source_uuid"],
                        model_id=args.model_id,
                        run_id=args.run_id,
                        run_at=run_at,
                        arch=args.arch,
                        imgsz=args.imgsz,
                        bbox=[],
                        score=None,
                        image_width=img_w,
                        image_height=img_h,
                        n_detections=0,
                    )

            processed += len(batch_rows)
            # Rolling throughput.
            if processed - last_log_i >= args.progress_every or processed == n_rows:
                now = time.perf_counter()
                d_n = processed - last_log_i
                d_t = now - last_log_t
                rate_recent = d_n / d_t if d_t > 0 else 0
                rate_overall = processed / (now - t0) if (now - t0) > 0 else 0
                print(
                    f"[batch_localize] {processed}/{n_rows} "
                    f"recent={rate_recent:.1f} img/s "
                    f"overall={rate_overall:.1f} img/s "
                    f"missing={n_missing} zero_det={n_zero_det} "
                    f"detected={n_detected} errors={n_errors}",
                    flush=True,
                )
                last_log_t = now
                last_log_i = processed

    t_total = time.perf_counter() - t0
    summary = {
        "n_rows": n_rows,
        "n_detected": n_detected,
        "n_zero_det": n_zero_det,
        "n_missing": n_missing,
        "n_errors": n_errors,
        "elapsed_seconds": round(t_total, 2),
        "throughput_img_per_sec": round(n_rows / t_total, 2) if t_total > 0 else 0,
        "out": str(args.out),
    }
    print("[batch_localize] DONE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
