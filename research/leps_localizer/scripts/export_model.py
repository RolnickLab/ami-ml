"""Export an Ultralytics-trained leps localizer (YOLO11/26, RT-DETR) to one
or more deployment formats: CoreML, ONNX, TFLite, TFJS, OpenVINO, TensorRT,
TorchScript, SavedModel.

Mobile target (iOS/Android/web): typically `--formats coreml onnx tflite tfjs`
at `--imgsz 640`. Server target: `--formats onnx engine` at `--imgsz 1280`.

DEIM weights are NOT supported here — DEIM ships its own ONNX exporter; add
a sibling script when promoting DEIM.

Run on the workspace VM (CPU export works while training holds the GPU):

    uv sync --extra detection --extra export-edge
    uv run --extra detection --extra export-edge \\
        python research/leps_localizer/scripts/export_model.py \\
        --weights /mnt/butterflies-fg-2026-05/runs/yolo26s-fg-2026-05/weights/best.pt \\
        --formats onnx coreml tflite tfjs \\
        --imgsz 640 --device cpu \\
        --out-dir /mnt/butterflies-fg-2026-05/exports/yolo26s
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

SUPPORTED_FORMATS = (
    "coreml",
    "onnx",
    "tflite",
    "tfjs",
    "openvino",
    "engine",
    "torchscript",
    "saved_model",
)


def _load_env_file(env_path: Path) -> None:
    """Lightweight .env loader (avoid the python-dotenv dep import here)."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


def _sha256_path(path: Path) -> str:
    """SHA256 over a file or recursively over a directory's file contents."""
    h = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    elif path.is_dir():
        for sub in sorted(path.rglob("*")):
            if sub.is_file():
                h.update(sub.relative_to(path).as_posix().encode())
                with sub.open("rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
    return h.hexdigest()


def _size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    if path.is_dir():
        total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        return total / (1024 * 1024)
    return 0.0


def _smoke_test_onnx(onnx_path: Path, imgsz: int) -> str | None:
    """Run a 1-image dummy inference. Return None on success, message on failure."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        return f"onnxruntime not installed: {e}"
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        outs = sess.run(None, {in_name: dummy})
        return None if outs else "no outputs"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _export_one(
    model,
    fmt: str,
    args: argparse.Namespace,
) -> dict:
    """Run a single Ultralytics export. Return a manifest record."""
    record: dict = {
        "format": fmt,
        "path": None,
        "size_mb": None,
        "sha256": None,
        "elapsed_s": None,
        "error": None,
        "smoke_test": None,
    }
    kwargs: dict = {
        "format": fmt,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
    }
    if args.half:
        kwargs["half"] = True
    if args.int8:
        kwargs["int8"] = True
    if args.data is not None:
        kwargs["data"] = str(args.data)
    if args.nms:
        kwargs["nms"] = True
    if fmt == "onnx":
        kwargs["simplify"] = args.simplify
        kwargs["opset"] = args.opset

    t0 = time.monotonic()
    try:
        out = model.export(**kwargs)
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()
        record["elapsed_s"] = round(time.monotonic() - t0, 2)
        return record
    record["elapsed_s"] = round(time.monotonic() - t0, 2)

    out_path = Path(out) if isinstance(out, (str, Path)) else None
    if out_path is None or not out_path.exists():
        record["error"] = f"export returned {out!r} - path not found"
        return record

    record["path"] = str(out_path)
    record["size_mb"] = round(_size_mb(out_path), 2)
    record["sha256"] = _sha256_path(out_path)

    if args.smoke_test and fmt == "onnx":
        record["smoke_test"] = _smoke_test_onnx(out_path, args.imgsz) or "ok"

    return record


def _maybe_copy(src: Path, out_dir: Path) -> Path:
    """Copy artifact (file or dir) into out_dir, return new path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.name
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return dst


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--weights", type=Path, required=True, help="best.pt path")
    p.add_argument(
        "--formats",
        nargs="+",
        required=True,
        choices=SUPPORTED_FORMATS,
        help="export targets",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--half", action="store_true", help="fp16 export")
    p.add_argument("--int8", action="store_true", help="int8 quantization")
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="data.yaml for int8 calibration",
    )
    p.add_argument(
        "--nms",
        action="store_true",
        help="embed NMS in graph (CoreML/TFLite where supported)",
    )
    p.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="onnx-simplifier (default on)",
    )
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='"cpu" or GPU index. CPU is safest while training holds the MIG slice',
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="parent dir for export artifacts. Default: same dir as --weights",
    )
    p.add_argument(
        "--timestamp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "create a fresh `export-<UTC ts>` subdir inside --out-dir per invocation"
            " (default on). --no-timestamp writes directly into --out-dir, clobbering"
            " any prior run's manifest + artifacts."
        ),
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="run 1-image dummy inference for ONNX",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print plan and exit, do not export",
    )
    p.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="path to .env (loaded for COREMLTOOLS_*, etc.)",
    )
    args = p.parse_args()

    if not args.weights.exists():
        print(f"ERROR: weights not found: {args.weights}", file=sys.stderr)
        return 2
    if args.int8 and args.data is None:
        print(
            "WARN: --int8 without --data; calibration set may be required",
            file=sys.stderr,
        )

    _load_env_file(args.env_file)

    parent_dir = args.out_dir if args.out_dir is not None else args.weights.parent
    if args.timestamp:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = parent_dir / f"export-{ts}"
    else:
        out_dir = parent_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"weights : {args.weights}")
    print(f"formats : {', '.join(args.formats)}")
    print(f"imgsz   : {args.imgsz}  batch: {args.batch}  device: {args.device}")
    print(
        f"flags   : half={args.half} int8={args.int8} nms={args.nms}"
        f" simplify={args.simplify} opset={args.opset}"
    )
    print(f"out_dir : {out_dir}")
    print("=" * 72)

    if args.dry_run:
        print("[dry-run] no exports performed")
        return 0

    from ultralytics import YOLO  # lazy

    weights_sha = _sha256_path(args.weights)
    weights_size_mb = round(_size_mb(args.weights), 2)

    try:
        import torch
        import ultralytics

        ultra_ver = ultralytics.__version__
        torch_ver = torch.__version__
    except Exception:
        ultra_ver = torch_ver = "unknown"

    model = YOLO(str(args.weights))

    records: list[dict] = []
    for fmt in args.formats:
        print(f"\n--- exporting {fmt} ---")
        rec = _export_one(model, fmt, args)
        if rec["error"]:
            print(f"  FAIL ({rec['elapsed_s']}s): {rec['error']}")
        else:
            print(
                f"  OK ({rec['elapsed_s']}s): {rec['path']}"
                f"  {rec['size_mb']} MB  sha256={rec['sha256'][:12]}"
            )
            src_path = Path(rec["path"])
            if src_path.parent.resolve() != out_dir.resolve():
                dst = _maybe_copy(src_path, out_dir)
                rec["path"] = str(dst)
                print(f"  copied -> {dst}")
            if rec["smoke_test"]:
                print(f"  smoke_test: {rec['smoke_test']}")
        records.append(rec)

    manifest = {
        "weights": str(args.weights),
        "weights_sha256": weights_sha,
        "weights_size_mb": weights_size_mb,
        "ultralytics_version": ultra_ver,
        "torch_version": torch_ver,
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "exports": records,
    }
    manifest_path = out_dir / "EXPORT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nmanifest -> {manifest_path}")

    n_ok = sum(1 for r in records if not r["error"])
    n_fail = len(records) - n_ok
    print(f"\nsummary: {n_ok} ok, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
