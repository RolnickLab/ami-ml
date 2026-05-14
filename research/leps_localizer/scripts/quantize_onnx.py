"""Quantize a YOLO26 ONNX export to FP16 / dynamic INT8 / static INT8.

Static INT8 requires a calibration image dir — pass a few hundred FG val
images so per-tensor activation ranges are realistic.

Usage:
    # FP16
    python research/leps_localizer/scripts/quantize_onnx.py fp16 \\
        --src best.onnx --dst best.fp16.onnx

    # Dynamic INT8 (weights only, no calib data)
    python research/leps_localizer/scripts/quantize_onnx.py dyn-int8 \\
        --src best.onnx --dst best.dyn-int8.onnx

    # Static INT8 (weights + activations, needs calib)
    python research/leps_localizer/scripts/quantize_onnx.py static-int8 \\
        --src best.onnx --dst best.static-int8.onnx \\
        --calib-dir /mnt/butterflies-fg-2026-05/yolo/images/val \\
        --calib-n 256 --imgsz 640
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from PIL import Image


def letterbox(img: np.ndarray, sz: int, pad: int = 114) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(sz / w, sz / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    px, py = (sz - nw) // 2, (sz - nh) // 2
    canvas = np.full((sz, sz, 3), pad, dtype=np.uint8)
    resized = np.asarray(
        Image.fromarray(img).resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.uint8
    )
    canvas[py : py + nh, px : px + nw] = resized
    return canvas


class FGCalibReader(CalibrationDataReader):
    def __init__(
        self, image_dir: Path, n: int, imgsz: int, input_name: str, seed: int = 0
    ):
        all_paths = sorted(
            list(image_dir.glob("*.jpg"))
            + list(image_dir.glob("*.jpeg"))
            + list(image_dir.glob("*.png"))
        )
        if not all_paths:
            raise SystemExit(f"no images in {image_dir}")
        rng = random.Random(seed)
        rng.shuffle(all_paths)
        self.paths = all_paths[:n]
        self.imgsz = imgsz
        self.input_name = input_name
        self._iter = None

    def _gen(self):
        for p in self.paths:
            img = np.asarray(Image.open(p).convert("RGB"))
            lb = letterbox(img, self.imgsz)
            tensor = lb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            yield {self.input_name: tensor}

    def get_next(self):  # type: ignore[override]
        # ORT's CalibrationDataReader contract: return dict for each step,
        # or None to signal end of calibration iteration.
        if self._iter is None:
            self._iter = self._gen()
        return next(self._iter, None)

    def rewind(self):
        self._iter = None


def fp16(src: Path, dst: Path) -> None:
    from onnxconverter_common import float16  # noqa: WPS433

    model = onnx.load(str(src))
    # Resize, NonMaxSuppression, TopK, and Where don't survive blanket fp16
    # conversion in YOLO end2end graphs — they have mixed-type IO that
    # onnxconverter-common doesn't bridge cleanly. Block them so they stay
    # FP32; surrounding nodes Cast as needed.
    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=["Resize", "TopK", "NonMaxSuppression", "Where", "GatherND"],
    )
    onnx.save(fp16_model, str(dst))


def dyn_int8(src: Path, dst: Path) -> None:
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
    )


def static_int8(
    src: Path, dst: Path, calib_dir: Path, calib_n: int, imgsz: int
) -> None:
    # ORT static quant requires shape-inference + symbolic shape preprocessing
    # before QDQ insertion. Without it, conv input ranges get wrong calibration
    # and outputs go to zero. Run the official preprocess pass first.
    from onnxruntime.quantization.shape_inference import (  # noqa: WPS433
        quant_pre_process,
    )

    pre_path = dst.with_suffix(".pre.onnx")
    quant_pre_process(
        input_model=str(src),
        output_model_path=str(pre_path),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
    )

    model = onnx.load(str(pre_path))
    input_name = model.graph.input[0].name
    reader = FGCalibReader(calib_dir, calib_n, imgsz, input_name)
    quantize_static(
        model_input=str(pre_path),
        model_output=str(dst),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        # Most ORT CPU EPs prefer QUInt8 activations + QInt8 weights.
        # Using QInt8 activations caused all-zero predictions on YOLO26-s.
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    pre_path.unlink(missing_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    a = sub.add_parser("fp16")
    a.add_argument("--src", type=Path, required=True)
    a.add_argument("--dst", type=Path, required=True)

    b = sub.add_parser("dyn-int8")
    b.add_argument("--src", type=Path, required=True)
    b.add_argument("--dst", type=Path, required=True)

    c = sub.add_parser("static-int8")
    c.add_argument("--src", type=Path, required=True)
    c.add_argument("--dst", type=Path, required=True)
    c.add_argument("--calib-dir", type=Path, required=True)
    c.add_argument("--calib-n", type=int, default=256)
    c.add_argument("--imgsz", type=int, default=640)

    args = p.parse_args()
    args.dst.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "fp16":
        fp16(args.src, args.dst)
    elif args.mode == "dyn-int8":
        dyn_int8(args.src, args.dst)
    elif args.mode == "static-int8":
        static_int8(args.src, args.dst, args.calib_dir, args.calib_n, args.imgsz)
    else:
        print(f"unknown mode {args.mode}", file=sys.stderr)
        return 2

    src_mb = args.src.stat().st_size / (1024 * 1024)
    dst_mb = args.dst.stat().st_size / (1024 * 1024)
    print(f"{args.mode}: {src_mb:.2f} MB -> {dst_mb:.2f} MB  ({dst_mb / src_mb:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
