"""Pre-compress an ONNX file with Brotli (and optionally gzip) so the demo
server can stream it with Content-Encoding without runtime CPU cost.

Output sits next to the source as ``<file>.onnx.br`` and ``<file>.onnx.gz``.

Usage:
    python research/leps_localizer/scripts/compress_onnx.py path/to/model.onnx
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import brotli


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", type=Path)
    p.add_argument("--quality", type=int, default=11, help="brotli 0-11")
    p.add_argument("--no-gzip", action="store_true")
    args = p.parse_args()

    data = args.src.read_bytes()
    src_mb = len(data) / 1024 / 1024

    br_path = args.src.with_suffix(args.src.suffix + ".br")
    br = brotli.compress(data, quality=args.quality)
    br_path.write_bytes(br)
    br_mb = len(br) / 1024 / 1024
    print(
        f"brotli q{args.quality}: {src_mb:.2f} MB -> {br_mb:.2f} MB ({br_mb / src_mb:.0%})"
    )

    if not args.no_gzip:
        gz_path = args.src.with_suffix(args.src.suffix + ".gz")
        gz = gzip.compress(data, compresslevel=9)
        gz_path.write_bytes(gz)
        gz_mb = len(gz) / 1024 / 1024
        print(f"gzip -9: {src_mb:.2f} MB -> {gz_mb:.2f} MB ({gz_mb / src_mb:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
