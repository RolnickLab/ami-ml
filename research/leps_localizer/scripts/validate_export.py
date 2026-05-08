"""Cross-format inference parity check for classifier exports.

Reads `EXPORT_MANIFEST.json` from a directory produced by
`export_classifier.py` and runs the same set of test images through:

  * PyTorch reference (timm + checkpoint, always normalized input)
  * ONNX (via onnxruntime CPU)
  * TorchScript

CoreML (`.mlpackage`) is **skipped on Linux** — load it from the
`macos-vm` Mac runtime separately. This script only validates the
formats that have a usable runtime here.

Usage:

    uv run --extra detection --extra export-edge \\
        python research/leps_localizer/scripts/validate_export.py \\
        --export-dir /home/michael/Projects/LepsAI/models/staging/export-20260508T234625Z \\
        --images "/home/michael/Projects/LepsAI/tools/test_images/*.jpg" \\
        --top-k 5

Outputs a per-image top-K table for each format plus a parity summary
(max |logit_pt - logit_onnx|, top-1 agreement %, top-1 accuracy
recovered from `<Genus>_<species>_<n>.jpg` filename pattern).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

DEFAULT_IMAGES_GLOB = "/home/michael/Projects/LepsAI/tools/test_images/*.jpg"


def _load_manifest(export_dir: Path) -> dict:
    p = export_dir / "EXPORT_MANIFEST.json"
    if not p.exists():
        sys.exit(f"ERROR: no EXPORT_MANIFEST.json in {export_dir}")
    return json.loads(p.read_text())


def _filename_label(path: Path) -> str | None:
    """Recover '<Genus>_<species>_<n>.jpg' → 'Genus species'."""
    parts = path.stem.split("_")
    if len(parts) < 2:
        return None
    return f"{parts[0]} {parts[1]}"


def _build_transform(imgsz: int, mean: list[float], std: list[float]):
    """Standard timm-style eval transform: Resize(imgsz/0.875) →
    CenterCrop(imgsz) → ToTensor → Normalize."""
    from torchvision import transforms

    resize_to = int(round(imgsz / 0.875))
    return transforms.Compose(
        [
            transforms.Resize(
                resize_to, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(imgsz),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _load_pytorch_ref(arch: str, num_classes: int, weights_path: Path):
    import timm
    import torch

    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    return model.eval()


def _topk(logits, labels: list[str], k: int) -> list[tuple[int, str, float]]:
    import numpy as np

    arr = logits if isinstance(logits, np.ndarray) else logits.detach().cpu().numpy()
    arr = arr.flatten()
    probs = np.exp(arr - arr.max())
    probs = probs / probs.sum()
    idx = np.argsort(probs)[::-1][:k]
    return [(int(i), labels[int(i)], float(probs[int(i)])) for i in idx]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--export-dir", type=Path, required=True)
    p.add_argument(
        "--images",
        type=str,
        default=DEFAULT_IMAGES_GLOB,
        help=f"glob pattern (default: {DEFAULT_IMAGES_GLOB})",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="max |logit_pt - logit_onnx| gate (default 1e-3)",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="cap (default 20)",
    )
    args = p.parse_args()

    manifest = _load_manifest(args.export_dir)
    arch = manifest["arch"]
    num_classes = manifest["num_classes"]
    imgsz = manifest["imgsz"]
    mean = manifest["mean"]
    std = manifest["std"]
    weights_path = Path(manifest["weights_resolved"])
    label_map_path = Path(manifest["label_map_resolved"])
    raw_output = manifest["args"].get("raw_output", False)

    if not weights_path.exists():
        sys.exit(f"ERROR: weights not found: {weights_path}")
    labels_raw = json.loads(label_map_path.read_text())
    if isinstance(labels_raw, list):
        labels = list(labels_raw)
    else:
        labels = [labels_raw[str(i)] for i in range(num_classes)]

    image_paths = sorted(Path(p) for p in glob.glob(args.images))[: args.max_images]
    if not image_paths:
        sys.exit(f"ERROR: no images matched {args.images!r}")

    onnx_path = next(iter(args.export_dir.glob("*.onnx")), None)
    ts_path = next(iter(args.export_dir.glob("*.torchscript")), None)
    mlpkg_paths = list(args.export_dir.glob("*.mlpackage"))

    print("=" * 78)
    print(f"export-dir : {args.export_dir}")
    print(f"arch       : {arch}  classes: {num_classes}  imgsz: {imgsz}")
    print(f"raw_output : {raw_output}  (PT ref always uses fully-normalized input)")
    print(f"images     : {len(image_paths)} matched")
    print(f"PT ref     : {weights_path}")
    print(f"ONNX       : {onnx_path or 'not present'}")
    print(f"TorchScript: {ts_path or 'not present'}")
    if mlpkg_paths:
        print(f"CoreML     : {mlpkg_paths[0]}  (SKIPPED on Linux — run on macos-vm)")
    print("=" * 78)

    import numpy as np
    import torch
    from PIL import Image

    transform = _build_transform(imgsz, mean, std)

    pt_model = _load_pytorch_ref(arch, num_classes, weights_path)

    onnx_sess = None
    onnx_in_name = None
    if onnx_path is not None:
        try:
            import onnxruntime as ort

            onnx_sess = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            onnx_in_name = onnx_sess.get_inputs()[0].name
        except Exception as e:
            print(f"WARN: failed to load ONNX: {e}", file=sys.stderr)

    ts_model = None
    if ts_path is not None:
        try:
            ts_model = torch.jit.load(str(ts_path), map_location="cpu").eval()
        except Exception as e:
            print(f"WARN: failed to load TorchScript: {e}", file=sys.stderr)

    pt_correct = 0
    onnx_correct = 0
    ts_correct = 0
    pt_vs_onnx_max_diff = 0.0
    pt_vs_ts_max_diff = 0.0
    top1_agree_onnx = 0
    top1_agree_ts = 0
    n_eval = 0

    for img_path in image_paths:
        gt = _filename_label(img_path)
        img = Image.open(img_path).convert("RGB")

        normed = transform(img).unsqueeze(0)
        unnormed = (normed * torch.tensor(std).view(1, 3, 1, 1)) + torch.tensor(
            mean
        ).view(1, 3, 1, 1)

        with torch.no_grad():
            pt_logits = pt_model(normed).cpu().numpy().flatten()

        onnx_logits = None
        if onnx_sess is not None:
            feed = normed.numpy() if onnx_in_name == "input" else unnormed.numpy()
            onnx_logits = onnx_sess.run(None, {onnx_in_name: feed.astype(np.float32)})[
                0
            ].flatten()

        ts_logits = None
        if ts_model is not None:
            ts_in = normed if raw_output else unnormed
            with torch.no_grad():
                ts_logits = ts_model(ts_in).cpu().numpy().flatten()

        pt_top = _topk(pt_logits, labels, args.top_k)
        pt_top1_label = pt_top[0][1]

        print(f"\n--- {img_path.name} ---")
        if gt:
            print(f"  ground_truth = {gt}")
        print(f"  PT top-{args.top_k}:")
        for i, (idx, lbl, prob) in enumerate(pt_top, 1):
            print(f"    {i}. [{idx}] {lbl}  {prob:.4f}")
        if gt and pt_top1_label == gt:
            pt_correct += 1

        if onnx_logits is not None:
            diff = float(np.max(np.abs(pt_logits - onnx_logits)))
            pt_vs_onnx_max_diff = max(pt_vs_onnx_max_diff, diff)
            onnx_top = _topk(onnx_logits, labels, args.top_k)
            agree = pt_top[0][0] == onnx_top[0][0]
            top1_agree_onnx += int(agree)
            if gt and onnx_top[0][1] == gt:
                onnx_correct += 1
            mark = "ok" if agree else "MISMATCH"
            print(
                f"  ONNX top-1 [{onnx_top[0][0]}] {onnx_top[0][1]} "
                f"{onnx_top[0][2]:.4f} max-Δ={diff:.2e} {mark}"
            )

        if ts_logits is not None:
            diff = float(np.max(np.abs(pt_logits - ts_logits)))
            pt_vs_ts_max_diff = max(pt_vs_ts_max_diff, diff)
            ts_top = _topk(ts_logits, labels, args.top_k)
            agree = pt_top[0][0] == ts_top[0][0]
            top1_agree_ts += int(agree)
            if gt and ts_top[0][1] == gt:
                ts_correct += 1
            mark = "ok" if agree else "MISMATCH"
            print(
                f"  TS top-1 [{ts_top[0][0]}] {ts_top[0][1]} "
                f"{ts_top[0][2]:.4f} max-Δ={diff:.2e} {mark}"
            )

        n_eval += 1

    print("\n" + "=" * 78)
    print(f"images evaluated: {n_eval}")
    print(f"PT top-1 acc (gt-known): {pt_correct}/{n_eval}")
    if onnx_sess is not None:
        gate_ok = pt_vs_onnx_max_diff < args.tolerance
        print(
            f"ONNX top-1 acc {onnx_correct}/{n_eval}, agree-w-PT {top1_agree_onnx}/{n_eval}"
        )
        print(
            f"ONNX max-Δlogit {pt_vs_onnx_max_diff:.2e} "
            f"(gate {args.tolerance:.0e}) "
            f"{'PASS' if gate_ok else 'FAIL'}"
        )
    if ts_model is not None:
        gate_ok = pt_vs_ts_max_diff < args.tolerance
        print(
            f"TS top-1 acc {ts_correct}/{n_eval}, agree-w-PT {top1_agree_ts}/{n_eval}"
        )
        print(
            f"TS max-Δlogit {pt_vs_ts_max_diff:.2e} "
            f"(gate {args.tolerance:.0e}) "
            f"{'PASS' if gate_ok else 'FAIL'}"
        )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
