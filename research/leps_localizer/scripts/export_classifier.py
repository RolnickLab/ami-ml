"""Export a timm-based species classifier to deployment formats: CoreML
(iOS), ONNX, and TorchScript.

Sibling to `export_model.py` (Ultralytics detector exporter). Detectors
ship via `model.export()` from Ultralytics; classifiers are plain
torch+timm modules and need a different path: trace, normalize-baked,
then `coremltools.convert` / `torch.onnx.export` / `torch.jit.save`.

Two CoreML output patterns:

* **Default (image-native)**: input is `ct.ImageType` named `image`
  with ImageNet normalization baked into the traced graph; output uses
  `ct.ClassifierConfig(labels)` so the model returns a
  `{species_name: probability}` dict natively. Best for new iOS apps —
  no preprocessing or label lookup on the device side.

* **`--raw-output` (LepsAI iOS pattern)**: input is `ct.TensorType`
  named `input` (`(1, 3, H, W)` float, app-normalized); output is
  `ct.TensorType` named `logits`. App handles resize, `/255`, mean/std
  normalization, softmax, and label lookup against the side-loaded
  category-map JSON. Required for the existing LepsAI
  `CoreMLClassifier.swift` codepath.

A `<model-id>-category-map.json` (enriched format,
`[{class_index, scientific_name}, ...]`) is always written next to the
artifact. Pass `--ios-bundle` to also emit a stub
`<model-id>.model-info.json` matching LepsAI's `ModelRegistry` schema
(fill in the TODO display fields before bundling into the iOS app).

Inputs (`--weights`, `--label-map`) accept:
  * a local path
  * `hf://<repo_id>/<filename>` — fetched from Hugging Face

Example (Mohamed's global butterflies model, 8851 classes, 512×512):

    uv run --extra detection --extra export-edge \\
        python research/leps_localizer/scripts/export_classifier.py \\
        --weights hf://mohammedelabbas/global-butterflies-max1000img-512/resnet50_20260504_135731_checkpoint.pt \\
        --label-map hf://mohammedelabbas/global-butterflies-max1000img-512/label_map.json \\
        --arch resnet50 --num-classes 8851 --imgsz 512 \\
        --formats coreml onnx \\
        --out-dir ./exports/global-butterflies-resnet50-512 \\
        --smoke-test

Default mean/std is ImageNet. Override via `--mean`/`--std` if your
trainer used a different normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SUPPORTED_FORMATS = ("coreml", "onnx", "torchscript")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HF_CACHE = Path.home() / ".cache" / "ami-ml" / "hf"


def _load_env_file(env_path: Path) -> None:
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


def _resolve_hf(spec: str) -> Path:
    """Resolve hf://<repo_id>/<filename> to a cached local path."""
    rest = spec[len("hf://") :]
    parts = rest.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"bad hf spec: {spec!r} (expected hf://<owner>/<repo>/<filename>)"  # noqa: E231
        )
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    cache_dir = HF_CACHE / repo_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / filename
    if local.exists() and local.stat().st_size > 0:
        return local
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"  # noqa: E231
    print(f"  hf download: {url}")
    tmp = local.with_suffix(local.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(local)
    return local


def _resolve(spec: str) -> Path:
    if spec.startswith("hf://"):
        return _resolve_hf(spec)
    return Path(spec)


def _load_label_list(label_map_path: Path, num_classes: int) -> list[str]:
    """label_map.json is {"0": "Aaa bbb", ...}. Return ordered list[str]."""
    raw = json.loads(label_map_path.read_text())
    if isinstance(raw, list):
        labels = list(raw)
    elif isinstance(raw, dict):
        try:
            labels = [raw[str(i)] for i in range(num_classes)]
        except KeyError as e:
            raise ValueError(f"label_map missing index {e}") from e
    else:
        raise ValueError(f"unexpected label_map type: {type(raw).__name__}")
    if len(labels) != num_classes:
        raise ValueError(f"label_map has {len(labels)} entries, expected {num_classes}")
    return labels


def _build_model(
    arch: str,
    num_classes: int,
    weights_path: Path,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    raw_output: bool,
):
    """Load timm model. When `raw_output` is False (default), wrap with
    ImageNet normalization so CoreML/ONNX consumers can pass a plain
    [0, 1] image tensor. When `raw_output` is True, return the bare
    timm model — the consumer is expected to do its own resize +
    `/255` + mean/std normalization (matches LepsAI iOS pattern)."""
    import timm
    import torch
    import torch.nn as nn

    base = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    missing, unexpected = base.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"  state_dict: missing={len(missing)} unexpected={len(unexpected)}",
            file=sys.stderr,
        )

    if raw_output:
        return base.eval()

    class Wrapped(nn.Module):
        def __init__(self, m, mean_, std_):
            super().__init__()
            self.m = m
            self.register_buffer("mean", torch.tensor(mean_).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std_).view(1, 3, 1, 1))

        def forward(self, x):
            return self.m((x - self.mean) / self.std)

    return Wrapped(base, mean, std).eval()


def _smoke_test_onnx(onnx_path: Path, imgsz: int, num_classes: int) -> str:
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
        if not outs:
            return "no outputs"
        shape = tuple(outs[0].shape)
        if shape != (1, num_classes):
            return f"unexpected output shape {shape}, want (1, {num_classes})"
        return "ok"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _smoke_test_coreml(mlpkg_path: Path) -> str:
    try:
        import coremltools as ct
    except ImportError as e:
        return f"coremltools not installed: {e}"
    try:
        spec = ct.models.MLModel(str(mlpkg_path)).get_spec()
        out_names = [o.name for o in spec.description.output]
        return f"ok (outputs: {out_names})"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _export_coreml(model, args, labels: list[str], out_dir: Path) -> dict:
    import coremltools as ct
    import torch

    rec: dict = {
        "format": "coreml",
        "path": None,
        "size_mb": None,
        "sha256": None,
        "elapsed_s": None,
        "error": None,
        "smoke_test": None,
    }
    t0 = time.monotonic()
    try:
        example = torch.zeros(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)
        traced = torch.jit.trace(model, example, strict=False)
        if args.raw_output:
            inputs = [
                ct.TensorType(
                    name="input",
                    shape=(1, 3, args.imgsz, args.imgsz),
                )
            ]
            outputs = [ct.TensorType(name="logits")]
            convert_kwargs: dict = dict(
                inputs=inputs,
                outputs=outputs,
                convert_to="mlprogram",
                compute_units=ct.ComputeUnit.ALL,
                minimum_deployment_target=ct.target.iOS17,
            )
        else:
            convert_kwargs = dict(
                inputs=[
                    ct.ImageType(
                        name="image",
                        shape=(1, 3, args.imgsz, args.imgsz),
                        scale=1 / 255.0,
                        bias=[0.0, 0.0, 0.0],
                        color_layout=ct.colorlayout.RGB,
                    )
                ],
                convert_to="mlprogram",
                compute_units=ct.ComputeUnit.ALL,
                minimum_deployment_target=ct.target.iOS16,
                classifier_config=ct.ClassifierConfig(labels),
            )
        cml = ct.convert(traced, **convert_kwargs)
        try:
            cml.short_description = (  # type: ignore[union-attr]
                f"{args.arch} classifier, {args.num_classes} classes, "
                f"{args.imgsz}x{args.imgsz} RGB"
            )
        except AttributeError:
            pass
        out = out_dir / f"{args.model_id}.mlpackage"
        if out.exists():
            import shutil

            shutil.rmtree(out)
        cml.save(str(out))  # type: ignore[union-attr]
        rec["elapsed_s"] = round(time.monotonic() - t0, 2)
        rec["path"] = str(out)
        rec["size_mb"] = round(_size_mb(out), 2)
        rec["sha256"] = _sha256_path(out)
        if args.smoke_test:
            rec["smoke_test"] = _smoke_test_coreml(out)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
        rec["elapsed_s"] = round(time.monotonic() - t0, 2)
    return rec


def _export_onnx(model, args, out_dir: Path) -> dict:
    import torch

    rec: dict = {
        "format": "onnx",
        "path": None,
        "size_mb": None,
        "sha256": None,
        "elapsed_s": None,
        "error": None,
        "smoke_test": None,
    }
    t0 = time.monotonic()
    try:
        example = torch.zeros(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)
        out = out_dir / f"{args.model_id}.onnx"
        in_name = "input" if args.raw_output else "image"
        torch.onnx.export(
            model,
            (example,),
            str(out),
            input_names=[in_name],
            output_names=["logits"],
            opset_version=args.opset,
            dynamic_axes=(
                {in_name: {0: "batch"}, "logits": {0: "batch"}}
                if args.dynamic_batch
                else None
            ),
        )
        if args.simplify:
            try:
                import onnx
                from onnxslim import slim

                slimmed = slim(onnx.load(str(out)))
                onnx.save(slimmed, str(out))
            except ImportError:
                pass
        rec["elapsed_s"] = round(time.monotonic() - t0, 2)
        rec["path"] = str(out)
        rec["size_mb"] = round(_size_mb(out), 2)
        rec["sha256"] = _sha256_path(out)
        if args.smoke_test:
            rec["smoke_test"] = _smoke_test_onnx(out, args.imgsz, args.num_classes)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
        rec["elapsed_s"] = round(time.monotonic() - t0, 2)
    return rec


def _export_torchscript(model, args, out_dir: Path) -> dict:
    import torch

    rec: dict = {
        "format": "torchscript",
        "path": None,
        "size_mb": None,
        "sha256": None,
        "elapsed_s": None,
        "error": None,
        "smoke_test": None,
    }
    t0 = time.monotonic()
    try:
        example = torch.zeros(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)
        traced = torch.jit.trace(model, example, strict=False)
        out = out_dir / f"{args.model_id}.torchscript"
        traced.save(str(out))
        rec["elapsed_s"] = round(time.monotonic() - t0, 2)
        rec["path"] = str(out)
        rec["size_mb"] = round(_size_mb(out), 2)
        rec["sha256"] = _sha256_path(out)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
        rec["elapsed_s"] = round(time.monotonic() - t0, 2)
    return rec


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="path to .pt checkpoint, or hf://<owner>/<repo>/<filename>",
    )
    p.add_argument(
        "--label-map",
        type=str,
        required=True,
        help="path or hf:// spec to label_map.json (idx-keyed dict or list)",
    )
    p.add_argument("--arch", type=str, default="resnet50", help="timm arch name")
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument(
        "--formats",
        nargs="+",
        required=True,
        choices=SUPPORTED_FORMATS,
        help="export targets",
    )
    p.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=list(IMAGENET_MEAN),
        metavar=("R", "G", "B"),
    )
    p.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=list(IMAGENET_STD),
        metavar=("R", "G", "B"),
    )
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run onnxslim on ONNX output (default on)",
    )
    p.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="export ONNX with dynamic batch dim (CoreML stays fixed-batch)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="parent dir for export artifacts",
    )
    p.add_argument(
        "--timestamp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "create a fresh `export-<UTC ts>` subdir per invocation"
            " (default on). --no-timestamp writes directly into --out-dir."
        ),
    )
    p.add_argument(
        "--raw-output",
        action="store_true",
        help=(
            "skip CoreML ClassifierConfig and emit raw logits. Required for"
            " LepsAI iOS app (CoreMLClassifier expects logits + does softmax"
            " + argmax against the side-loaded category map)."
        ),
    )
    p.add_argument(
        "--model-id",
        type=str,
        default=None,
        help=(
            "iOS modelID (also used as basename for category-map.json /"
            " model-info.json side files). Defaults to <arch>_<imgsz>_classifier."
        ),
    )
    p.add_argument(
        "--ios-bundle",
        action="store_true",
        help=(
            "write a <model-id>.model-info.json template alongside the"
            " category-map.json (LepsAI iOS bundle pattern). Display fields"
            " are stubbed; fill them in before bundling."
        ),
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="run dummy inference for ONNX + load-check for CoreML",
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
    )
    args = p.parse_args()

    if args.model_id is None:
        args.model_id = f"{args.arch}_{args.imgsz}_classifier"

    _load_env_file(args.env_file)

    print("=" * 72)
    print(f"weights   : {args.weights}")
    print(f"label_map : {args.label_map}")
    print(f"arch      : {args.arch}  num_classes: {args.num_classes}")
    print(f"imgsz     : {args.imgsz}")
    print(f"model_id  : {args.model_id}")
    print(f"formats   : {', '.join(args.formats)}")
    print(f"mean/std  : {tuple(args.mean)} / {tuple(args.std)}")
    print(f"raw_output: {args.raw_output}  ios_bundle: {args.ios_bundle}")
    print(f"out_dir   : {args.out_dir}")
    print("=" * 72)

    if args.dry_run:
        print("[dry-run] no exports performed")
        return 0

    weights_path = _resolve(args.weights)
    label_map_path = _resolve(args.label_map)
    if not weights_path.exists():
        print(f"ERROR: weights not found: {weights_path}", file=sys.stderr)
        return 2
    if not label_map_path.exists():
        print(f"ERROR: label_map not found: {label_map_path}", file=sys.stderr)
        return 2

    labels = _load_label_list(label_map_path, args.num_classes)
    print(f"loaded {len(labels)} labels (sample: {labels[:3]} ... {labels[-1]!r})")

    if args.timestamp:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = args.out_dir / f"export-{ts}"
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_sha = _sha256_path(weights_path)
    weights_size_mb = round(_size_mb(weights_path), 2)

    try:
        import timm
        import torch

        timm_ver = timm.__version__
        torch_ver = torch.__version__
    except Exception:
        timm_ver = torch_ver = "unknown"

    print(f"\nbuilding model ({args.arch}, {args.num_classes} classes)")
    model = _build_model(
        args.arch,
        args.num_classes,
        weights_path,
        tuple(args.mean),
        tuple(args.std),
        raw_output=args.raw_output,
    )

    dispatch = {
        "coreml": lambda: _export_coreml(model, args, labels, out_dir),
        "onnx": lambda: _export_onnx(model, args, out_dir),
        "torchscript": lambda: _export_torchscript(model, args, out_dir),
    }

    records: list[dict] = []
    for fmt in args.formats:
        print(f"\n--- exporting {fmt} ---")
        rec = dispatch[fmt]()
        if rec["error"]:
            print(f"  FAIL ({rec['elapsed_s']}s): {rec['error']}")
        else:
            print(
                f"  OK ({rec['elapsed_s']}s): {rec['path']}"
                f"  {rec['size_mb']} MB  sha256={rec['sha256'][:12]}"
            )
            if rec["smoke_test"]:
                print(f"  smoke_test: {rec['smoke_test']}")
        records.append(rec)

    manifest = {
        "weights": str(args.weights),
        "weights_resolved": str(weights_path),
        "weights_sha256": weights_sha,
        "weights_size_mb": weights_size_mb,
        "label_map": str(args.label_map),
        "label_map_resolved": str(label_map_path),
        "num_classes": args.num_classes,
        "arch": args.arch,
        "imgsz": args.imgsz,
        "mean": list(args.mean),
        "std": list(args.std),
        "timm_version": timm_ver,
        "torch_version": torch_ver,
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "exports": records,
    }
    manifest_path = out_dir / "EXPORT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nmanifest -> {manifest_path}")

    cat_map_path = out_dir / f"{args.model_id}-category-map.json"
    cat_map = [
        {"class_index": i, "scientific_name": name} for i, name in enumerate(labels)
    ]
    cat_map_path.write_text(json.dumps(cat_map, indent=2))
    print(f"category-map -> {cat_map_path}")

    if args.ios_bundle:
        info = {
            "modelID": args.model_id,
            "displayName": "TODO: human-readable display name",
            "author": "TODO",
            "subject": "TODO (e.g. Butterflies)",
            "region": "TODO (e.g. Vermont, Global)",
            "speciesCount": args.num_classes,
            "version": 1,
            "inferenceMode": "onDevice",
            "supportsGeoPrior": False,
            "inputSize": args.imgsz,
            "categoryMapFile": f"{args.model_id}-category-map",
            "modelFile": args.model_id,
        }
        info_path = out_dir / f"{args.model_id}.model-info.json"
        info_path.write_text(json.dumps(info, indent=2))
        print(f"model-info -> {info_path} (fill in TODO fields before bundling)")

    n_ok = sum(1 for r in records if not r["error"])
    n_fail = len(records) - n_ok
    print(f"\nsummary: {n_ok} ok, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
