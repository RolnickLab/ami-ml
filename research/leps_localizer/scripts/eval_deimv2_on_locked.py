"""Evaluate a DEIMv2 checkpoint against a locked index.jsonl eval set.

Companion to ``eval_on_locked.py`` (YOLO/RT-DETR). Reuses
``src.localization.eval_locked`` so DEIMv2 results land in the same metrics
schema (per-bucket recall, missed_completely, etc.) as the YOLO runs in
``/mnt/butterflies-fg-2026-05/eval/``.

DEIMv2 lives in its own repo with pinned ``torch==2.5.1`` /
``torchvision==0.20.1`` to dodge the v2-transforms breakage that blocks DEIM
v1. Run this script with the DEIMv2 venv:

    /path/to/DEIMv2/.venv/bin/python research/leps_localizer/scripts/eval_deimv2_on_locked.py \\
        --deimv2-root /path/to/DEIMv2 \\
        --config /path/to/DEIMv2/configs/deimv2_butterflies_s.yml \\
        --resume /path/to/runs/<run-name>/best_stg1.pth \\
        --index /path/to/leeds-index.jsonl \\
        --image-root /path/to/leeds-butterflies \\
        --out-dir /path/to/eval/deimv2_s_leeds \\
        --device cuda:0

The DEIMv2 root must point at a clone of
https://github.com/Intellindust-AI-Lab/DEIMv2 (provides ``engine.core``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

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


def _import_deimv2(deimv2_root: Path):
    if not (deimv2_root / "engine" / "core" / "__init__.py").exists():
        raise SystemExit(
            f"--deimv2-root={deimv2_root} does not look like a DEIMv2 clone"
            " (missing engine/core/__init__.py)"
        )
    if str(deimv2_root) not in sys.path:
        sys.path.insert(0, str(deimv2_root))
    from engine.core import YAMLConfig  # noqa: WPS433

    return YAMLConfig


def build_deimv2_predictor(
    YAMLConfig,
    config: Path,
    resume: Path,
    *,
    device: str,
    score_thresh: float,
):
    cfg = YAMLConfig(str(config), resume=str(resume))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(str(resume), map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    eval_size = cfg.yaml_cfg["eval_spatial_size"]
    if isinstance(eval_size, int):
        eval_size = (eval_size, eval_size)
    elif isinstance(eval_size, (list, tuple)) and len(eval_size) == 2:
        eval_size = tuple(eval_size)
    else:
        raise ValueError(f"unexpected eval_spatial_size: {eval_size!r}")

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
    def predict(img: Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        orig_size = torch.tensor([[w, h]], device=device)
        im_data = transforms(img).unsqueeze(0).to(device)
        _labels, boxes, scores = model(im_data, orig_size)
        scr = scores[0].detach().cpu().numpy()
        box = boxes[0].detach().cpu().numpy()
        keep = scr >= score_thresh
        return [b.tolist() for b in box[keep]]

    return predict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--deimv2-root",
        type=Path,
        default=Path(os.environ.get("DEIMV2_ROOT", "")),
        help="Path to DEIMv2 clone (or set DEIMV2_ROOT env var)",
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", type=Path, required=True, help="DEIMv2 .pth")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--score-thresh", type=float, default=0.45)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--title", type=str, default="DEIMv2 locked-eval")
    args = p.parse_args()

    if not args.deimv2_root or not args.deimv2_root.exists():
        raise SystemExit(
            "Pass --deimv2-root or set DEIMV2_ROOT env var to a DEIMv2 clone"
        )

    YAMLConfig = _import_deimv2(args.deimv2_root)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = read_index_entries(args.index, split=args.split)
    if not entries:
        raise SystemExit(f"no entries in {args.index} (split={args.split!r})")

    predict_fn = build_deimv2_predictor(
        YAMLConfig,
        args.config,
        args.resume,
        device=args.device,
        score_thresh=args.score_thresh,
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
        "model": str(args.resume),
        "config": str(args.config),
        "index": str(args.index),
        "image_root": str(args.image_root),
        "split": args.split or "all",
        "arch": "deimv2",
        "score_thresh": str(args.score_thresh),
    }
    report_path.write_text(
        format_markdown_report(
            agg, title=args.title, iou_threshold=args.iou_threshold, extra=extra
        )
    )

    print(f"wrote {samples_path}")
    print(f"wrote {metrics_path}")
    print(f"wrote {report_path}")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
