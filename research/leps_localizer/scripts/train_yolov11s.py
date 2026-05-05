"""Train YOLOv11s single-class arthropod detector on the prepared FG
butterfly train/val split.

Run on the Arbutus H100 24GB MIG VM. Requires the detection extra:

    poetry install --with detection

Default args target the 2,572-train / 290-val FG butterfly set built by
prepare_yolo_dataset.py. Override with --data / --epochs / etc.

Outputs land in `research/leps_localizer/runs/<name>/` (gitignored).
The best checkpoint is symlinked to `runs/<name>/weights/best.pt` by
Ultralytics; pass that to eval_on_locked.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=Path(
            "research/leps_localizer/data/"
            "yolo_butterflies-fg-train-2026-05/data.yaml"
        ),
    )
    p.add_argument("--model", type=str, default="yolo11s.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--project",
        type=Path,
        default=Path("research/leps_localizer/runs"),
    )
    p.add_argument("--name", type=str, default="yolov11s-fg-2026-05")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="disable Ultralytics' wandb logger",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260505,
        help="Ultralytics RNG seed (independent of split-assignment seed)",
    )
    args = p.parse_args()

    from ultralytics import YOLO  # lazy: only import when training

    if args.no_wandb:
        import os

        os.environ["WANDB_DISABLED"] = "true"

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        # single-class training: setting `single_cls=True` collapses any
        # multi-class labels into one. Our labels are already class_id=0.
        single_cls=True,
    )


if __name__ == "__main__":
    main()
