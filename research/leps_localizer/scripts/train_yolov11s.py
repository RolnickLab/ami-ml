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
import os
from pathlib import Path


def _load_env_file(env_path: Path) -> None:
    """Lightweight .env loader (avoid the python-dotenv dep import here).

    Only sets keys not already in os.environ.
    """
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
    p.add_argument(
        "--copy-paste",
        type=float,
        default=0.0,
        help="probability of copy-paste augmentation per image (0.0-1.0). "
        "Big-bang aug for small-object detection.",
    )
    p.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="probability of mosaic augmentation (Ultralytics default 1.0)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="image scale jitter range (Ultralytics default 0.5)",
    )
    p.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="path to .env file with WANDB_API_KEY etc (relative to cwd)",
    )
    args = p.parse_args()

    _load_env_file(args.env_file)

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    from ultralytics import YOLO  # lazy: only import when training

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
        copy_paste=args.copy_paste,
        mosaic=args.mosaic,
        scale=args.scale,
        # single-class training: setting `single_cls=True` collapses any
        # multi-class labels into one. Our labels are already class_id=0.
        single_cls=True,
    )


if __name__ == "__main__":
    main()
