"""Train YOLO26-s single-class arthropod detector on the prepared FG
butterfly train/val split.

Drop-in successor to train_yolov11s.py. YOLO26 (Jan 2026) is NMS-free
(STAL + ProgLoss for small objects) and has native CoreML/TFLite/ONNX
export — chosen as the new mobile-class default. Architecture is
similar parameter count to YOLOv11s.

Run on the Arbutus H100 24GB MIG VM. Requires the detection extra:

    uv sync --extra detection

Outputs land in `research/leps_localizer/runs/<name>/` (gitignored).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


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
    p.add_argument("--model", type=str, default="yolo26s.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--project",
        type=Path,
        default=Path("research/leps_localizer/runs"),
    )
    p.add_argument("--name", type=str, default="yolo26s-fg-2026-05")
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
        help="probability of copy-paste augmentation per image (0.0-1.0)",
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

    if not args.no_wandb and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
        except ImportError:
            pass
        else:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "leps_localizer"),
                entity=os.environ.get("WANDB_ENTITY"),
                name=args.name,
                config={
                    "model": args.model,
                    "arch": "yolo26-s",
                    "epochs": args.epochs,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "copy_paste": args.copy_paste,
                    "mosaic": args.mosaic,
                    "scale": args.scale,
                    "data": str(args.data),
                    "seed": args.seed,
                },
            )

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
        single_cls=True,
    )


if __name__ == "__main__":
    main()
