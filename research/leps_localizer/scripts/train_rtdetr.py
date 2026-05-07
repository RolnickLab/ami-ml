"""Train RT-DETR-l (Ultralytics) single-class arthropod detector on the
prepared FG butterfly train/val split.

Architectural baseline #2 alongside `train_yolov11s.py`. Uses the same
YOLO-format dataset.yaml. RT-DETR-l is ~33M params, 108 GFLOPs — about
3.5x larger than YOLOv11s. We train at imgsz=640 (RT-DETR's architectural
sweet spot — its positional encodings were tuned there); the comparison
report should note YOLO Run 1 used imgsz=1536, so this is a per-arch
"native" comparison rather than identical-resolution.

Run on the Arbutus H100 24GB MIG VM. Requires the detection extra:

    uv sync --extra detection
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
    p.add_argument("--model", type=str, default="rtdetr-l.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--project",
        type=Path,
        default=Path("research/leps_localizer/runs"),
    )
    p.add_argument("--name", type=str, default="rtdetr-l-fg-2026-05")
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
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="path to .env file with WANDB_API_KEY etc (relative to cwd)",
    )
    args = p.parse_args()

    _load_env_file(args.env_file)

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    # Ultralytics' wandb callback derives the project name from `args.project`
    # (the local checkpoint dir), sanitizing path separators into dashes —
    # pre-init wandb here so the callback reuses our run instead.
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
                    "arch": "rtdetr-l",
                    "epochs": args.epochs,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "data": str(args.data),
                    "seed": args.seed,
                },
            )

    from ultralytics import RTDETR  # lazy: only import when training

    model = RTDETR(args.model)
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
        # RT-DETR's trainer ignores YOLO-specific augmentations (mosaic,
        # copy_paste). Keep the default Ultralytics aug stack.
    )


if __name__ == "__main__":
    main()
