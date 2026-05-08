"""Multi-model side-by-side comparison gym for the leps localizer.

Drop an image (or pull a held-out Leeds sample with its GT bbox) → fan out
to all selected models → grid of annotated outputs + per-model metrics row
(N preds, ms, best_iou vs GT, best_containment vs GT).

Reuses the predict_fn pattern from `eval_on_locked.py` and `eval_sahi.py`
so the same predictors can be benchmarked here interactively. Built for
research, not as the public demo — for that, strip to ONNX + clean UI in
a separate Space.

Run:

    uv sync --extra detection --extra demo
    python research/leps_localizer/scripts/serve_gym.py \\
        --yolo-pt /mnt/butterflies-fg-2026-05/runs/yolov11s-fg-2026-05-r2/weights/best.pt \\
        --rtdetr-pt /mnt/butterflies-fg-2026-05/runs/rtdetr-l-fg-2026-05/weights/best.pt \\
        --add-sahi \\
        --leeds-index /mnt/butterflies-fg-2026-05/metadata/leeds-index.jsonl \\
        --leeds-images /mnt/butterflies-fg-2026-05/datasets/leeds-butterflies/images \\
        --port 7860
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from src.localization.leps_data import IndexEntry, read_index_entries  # noqa: E402
from src.localization.metrics import containment_iou, iou_xyxy  # noqa: E402

# Predictor: (PIL.Image, conf) -> list of (xyxy, score) tuples.
PredictResult = list[tuple[list[float], float]]
PredictFn = Callable[[Image.Image, float], PredictResult]

NA = "—"


@dataclass
class Predictor:
    name: str
    arch: str  # "yolo" | "rtdetr" | "sahi-yolo"
    predict: PredictFn


def _checkpoint_label(model_path: Path, suffix: str = "") -> str:
    # runs/<run-name>/weights/best.pt -> <run-name>
    parent2 = model_path.parent.parent
    run_name = parent2.name if parent2 else model_path.stem
    return run_name + suffix


def _build_ultralytics_predictor(
    model_path: Path, *, device: str, arch: str
) -> Predictor:
    """Build a YOLO or RT-DETR predictor via Ultralytics' shared interface."""
    if arch == "rtdetr":
        from ultralytics import RTDETR as _Cls  # noqa: WPS433
    else:
        from ultralytics import YOLO as _Cls  # noqa: WPS433
    model = _Cls(str(model_path))
    name = arch + ":" + _checkpoint_label(model_path)

    def predict(img: Image.Image, conf: float) -> PredictResult:
        results = model.predict(img, conf=conf, device=device, verbose=False)
        out: PredictResult = []
        for r in results:
            boxes_attr = getattr(r, "boxes", None)
            if boxes_attr is None:
                continue
            xyxy = boxes_attr.xyxy
            scores = getattr(boxes_attr, "conf", None)
            arr_xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else xyxy
            if scores is not None:
                arr_sc = scores.cpu().numpy() if hasattr(scores, "cpu") else scores
            else:
                arr_sc = [0.0] * len(arr_xyxy)
            for box, sc in zip(arr_xyxy.tolist(), list(arr_sc)):
                out.append((list(box), float(sc)))
        return out

    return Predictor(name=name, arch=arch, predict=predict)


def _build_sahi_predictor(
    model_path: Path,
    *,
    device: str,
    slice_size: int = 768,
    overlap: float = 0.2,
    base_conf: float = 0.05,
) -> Predictor:
    """SAHI-tiled YOLO predictor. base_conf set low at build time so the UI
    confidence slider can filter post-hoc without rebuilding the model."""
    from sahi import AutoDetectionModel  # noqa: WPS433
    from sahi.predict import get_sliced_prediction  # noqa: WPS433

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(model_path),
        confidence_threshold=base_conf,
        device=("cuda:" + device) if device.isdigit() else device,
    )
    suffix = "+sahi(" + str(slice_size) + "," + str(overlap) + ")"
    name = "sahi-yolo:" + _checkpoint_label(model_path, suffix=suffix)

    def predict(img: Image.Image, conf: float) -> PredictResult:
        result = get_sliced_prediction(
            img,
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            verbose=0,
        )
        out: PredictResult = []
        for op in result.object_prediction_list:
            sc = float(op.score.value)
            if sc < conf:
                continue
            b = op.bbox
            out.append(([b.minx, b.miny, b.maxx, b.maxy], sc))
        return out

    return Predictor(name=name, arch="sahi-yolo", predict=predict)


def _draw_overlay(
    img: Image.Image,
    boxes: PredictResult,
    *,
    gt_box: list[float] | None = None,
    title: str = "",
) -> Image.Image:
    """Draw GT (red) and preds (lime) on a copy of img with optional title."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    if gt_box is not None:
        x1, y1, x2, y2 = gt_box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    for box, sc in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(50, 255, 50), width=3)
        draw.text((x1 + 4, y1 + 4), "%.2f" % sc, fill=(50, 255, 50))
    if title:
        draw.rectangle([0, 0, max(8 + 8 * len(title), 200), 24], fill="black")
        draw.text((6, 6), title, fill="yellow")
    return out


def _consensus_iou(per_model_top: list[list[float] | None]) -> float:
    """Mean pairwise IoU across each model's top pred. Score for "do the
    selected models agree on where the subject is" when GT is unknown."""
    valid = [b for b in per_model_top if b is not None]
    if len(valid) < 2:
        return 0.0
    pairs = [
        iou_xyxy(valid[i], valid[j])
        for i in range(len(valid))
        for j in range(i + 1, len(valid))
    ]
    return sum(pairs) / len(pairs) if pairs else 0.0


def _resolve_leeds_image(image_root: Path, image_key: str) -> Path | None:
    candidate = image_root / image_key
    if candidate.exists():
        return candidate
    bare = image_root / Path(image_key).name
    if bare.exists():
        return bare
    return None


def _load_leeds_sample(
    leeds_entries: list[IndexEntry],
    image_root: Path,
    idx_str: str,
) -> tuple[Image.Image | None, str]:
    if not leeds_entries:
        return None, ""
    try:
        idx = int(idx_str) if idx_str.strip() else random.randrange(len(leeds_entries))
    except (ValueError, TypeError):
        idx = random.randrange(len(leeds_entries))
    idx = max(0, min(idx, len(leeds_entries) - 1))
    entry = leeds_entries[idx]
    path = _resolve_leeds_image(image_root, entry.image_key)
    if path is None:
        return None, ""
    with Image.open(path) as raw:
        img = raw.convert("RGB").copy()
    gt_str = ",".join("%.1f" % x for x in entry.bbox_xyxy)
    return img, gt_str


def _parse_gt(gt_str: str) -> list[float] | None:
    if not gt_str:
        return None
    try:
        parts = [float(x.strip()) for x in gt_str.split(",")]
    except ValueError:
        return None
    return parts if len(parts) == 4 else None


def _nms_filter(
    boxes_scores: PredictResult, *, iou_threshold: float, top_k: int
) -> PredictResult:
    """Greedy NMS + top-k. Cleans up DETR-style query duplicates."""
    if not boxes_scores:
        return []
    sorted_bs = sorted(boxes_scores, key=lambda bs: bs[1], reverse=True)
    kept: PredictResult = []
    for box, score in sorted_bs:
        drop = False
        for kept_box, _ in kept:
            if iou_xyxy(box, kept_box) >= iou_threshold:
                drop = True
                break
        if drop:
            continue
        kept.append((box, score))
        if top_k > 0 and len(kept) >= top_k:
            break
    return kept


def _run_one(
    name: str,
    pred: Predictor,
    img: Image.Image,
    conf: float,
    nms_iou: float,
    top_k: int,
    gt_box: list[float] | None,
) -> tuple[Image.Image, list[object], list[float] | None]:
    t0 = time.perf_counter()
    raw = pred.predict(img, conf)
    ms = (time.perf_counter() - t0) * 1000.0
    boxes = _nms_filter(raw, iou_threshold=nms_iou, top_k=top_k)
    best_iou = 0.0
    best_cont = 0.0
    top_box: list[float] | None = None
    if boxes:
        top_box = boxes[0][0]
        if gt_box is not None:
            best_iou = max(iou_xyxy(b, gt_box) for b, _ in boxes)
            best_cont = max(containment_iou(b, gt_box) for b, _ in boxes)
    title = (
        name
        + " | n="
        + str(len(boxes))
        + "/"
        + str(len(raw))
        + " | "
        + ("%.0f" % ms)
        + "ms"
    )
    annotated = _draw_overlay(img, boxes, gt_box=gt_box, title=title)
    iou_cell = round(best_iou, 3) if gt_box is not None else NA
    cont_cell = round(best_cont, 3) if gt_box is not None else NA
    row = [name, len(boxes), len(raw), round(ms, 1), iou_cell, cont_cell]
    return annotated, row, top_box


def _run_models(
    predictors: dict[str, Predictor],
    img: Image.Image | None,
    gt_str: str,
    conf: float,
    nms_iou: float,
    top_k: int,
    selected: list[str],
) -> tuple[list[tuple[Image.Image, str]], list[list[object]]]:
    if img is None:
        return [], []
    gt_box = _parse_gt(gt_str)
    gallery: list[tuple[Image.Image, str]] = []
    rows: list[list[object]] = []
    per_model_top: list[list[float] | None] = []
    for name in selected:
        pred = predictors.get(name)
        if pred is None:
            continue
        annotated, row, top_box = _run_one(
            name, pred, img, conf, nms_iou, top_k, gt_box
        )
        per_model_top.append(top_box)
        gallery.append((annotated, name))
        rows.append(row)
    if gt_box is None and len(per_model_top) >= 2:
        cons = _consensus_iou(per_model_top)
        rows.append(["[consensus IoU]", NA, NA, NA, round(cons, 3), NA])
    return gallery, rows


def _register_predictors(args: argparse.Namespace) -> dict[str, Predictor]:
    predictors: dict[str, Predictor] = {}
    for path in args.yolo_pt:
        pred = _build_ultralytics_predictor(path, device=args.device, arch="yolo")
        predictors[pred.name] = pred
    for path in args.rtdetr_pt:
        pred = _build_ultralytics_predictor(path, device=args.device, arch="rtdetr")
        predictors[pred.name] = pred
    if args.add_sahi:
        if not args.yolo_pt:
            raise SystemExit("--add-sahi requires at least one --yolo-pt")
        pred = _build_sahi_predictor(
            args.yolo_pt[0],
            device=args.device,
            slice_size=args.sahi_slice,
            overlap=args.sahi_overlap,
        )
        predictors[pred.name] = pred
    if not predictors:
        raise SystemExit(
            "No predictors registered. " "Pass at least one --yolo-pt or --rtdetr-pt."
        )
    return predictors


def _build_ui(
    predictors: dict[str, Predictor],
    leeds_entries: list[IndexEntry],
    leeds_images: Path | None,
):
    import gradio as gr  # noqa: WPS433

    leeds_root = leeds_images  # closure-captured

    def _on_load_leeds(idx_str: str):
        if leeds_root is None:
            return None, ""
        return _load_leeds_sample(leeds_entries, leeds_root, idx_str)

    def _on_run(img, gt_str, conf, nms_iou, top_k, selected):
        return _run_models(predictors, img, gt_str, conf, nms_iou, int(top_k), selected)

    with gr.Blocks(title="Leps Localizer Gym") as demo:
        gr.Markdown(
            "# Leps Localizer Gym\n"
            "Drop an image or pull a held-out Leeds sample. Selected models "
            "all run on the same image; predictions and metrics shown side "
            "by side. GT box (if Leeds): red. Predictions: lime."
        )
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(
                    type="pil",
                    label="Image (drop here OR load Leeds sample below)",
                    height=380,
                )
                with gr.Row():
                    leeds_idx = gr.Textbox(
                        label="Leeds idx (blank = random)",
                        value="",
                        scale=2,
                    )
                    load_leeds_btn = gr.Button("Load Leeds", scale=1)
                gt_text = gr.Textbox(
                    label=(
                        "GT bbox xyxy (auto-filled by Leeds; " "clear for no-GT mode)"
                    ),
                    value="",
                )
                conf = gr.Slider(
                    0.05,
                    0.95,
                    value=0.25,
                    step=0.05,
                    label="Confidence threshold",
                )
                nms_iou = gr.Slider(
                    0.1,
                    0.9,
                    value=0.45,
                    step=0.05,
                    label="NMS IoU threshold (drop overlapping preds)",
                )
                top_k = gr.Slider(
                    0,
                    10,
                    value=5,
                    step=1,
                    label="Top K (0 = keep all post-NMS)",
                )
                models_cb = gr.CheckboxGroup(
                    choices=list(predictors.keys()),
                    value=list(predictors.keys()),
                    label="Models",
                )
                run_btn = gr.Button("Run all", variant="primary")
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Predictions",
                    columns=2,
                    height=560,
                    preview=True,
                )
                metrics = gr.Dataframe(
                    headers=[
                        "model",
                        "n_kept",
                        "n_raw",
                        "ms",
                        "best_iou",
                        "best_containment",
                    ],
                    label=(
                        "Per-model metrics "
                        "(n_kept = post-NMS+top-K; n_raw = raw model output)"
                    ),
                    wrap=True,
                )

        load_leeds_btn.click(
            fn=_on_load_leeds,
            inputs=[leeds_idx],
            outputs=[img_in, gt_text],
        )
        run_btn.click(
            fn=_on_run,
            inputs=[img_in, gt_text, conf, nms_iou, top_k, models_cb],
            outputs=[gallery, metrics],
        )
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--yolo-pt",
        type=Path,
        action="append",
        default=[],
        help="path to a YOLO .pt checkpoint (repeat for multiple)",
    )
    p.add_argument(
        "--rtdetr-pt",
        type=Path,
        action="append",
        default=[],
        help="path to an RT-DETR .pt checkpoint (repeat for multiple)",
    )
    p.add_argument(
        "--add-sahi",
        action="store_true",
        help="also register a SAHI-tiled variant of the FIRST --yolo-pt",
    )
    p.add_argument("--sahi-slice", type=int, default=768)
    p.add_argument("--sahi-overlap", type=float, default=0.2)
    p.add_argument("--leeds-index", type=Path, default=None)
    p.add_argument("--leeds-images", type=Path, default=None)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument(
        "--share",
        action="store_true",
        help="Gradio public-tunnel share link (skip if behind nginx)",
    )
    args = p.parse_args()

    predictors = _register_predictors(args)
    leeds_entries: list[IndexEntry] = []
    if args.leeds_index is not None and args.leeds_images is not None:
        leeds_entries = read_index_entries(args.leeds_index)
        n = len(leeds_entries)
        print("Loaded " + str(n) + " Leeds samples from " + str(args.leeds_index))

    print("Registered " + str(len(predictors)) + " predictors")
    for name in predictors:
        print("  - " + name)

    demo = _build_ui(predictors, leeds_entries, args.leeds_images)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
