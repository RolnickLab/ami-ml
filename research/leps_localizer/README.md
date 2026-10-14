# Lepidoptera Single-Subject Localizer

This directory contains the research code for training and evaluating a small,
fast detector that finds the butterfly (or other arthropod) in a photo and
returns a bounding box that a species classifier can crop to. The detector is
meant to run both server-side over large photo collections and client-side in a
web browser or mobile app, so the work compares accuracy against model size and
export format.

Methods, datasets and results are written up in
[`METHODS_AND_RESULTS.md`](METHODS_AND_RESULTS.md), with the per-run numbers in
[`results.csv`](results.csv). [`EBUTTERFLY_INTEGRATION.md`](EBUTTERFLY_INTEGRATION.md)
describes how to embed the browser build of the detector in a photo upload
flow, and [`demo/`](demo/) holds a small static demo page.

## Data

Training images come from Fieldguide photos whose uploaders drew a crop box
around the subject. Those photos are not distributed with this repository.
The held-out evaluation set is the public Leeds Butterfly dataset (Wang et al.,
2009), which you download separately.

The scripts read a JSON Lines index with one record per image. The fields used
by `src/localization/leps_data.py` are `photo_id`, `image_key` (the image path
relative to its dataset root), `width`, `height`, `bbox_xyxy` (pixel
coordinates), `bbox_area_fraction`, `bbox_is_square`, `split` and
`dataset_name`. Any source that can produce that index can be used.

## Typical workflow

All paths are passed on the command line. The examples below use two
environment variables for brevity: `DATA_ROOT` for the directory that holds the
indexes and images, and `RUNS_DIR` for training outputs.

1. Merge the source indexes into one train/val split with
   `scripts/repartition_fg.py --detector-dataset-root "$DATA_ROOT/source" --out "$DATA_ROOT/index.jsonl"`.
2. Materialize a YOLO-format dataset with `scripts/prepare_yolo_dataset.py`.
3. Train with `scripts/train_yolo26s.py`, `scripts/train_yolov11s.py` or
   `scripts/train_rtdetr.py`. Install the training dependencies with
   `uv sync --extra detection`. Logging to Weights & Biases is optional and
   reads `WANDB_API_KEY` and `WANDB_ENTITY` from the environment or a `.env`
   file.
4. Evaluate on the held-out set with `scripts/eval_on_locked.py` (and
   `eval_sahi.py`, `eval_deimv2_on_locked.py` for the other model families).
5. Export for edge deployment with `scripts/export_model.py`
   (`uv sync --extra export-edge`), then check the export with
   `scripts/validate_export.py`, `eval_onnx_recall.py` or
   `eval_tflite_recall.py`.
6. Compare models side by side with `scripts/serve_gym.py`
   (`uv sync --extra demo`).

Reusable logic (index reading, split assignment, YOLO materialization and the
evaluation metrics) lives in `src/localization/` and is covered by
`tests/localization/`.

## Held-out evaluation

The Leeds Butterfly images are never used for training or model selection.
Section 1 of `METHODS_AND_RESULTS.md` explains the known biases of that set
(its boxes clip antennae, and most subjects fill the frame), which is why the
per-size-bucket recall is reported alongside the headline numbers.
