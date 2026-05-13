# Leps Localizer — Browser Demo

Single-page web app that runs the **YOLO26-s** butterfly localizer (FG-2026-05
trained, single class) entirely in the browser via `onnxruntime-web`.

## Files

- `index.html` — page shell, slider controls, canvas
- `app.js` — preprocess (letterbox 640), session run, postprocess
  (YOLO26 NMS-free → `[1, 300, 6]`), draw bboxes
- `style.css` — minimal dark UI
- `model/yolo26s-fg-640.onnx` — **not committed** (37 MB). Export via:

  ```bash
  ssh <workspace-vm> '.venv/bin/python research/leps_localizer/scripts/export_model.py \
    --weights /mnt/butterflies-fg-2026-05/runs/yolo26s-fg-2026-05-v2-2/weights/best.pt \
    --formats onnx --imgsz 640 --device cpu \
    --out-dir /mnt/butterflies-fg-2026-05/exports/yolo26s-v2-web --smoke-test'

  scp <workspace-vm>:/mnt/butterflies-fg-2026-05/exports/yolo26s-v2-web/export-*/best.onnx \
    research/leps_localizer/demo/model/yolo26s-fg-640.onnx
  ```

## Run locally

ONNX must be served over HTTP (file:// won't work — `fetch` blocked, WASM
needs same-origin). Use the included server — it sends COOP/COEP headers,
which unlocks `SharedArrayBuffer` and lets onnxruntime-web run WASM threads
(~3-4× faster than single-thread):

```bash
cd research/leps_localizer/demo
python serve.py 8000
# open http://localhost:8000
```

`python -m http.server` works too but won't set the headers, so inference
falls back to single-threaded WASM (~1.5-2 s/frame at 640).

## Notes

- **Backend**: WASM SIMD with threading. WebGL/WebGPU skipped — coverage
  of YOLO26 ops in those ORT backends is incomplete as of 1.20.x.
- **No NMS in JS**: YOLO26 is end2end / NMS-free. Output is already
  confidence-sorted top-300. Just threshold + clamp.
- **Letterbox**: gray (114) padding, identical recipe to Ultralytics.
- **Input shape**: 640×640 — matches the export. Re-export at different
  imgsz to change.
- **First load**: ~37 MB ONNX. Browser caches after first fetch; cold
  load ~2-4 s on a fast connection, warm load instant.
- **Inference latency** (WASM SIMD, modern laptop CPU): ~150-300 ms per
  frame at 640. Webcam runs sub-realtime; usable but not 60 fps.

## TF.js?

The Ultralytics `--formats tfjs` path was skipped because the workspace
VM is on Python 3.13 and `tensorflowjs` wheels are gated to `< 3.13`.
ONNX Runtime Web is the simpler path anyway — fewer conversion steps,
direct support for the YOLO graph. If you want TF.js, run the export
in a Python 3.12 venv on a different host.
