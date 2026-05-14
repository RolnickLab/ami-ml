# eButterfly Integration: TFJS Auto-Localizer

This document covers shipping the YOLO26-nano butterfly detector into the
eButterfly upload flow as a drop-in auto-suggestion for the existing manual
crop component. The detector runs entirely client-side via TensorFlow.js.

Reference implementation: `research/leps_localizer/demo/tfjs.{html,js}`.

## TL;DR

- Ship the `uint8` TFJS bundle: 4 files, ~2.5 MB uncompressed / ~1.7 MB on
  the wire with Brotli. R_all = 0.862 on the Leeds eval set (lossless vs
  FP32).
- Wire YOLO26 output `[1, 300, 6]` of `[x1, y1, x2, y2, conf, cls]` (in
  letterboxed pixel coords) into the crop component as the initial bbox.
  The user can still drag corners to override.
- **Resize the source image to 1280 px max edge before inference.** This is
  the single biggest perf win on phones — both for CPU and WebGL.
- Target backend: WebGL 2 (~97% of phones in 2026). CPU fallback only as a
  diagnostic — it is too slow for real interactive use.

---

## 1. What to host

Copy the contents of `research/leps_localizer/demo/model/tfjs/uint8/` to
your CDN. Four files:

```
uint8/
├── model.json                  # 602 KB  (39 KB brotli)
├── group1-shard1of1.bin        # 2.4 MB  (1.7 MB brotli)
├── model.json.br               # pre-compressed brotli (q=11)
└── group1-shard1of1.bin.br
```

Total wire size when served with `Content-Encoding: br`: **~1.7 MB**.
Cold-load including the `@tensorflow/tfjs@4.22.0` UMD bundle (~360 KB
brotli): **~2.0 MB**.

### Serving requirements

- **Brotli pre-compression**: serve the `.br` file when the client sends
  `Accept-Encoding: br`. Set `Content-Encoding: br` and the same MIME type
  as the uncompressed file (`application/json` for `model.json`,
  `application/octet-stream` for shards). See `demo/serve.py` for a
  reference implementation.
- **CORS**: if the model is on a separate origin from the page, add
  `Access-Control-Allow-Origin: *` (or your eButterfly origin) on the model
  files.
- **Cache headers**: model files are content-addressed by the shard name in
  `model.json`, so `Cache-Control: public, max-age=31536000, immutable` is
  safe. The version of `tfjs` itself should be pinned via the script tag.

### FP16 variant

Also available at `tfjs/fp16/` (~4 MB brotli). Same recall, no perf
advantage — `uint8` weights dequantize to FP32 at load time so runtime
math is identical. Don't ship FP16 unless you have a specific reason.

---

## 2. Loading the model

```html
<script
  src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"
  crossorigin="anonymous"
></script>
```

Or via npm:

```bash
npm install @tensorflow/tfjs@4.22.0
```

```ts
import * as tf from "@tensorflow/tfjs";

const MODEL_URL = "https://cdn.ebutterfly.org/models/yolo26n-uint8/model.json";
const IMGSZ = 640;

let model: tf.GraphModel | null = null;

export async function initLocalizer(): Promise<{ backend: string; warmMs: number }> {
  await tf.ready();  // tfjs picks best available backend: webgl > cpu
  model = await tf.loadGraphModel(MODEL_URL);

  // Warmup: first inference on a fresh WebGL context compiles all GLSL
  // shaders (~1-3 s of stutter). Burn a zero-tensor pass on init so the
  // user's first real upload isn't the warmup frame.
  const t0 = performance.now();
  const zero = tf.zeros([1, IMGSZ, IMGSZ, 3]);
  const out = model.execute(zero) as tf.Tensor;
  await out.data();
  out.dispose();
  zero.dispose();
  const warmMs = performance.now() - t0;

  return { backend: tf.getBackend(), warmMs };
}
```

Call `initLocalizer()` once per session — ideally on the upload page mount,
not on app boot. The model file is heavy enough that you don't want it
loaded on pages where it won't be used.

If `warmMs > 1500`, the device is too slow to give an instant prediction —
fall back to the manual-only flow (skip the auto-suggestion, keep the
existing crop UI). See section 7.

---

## 3. Running inference

```ts
type Detection = {
  x1: number; y1: number; x2: number; y2: number;  // source-pixel coords
  conf: number; cls: number;
};

export async function detect(
  source: HTMLImageElement | HTMLCanvasElement | ImageBitmap,
  srcW: number,
  srcH: number,
  confThreshold = 0.25
): Promise<Detection[]> {
  if (!model) throw new Error("call initLocalizer() first");

  // Letterbox the source into a 640×640 canvas (grey padding).
  const r = Math.min(IMGSZ / srcW, IMGSZ / srcH);
  const newW = Math.round(srcW * r);
  const newH = Math.round(srcH * r);
  const padX = Math.floor((IMGSZ - newW) / 2);
  const padY = Math.floor((IMGSZ - newH) / 2);

  const off = document.createElement("canvas");
  off.width = IMGSZ;
  off.height = IMGSZ;
  const octx = off.getContext("2d")!;
  octx.fillStyle = "rgb(114,114,114)";
  octx.fillRect(0, 0, IMGSZ, IMGSZ);
  octx.drawImage(source, 0, 0, srcW, srcH, padX, padY, newW, newH);

  // NHWC float tensor, normalized to [0,1].
  const tensor = tf.tidy(() =>
    tf.browser.fromPixels(off).toFloat().div(255).expandDims(0)
  );

  // Output [1, 300, 6] in letterboxed-pixel coords.
  const out = model.execute(tensor) as tf.Tensor;
  const arr = await out.data();
  out.dispose();
  tensor.dispose();

  const dets: Detection[] = [];
  for (let i = 0; i < arr.length; i += 6) {
    const conf = arr[i + 4];
    if (conf < confThreshold) continue;
    dets.push({
      x1: Math.max(0, (arr[i] - padX) / r),
      y1: Math.max(0, (arr[i + 1] - padY) / r),
      x2: Math.min(srcW, (arr[i + 2] - padX) / r),
      y2: Math.min(srcH, (arr[i + 3] - padY) / r),
      conf,
      cls: arr[i + 5] | 0,
    });
  }
  // Highest-confidence first.
  dets.sort((a, b) => b.conf - a.conf);
  return dets;
}
```

The end2end head includes NMS — no JS-side NMS needed.

---

## 4. Adapting the manual crop component

Pattern: the existing manual-crop component already takes a bbox as state.
Treat the detector as a *source* of an initial bbox; the user keeps full
control to override.

```tsx
import { useState, useEffect } from "react";
import { initLocalizer, detect, type Detection } from "./localizer";

type Bbox = { x1: number; y1: number; x2: number; y2: number };

function ButterflyUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [bbox, setBbox] = useState<Bbox | null>(null);
  const [autoSuggested, setAutoSuggested] = useState(false);
  const [status, setStatus] = useState<"idle" | "detecting" | "ready" | "failed">("idle");

  // Init localizer on mount. Stays warm for the session.
  useEffect(() => {
    initLocalizer().then(
      ({ backend, warmMs }) => {
        if (warmMs > 1500) {
          // Too slow — disable auto-suggestion, keep manual flow.
          setStatus("failed");
        } else {
          setStatus("idle");
        }
      },
      () => setStatus("failed")
    );
  }, []);

  async function onFilePicked(f: File) {
    setFile(f);
    setBbox(null);
    setAutoSuggested(false);

    if (status === "failed") return;  // fall through to manual crop

    setStatus("detecting");
    const bitmap = await createImageBitmap(f);
    const origW = bitmap.width;
    const origH = bitmap.height;

    // Downscale BEFORE inference — see section 5.
    const { canvas, w, h } = downscaleForInference(bitmap, 1280);
    bitmap.close();

    const dets = await detect(canvas, w, h);

    if (dets.length > 0 && dets[0].conf >= 0.25) {
      // Rescale detection back to the *original* pixel space if your crop
      // component operates on original image coords.
      const scaleUp = origW / w;
      setBbox({
        x1: dets[0].x1 * scaleUp,
        y1: dets[0].y1 * scaleUp,
        x2: dets[0].x2 * scaleUp,
        y2: dets[0].y2 * scaleUp,
      });
      setAutoSuggested(true);
    }
    setStatus("ready");
  }

  return (
    <>
      <FileInput onChange={onFilePicked} />
      {status === "detecting" && <Spinner label="finding butterfly..." />}
      {file && (
        <ManualCropTool
          src={file}
          bbox={bbox}
          onChange={setBbox}
          // Show a hint chip only when the bbox came from the detector,
          // not when the user has dragged it.
          hint={autoSuggested ? "auto-suggested · drag to refine" : null}
          // User edit -> we no longer claim auto-suggestion provenance.
          onUserEdit={() => setAutoSuggested(false)}
        />
      )}
    </>
  );
}
```

### UX recommendations

- **Always paint the picked image immediately.** Don't make the user wait
  on detection to see their photo. The demo at `tfjs.html` shows this
  pattern — `drawSource()` runs before inference.
- **Show a small overlay or spinner while detecting.** WebGL: 100-300 ms
  on phones, mostly imperceptible but not always. CPU: 3-15 s, definitely
  needs feedback.
- **No detection = fall through to manual.** If `dets` is empty or top
  confidence < threshold, leave `bbox` null. The user crops as usual.
- **Treat the auto-bbox as a suggestion, not a verdict.** Keep the existing
  drag-to-resize handles fully functional. Don't auto-submit.

---

## 5. Resize the source before inference (important)

**Yes — downscale to 1280 px max edge in JS before passing to `detect()`.**
This is the single biggest perf win on phones. It helps both CPU and WebGL,
though for different reasons.

### Why it matters

The detector itself runs on a fixed 640×640 letterboxed tensor — the model
is the same regardless of source size. The cost that scales with source
pixels is the **preprocess path**: `drawImage` into the 640×640 canvas,
`tf.browser.fromPixels` reading the canvas, GPU texture upload.

Modern phones produce 12-50 MP photos (e.g. iPhone 15 Pro: 24 MP / ProRAW
48 MP). Per-frame texture upload at 48 MP vs 1.6 MP (1280×900-ish) is
roughly a **30× pixel reduction**. The savings:

| Stack | 12 MP source | 1.6 MP source (1280 max) |
|---|---|---|
| WebGL preprocess | ~80-200 ms | ~10-30 ms |
| CPU preprocess | ~400-1500 ms | ~50-150 ms |
| Inference (640×640) | unchanged | unchanged |

CPU sees a bigger absolute win because `fromPixels` runs in JS land
(no GPU offload). On WebGL the texture upload itself is the bottleneck.

### Implementation

```ts
function downscaleForInference(
  source: ImageBitmap | HTMLImageElement,
  maxEdge: number
): { canvas: HTMLCanvasElement; w: number; h: number } {
  const srcW = source.width;
  const srcH = source.height;
  const longest = Math.max(srcW, srcH);
  const scale = longest > maxEdge ? maxEdge / longest : 1;
  const w = Math.round(srcW * scale);
  const h = Math.round(srcH * scale);

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { alpha: false })!;
  // imageSmoothingQuality: "medium" is a good speed/quality tradeoff.
  // "high" can be 2-3× slower on some browsers.
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "medium";
  ctx.drawImage(source, 0, 0, w, h);

  return { canvas, w, h };
}
```

Pass `{canvas, w, h}` into `detect()` instead of the original image.

### Recall impact

Negligible for typical eButterfly captures. The model trains on images
where the subject is usually 30-100% of the frame; downscaling from
12 MP → 1.6 MP keeps the subject above the detector's effective
~30 px-on-a-side minimum even for small framings. The Leeds eval set was
re-run after a 1280-cap downscale: R_all dropped from 0.862 to 0.860
(within noise).

The only failure mode is macro shots where the butterfly occupies <5% of
the frame, which is uncommon in citizen-science uploads. If you want a
safety net, you can run a second pass at full resolution only when the
first pass returned zero detections — but this is probably overkill.

### Why not resize even further (e.g. 640 max)?

You can. If the source is already ≤ 640 px, the letterbox path is nearly
free. But going below 1024-ish starts to bite into the recall margin for
modest-subject framings — the antenna and wing-edge detail that
distinguishes a small butterfly from a leaf shadow lives in those pixels.
1280 is a good middle ground.

---

## 6. Backend selection on mobile

Default to letting tfjs auto-pick. The UMD bundle of `@tensorflow/tfjs`
ships `webgl` and `cpu` backends only — it'll pick `webgl` whenever
available, fall back to `cpu` otherwise.

```ts
await tf.ready();
const backend = tf.getBackend();  // "webgl" | "cpu" | "wasm" (if loaded)
```

**Do not force a backend.** The auto-pick is correct in nearly all cases.
Use the warmup time signal (section 2) to decide whether to enable
auto-suggestion, not the backend name.

### WebGPU is opt-in

WebGPU support is **not** in the default `tfjs` UMD bundle. To enable, add
`@tensorflow/tfjs-backend-webgpu` as a separate import. Worth doing only
if perf metrics show WebGL is a bottleneck — typically WebGPU is 1.3-2×
faster than WebGL for this graph size, but the runtime is heavier and
not all phones have it (see section 7).

### WASM backend

Available via `@tensorflow/tfjs-backend-wasm`. Faster than `cpu` (~2-3×)
but slower than `webgl` (~10-30×). Useful as a fallback for phones with
broken WebGL drivers, but in 2026 those are rare enough that maintaining
the WASM build is usually not worth it.

---

## 7. Real-world mobile browser support (2026)

**WebGL 1.0**: ~99% of smartphones globally. Universally supported on iOS
8+ and Android Chrome since launch.

**WebGL 2.0** (required by tfjs 4.x): ~97% of smartphones in 2026.

- iOS Safari: 15+ (released Sept 2021). Anything still running iOS 14 is
  a vanishingly small slice.
- Android Chrome: full support since Chrome 56 (early 2017).
- Older Android WebView shells (in-app browsers on older phones): WebGL 2
  can be flaky. This is the main "warm-up was slow" failure mode.

**WebGPU**: ~75-85% of smartphones in 2026, still climbing.

- Android Chrome: stable on most devices since Chrome 121 (Jan 2024).
- iOS Safari: enabled by default in iOS 26 (Sept 2025).
- Some Android budget devices on Mali GPUs without Vulkan are excluded.

### Recommended target

- **Primary**: WebGL 2 with the standard `@tensorflow/tfjs` UMD bundle.
  Covers ~97% of phones.
- **Fallback**: when warmup exceeds 1500 ms, hide the auto-suggestion and
  keep the existing manual crop UI. This handles the long tail of weak
  GPUs, in-app browsers, and outright WebGL failures.
- **No need to ship WebGPU yet.** Worth revisiting in 12 months when
  coverage is >95%.

### Caveats

- **In-app browsers** (Facebook, Instagram, WeChat, etc.) sometimes have
  WebGL disabled or run on a different rendering path. Mitigation: the
  warmup-time check catches most of these and falls through to manual.
- **iOS low-power mode** throttles GPU. Inference may take 3-5× longer
  than steady-state. Not worth special-casing — the warmup detector
  handles it.
- **Bytecode caching**: subsequent visits load the cached shaders, so
  warmup drops from ~1-3 s to ~100-300 ms. The auto-suggestion will feel
  much snappier on return visits.

---

## 8. Testing checklist

Before shipping, exercise on real devices in this priority order:

1. **iOS Safari, recent iPhone** (e.g. iPhone 13+). Should be the smoothest
   experience — WebGL warmup ~500 ms, inference ~50-100 ms.
2. **Android Chrome, mid-range phone** (e.g. Pixel 6a, Galaxy A series).
   WebGL warmup ~1-2 s, inference ~80-200 ms.
3. **Android Chrome, low-end phone or in-app browser**. Validate the
   "warmup > 1500 ms → manual fallback" path actually triggers and the
   manual crop UI works as before.
4. **Desktop Chrome / Safari**. Should be near-instant. Useful for
   debugging the React component itself.

For each device, verify:

- [ ] First upload after page load runs without visible stutter.
- [ ] Auto-bbox is approximately correct (overlaps the butterfly).
- [ ] Dragging the bbox handles works as before.
- [ ] No bbox is shown when the image contains no butterfly.
- [ ] The manual crop flow is unbroken when the localizer fails / is slow.
- [ ] Network panel shows ~1.7 MB Brotli for model files (not 2.5 MB
      uncompressed).

---

## 9. Followups / out of scope

- **Quality tier classification**: The detector returns one class
  (`butterfly`). If you want family/species hints, that's a separate model
  (the `na-butterflies-v3` / `global-butterflies-max1000img-512`
  classifier). Apply it to the cropped region *after* the user confirms
  the bbox.
- **Multi-instance images**: The model can return up to 300 detections.
  The reference component picks the highest-confidence one. If eButterfly
  supports multi-individual reports, surface all `dets` and let the user
  pick.
- **Server-side fallback**: For browsers where the localizer fails
  entirely, you could POST the image to a server-side inference endpoint.
  Probably not worth building unless metrics show this is a real problem
  for a meaningful fraction of users.
