// Leps Localizer browser demo - YOLO26-s NMS-free single-class detector.
//
// Model output shape: [1, 300, 6] where 6 = [x1, y1, x2, y2, conf, cls]
// in pixel coords relative to letterboxed 640x640 input. End2end / NMS-free,
// so no JS-side NMS — just confidence threshold + max-det clamp.

const MODELS = {
  "yolo26n-fp16": "model/yolo26n-fg-640.fp16.onnx",
  "yolo26n": "model/yolo26n-fg-640.onnx",
  "yolo26s": "model/yolo26s-fg-640.onnx",
};
const IMGSZ = 640;
const CLASS_NAMES = ["butterfly"];

const els = {
  file: document.getElementById("file"),
  webcamBtn: document.getElementById("webcam-btn"),
  stopBtn: document.getElementById("stop-btn"),
  status: document.getElementById("status"),
  conf: document.getElementById("conf"),
  iou: document.getElementById("iou"),
  maxdet: document.getElementById("maxdet"),
  confV: document.getElementById("conf-v"),
  iouV: document.getElementById("iou-v"),
  maxdetV: document.getElementById("maxdet-v"),
  model: document.getElementById("model"),
  cv: document.getElementById("cv"),
  cam: document.getElementById("cam"),
  metaIn: document.getElementById("meta-in"),
  metaMs: document.getElementById("meta-ms"),
  metaN: document.getElementById("meta-n"),
  metaEp: document.getElementById("meta-ep"),
  canvasWrap: document.querySelector(".canvas-wrap"),
};

const ctx = els.cv.getContext("2d");
let session = null;
let inputName = null;
let webcamStream = null;
let webcamLoop = null;
let lastFrame = null; // {bitmap|video, w, h}

// ---------- model load ----------
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
ort.env.wasm.simd = true;

async function loadModel(key) {
  els.status.textContent = `loading ${key}…`;
  els.status.className = "";
  session = null;

  const url = MODELS[key];
  if (!url) {
    els.status.textContent = `unknown model: ${key}`;
    els.status.className = "error";
    return;
  }

  const t0 = performance.now();
  try {
    session = await ort.InferenceSession.create(url, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    inputName = session.inputNames[0];
    const ms = (performance.now() - t0).toFixed(0);
    els.status.textContent = `ready (${key}, ${ms} ms load) — pick an image or start webcam`;
    els.status.className = "ready";
    els.metaEp.textContent = `wasm-simd, threads=${ort.env.wasm.numThreads}`;
    if (lastFrame) rerun();
  } catch (err) {
    console.error(err);
    els.status.textContent = `model load failed: ${err.message}`;
    els.status.className = "error";
  }
}

// ---------- preprocessing ----------
// Letterbox to 640x640 preserving aspect ratio, pad with gray (114/255).
// Returns {tensor, scale, padX, padY, srcW, srcH}.
function preprocess(source, srcW, srcH) {
  const r = Math.min(IMGSZ / srcW, IMGSZ / srcH);
  const newW = Math.round(srcW * r);
  const newH = Math.round(srcH * r);
  const padX = Math.floor((IMGSZ - newW) / 2);
  const padY = Math.floor((IMGSZ - newH) / 2);

  const off = document.createElement("canvas");
  off.width = IMGSZ;
  off.height = IMGSZ;
  const octx = off.getContext("2d");
  octx.fillStyle = "rgb(114,114,114)";
  octx.fillRect(0, 0, IMGSZ, IMGSZ);
  octx.drawImage(source, 0, 0, srcW, srcH, padX, padY, newW, newH);

  const imageData = octx.getImageData(0, 0, IMGSZ, IMGSZ).data;
  // Channels-first float32, normalized 0-1.
  const n = IMGSZ * IMGSZ;
  const data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    data[i] = imageData[i * 4] / 255;          // R
    data[n + i] = imageData[i * 4 + 1] / 255;  // G
    data[2 * n + i] = imageData[i * 4 + 2] / 255; // B
  }
  const tensor = new ort.Tensor("float32", data, [1, 3, IMGSZ, IMGSZ]);
  return { tensor, scale: r, padX, padY, srcW, srcH };
}

// ---------- postprocessing ----------
// YOLO26 end2end output [1, 300, 6] — already NMS-free, confidence-sorted.
// We just threshold and clamp to maxdet.
function postprocess(output, pre, confTh, maxDet) {
  const data = output.data;
  const dims = output.dims; // [1, N, 6]
  const N = dims[1];
  const dets = [];
  for (let i = 0; i < N; i++) {
    const base = i * 6;
    const x1 = data[base];
    const y1 = data[base + 1];
    const x2 = data[base + 2];
    const y2 = data[base + 3];
    const conf = data[base + 4];
    const cls = data[base + 5] | 0;
    if (conf < confTh) continue;
    // Reverse letterbox.
    const ox1 = (x1 - pre.padX) / pre.scale;
    const oy1 = (y1 - pre.padY) / pre.scale;
    const ox2 = (x2 - pre.padX) / pre.scale;
    const oy2 = (y2 - pre.padY) / pre.scale;
    dets.push({
      x1: Math.max(0, ox1),
      y1: Math.max(0, oy1),
      x2: Math.min(pre.srcW, ox2),
      y2: Math.min(pre.srcH, oy2),
      conf,
      cls,
    });
    if (dets.length >= maxDet) break;
  }
  return dets;
}

// ---------- rendering ----------
function drawSource(source, srcW, srcH) {
  els.cv.width = srcW;
  els.cv.height = srcH;
  ctx.drawImage(source, 0, 0, srcW, srcH);
}

function drawDetections(dets) {
  const w = els.cv.width;
  const fontPx = Math.max(12, Math.round(w / 60));
  ctx.lineWidth = Math.max(2, Math.round(w / 400));
  ctx.strokeStyle = "rgb(74, 222, 128)";
  ctx.fillStyle = "rgb(74, 222, 128)";
  ctx.font = `${fontPx}px -apple-system, BlinkMacSystemFont, sans-serif`;

  for (const d of dets) {
    const bw = d.x2 - d.x1;
    const bh = d.y2 - d.y1;
    ctx.strokeRect(d.x1, d.y1, bw, bh);
    const label = `${CLASS_NAMES[d.cls] || "obj"} ${d.conf.toFixed(2)}`;
    const tw = ctx.measureText(label).width + 8;
    const th = fontPx + 4;
    ctx.fillRect(d.x1, Math.max(0, d.y1 - th), tw, th);
    ctx.save();
    ctx.fillStyle = "#0f1115";
    ctx.fillText(label, d.x1 + 4, Math.max(fontPx, d.y1 - 4));
    ctx.restore();
  }
}

function drawLoadingOverlay(label) {
  const w = els.cv.width;
  const h = els.cv.height;
  const padY = Math.max(40, Math.round(h * 0.08));
  ctx.save();
  ctx.fillStyle = "rgba(15, 17, 21, 0.55)";
  ctx.fillRect(0, 0, w, padY);
  const fontPx = Math.max(14, Math.round(w / 50));
  ctx.font = `${fontPx}px -apple-system, BlinkMacSystemFont, sans-serif`;
  ctx.fillStyle = "rgb(74, 222, 128)";
  ctx.textBaseline = "middle";
  ctx.fillText(label, Math.round(w * 0.02), padY / 2);
  ctx.restore();
}

// ---------- inference ----------
let inferenceInFlight = false;

async function runOnce(source, srcW, srcH, opts = {}) {
  if (!session) return;
  if (inferenceInFlight && !opts.fromWebcam) return;
  inferenceInFlight = true;

  const confTh = parseFloat(els.conf.value);
  const maxDet = parseInt(els.maxdet.value, 10);

  if (!opts.fromWebcam) {
    drawSource(source, srcW, srcH);
    drawLoadingOverlay(opts.label || "running detection…");
    els.metaIn.textContent = `${srcW}×${srcH}`;
    els.metaMs.textContent = "…";
    els.metaN.textContent = "…";
    await new Promise((r) => requestAnimationFrame(r));
  }

  const pre = preprocess(source, srcW, srcH);
  const t0 = performance.now();
  const outputs = await session.run({ [inputName]: pre.tensor });
  const t1 = performance.now();
  const outName = session.outputNames[0];
  const dets = postprocess(outputs[outName], pre, confTh, maxDet);

  drawSource(source, srcW, srcH);
  drawDetections(dets);

  els.metaIn.textContent = `${srcW}×${srcH}`;
  els.metaMs.textContent = `${(t1 - t0).toFixed(1)} ms`;
  els.metaN.textContent = `${dets.length}`;

  lastFrame = { source, srcW, srcH };
  inferenceInFlight = false;
}

// Re-run with current sliders against last frame.
function rerun() {
  if (lastFrame) {
    runOnce(lastFrame.source, lastFrame.srcW, lastFrame.srcH).catch(console.error);
  }
}

// ---------- input handling ----------
async function handleFile(file) {
  if (!file) return;
  const bitmap = await createImageBitmap(file);
  await runOnce(bitmap, bitmap.width, bitmap.height);
}

els.file.addEventListener("change", (e) => handleFile(e.target.files[0]));

// Drag-and-drop.
["dragenter", "dragover"].forEach((ev) =>
  els.canvasWrap.addEventListener(ev, (e) => {
    e.preventDefault();
    els.canvasWrap.classList.add("drop-active");
  })
);
["dragleave", "drop"].forEach((ev) =>
  els.canvasWrap.addEventListener(ev, (e) => {
    e.preventDefault();
    els.canvasWrap.classList.remove("drop-active");
  })
);
els.canvasWrap.addEventListener("drop", (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f) handleFile(f);
});

// ---------- webcam ----------
async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: 1280, height: 720 },
      audio: false,
    });
  } catch (err) {
    els.status.textContent = `webcam: ${err.message}`;
    els.status.className = "error";
    return;
  }
  els.cam.srcObject = webcamStream;
  await els.cam.play();
  els.webcamBtn.disabled = true;
  els.stopBtn.disabled = false;

  const loop = async () => {
    if (!webcamStream) return;
    if (els.cam.readyState >= 2 && !document.hidden) {
      await runOnce(els.cam, els.cam.videoWidth, els.cam.videoHeight, {
        fromWebcam: true,
      });
    }
    webcamLoop = requestAnimationFrame(loop);
  };
  loop();
}

function stopWebcam() {
  if (webcamLoop) cancelAnimationFrame(webcamLoop);
  webcamLoop = null;
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
  els.cam.srcObject = null;
  els.webcamBtn.disabled = false;
  els.stopBtn.disabled = true;
  lastFrame = null;
}

els.webcamBtn.addEventListener("click", startWebcam);
els.stopBtn.addEventListener("click", stopWebcam);

// ---------- slider wiring ----------
function bindSlider(input, label) {
  const update = () => {
    label.textContent = parseFloat(input.value).toFixed(
      input.step.includes(".") ? 2 : 0
    );
    rerun();
  };
  input.addEventListener("input", update);
  update();
}
bindSlider(els.conf, els.confV);
bindSlider(els.iou, els.iouV);
bindSlider(els.maxdet, els.maxdetV);

els.model.addEventListener("change", (e) => loadModel(e.target.value));

// ---------- init ----------
loadModel(els.model.value);
