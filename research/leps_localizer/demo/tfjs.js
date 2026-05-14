// TFJS branch of the leps localizer demo. YOLO26 nano via TFJS graph
// model. Same end2end head as the ONNX path — output [1, 300, 6] of
// [x1, y1, x2, y2, conf, cls] in letterboxed-pixel coords. No JS NMS
// needed.
//
// Differences vs the ORT path:
//   - Input layout NHWC, not NCHW
//   - Different runtime: @tensorflow/tfjs (~440 KB brotli + ~600 KB WASM
//     if wasm backend chosen)
//   - Sharded weights: model.json + N x groupN-shardKofM.bin

const MODELS = {
  uint8: "model/tfjs/uint8/model.json",
  fp16: "model/tfjs/fp16/model.json",
};
const IMGSZ = 640;
const CLASS_NAMES = ["butterfly"];

const els = {
  file: document.getElementById("file"),
  webcamBtn: document.getElementById("webcam-btn"),
  stopBtn: document.getElementById("stop-btn"),
  status: document.getElementById("status"),
  conf: document.getElementById("conf"),
  maxdet: document.getElementById("maxdet"),
  confV: document.getElementById("conf-v"),
  maxdetV: document.getElementById("maxdet-v"),
  model: document.getElementById("model"),
  backend: document.getElementById("backend"),
  cv: document.getElementById("cv"),
  cam: document.getElementById("cam"),
  metaIn: document.getElementById("meta-in"),
  metaMs: document.getElementById("meta-ms"),
  metaN: document.getElementById("meta-n"),
  metaEp: document.getElementById("meta-ep"),
  canvasWrap: document.querySelector(".canvas-wrap"),
};

const ctx = els.cv.getContext("2d");
let model = null;
let webcamStream = null;
let webcamLoop = null;
let lastFrame = null;

async function ensureBackend(name) {
  if (tf.getBackend() === name) return;
  const ok = await tf.setBackend(name);
  if (!ok) {
    throw new Error(`backend ${name} not available in this browser`);
  }
  await tf.ready();
}

// First inference on a fresh WebGL context compiles all GLSL shaders
// (~1-3 s of stutter). Run a zero-tensor pass right after load so the
// user's first real frame isn't the warm-up frame.
async function warmup() {
  if (!model) return;
  const zero = tf.zeros([1, IMGSZ, IMGSZ, 3]);
  const out = model.execute(zero);
  await out.data();
  out.dispose();
  zero.dispose();
}

async function loadModel(key, backendName) {
  els.status.textContent = `loading ${key} on ${backendName}…`;
  els.status.className = "";
  if (model) {
    model.dispose();
    model = null;
  }
  try {
    await ensureBackend(backendName);
    const t0 = performance.now();
    model = await tf.loadGraphModel(MODELS[key]);
    const loadMs = (performance.now() - t0).toFixed(0);
    els.status.textContent = `warming up ${backendName}…`;
    const tw = performance.now();
    await warmup();
    const warmMs = (performance.now() - tw).toFixed(0);
    els.status.textContent = `ready (${key}, ${backendName}, load=${loadMs} ms, warm=${warmMs} ms)`;
    els.status.className = "ready";
    els.metaEp.textContent = `tfjs ${tf.version_core}, ${tf.getBackend()}`;
    if (lastFrame) rerun();
  } catch (err) {
    console.error(err);
    els.status.textContent = `load failed: ${err.message}`;
    els.status.className = "error";
  }
}

// Letterbox NHWC tensor. Returns {tensor:[1,640,640,3], scale, padX, padY,
// srcW, srcH}. Uses an offscreen canvas + tf.browser.fromPixels for the
// fast path.
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

  const tensor = tf.tidy(() =>
    tf.browser.fromPixels(off).toFloat().div(255).expandDims(0)
  );
  return { tensor, scale: r, padX, padY, srcW, srcH };
}

function postprocess(out, pre, confTh, maxDet) {
  const N = out.shape[1];
  const dets = [];
  for (let i = 0; i < N; i++) {
    const base = i * 6;
    const x1 = out.data[base];
    const y1 = out.data[base + 1];
    const x2 = out.data[base + 2];
    const y2 = out.data[base + 3];
    const conf = out.data[base + 4];
    const cls = out.data[base + 5] | 0;
    if (conf < confTh) continue;
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
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
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

// Translucent banner + spinner ring overlaid on the canvas while inference
// is running. Sized relative to the canvas so it works on big and small
// source images.
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

let inferenceInFlight = false;

async function runOnce(source, srcW, srcH, opts = {}) {
  if (!model) return;
  if (inferenceInFlight && !opts.fromWebcam) return;
  inferenceInFlight = true;

  const confTh = parseFloat(els.conf.value);
  const maxDet = parseInt(els.maxdet.value, 10);

  // For pick-from-file: draw image + loading banner BEFORE running
  // inference, so the user sees the picked frame instantly even if
  // detection takes 1-3 s. Webcam loop skips this — its frames already
  // animate from the <video>.
  if (!opts.fromWebcam) {
    drawSource(source, srcW, srcH);
    drawLoadingOverlay(opts.label || "running detection…");
    els.metaIn.textContent = `${srcW}×${srcH}`;
    els.metaMs.textContent = "…";
    els.metaN.textContent = "…";
    // Yield once so the canvas paints before we block on inference.
    await new Promise((r) => requestAnimationFrame(r));
  }

  const pre = preprocess(source, srcW, srcH);
  const t0 = performance.now();
  let arr;
  try {
    const outTensor = model.execute(pre.tensor);
    arr = await outTensor.data();
    outTensor.dispose();
  } finally {
    pre.tensor.dispose();
  }
  const t1 = performance.now();

  const out = { data: arr, shape: [1, arr.length / 6, 6] };
  const dets = postprocess(out, pre, confTh, maxDet);

  drawSource(source, srcW, srcH);
  drawDetections(dets);

  els.metaIn.textContent = `${srcW}×${srcH}`;
  els.metaMs.textContent = `${(t1 - t0).toFixed(1)} ms`;
  els.metaN.textContent = `${dets.length}`;
  lastFrame = { source, srcW, srcH };
  inferenceInFlight = false;
}

function rerun() {
  if (lastFrame) {
    runOnce(lastFrame.source, lastFrame.srcW, lastFrame.srcH).catch(
      console.error
    );
  }
}

async function handleFile(file) {
  if (!file) return;
  const bitmap = await createImageBitmap(file);
  await runOnce(bitmap, bitmap.width, bitmap.height);
}

els.file.addEventListener("change", (e) => handleFile(e.target.files[0]));

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
bindSlider(els.maxdet, els.maxdetV);

els.model.addEventListener("change", () =>
  loadModel(els.model.value, els.backend.value)
);
els.backend.addEventListener("change", () =>
  loadModel(els.model.value, els.backend.value)
);

loadModel(els.model.value, els.backend.value);
