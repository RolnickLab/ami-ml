# Leps Localizer — Methods & Results

Single-class arthropod (butterfly) localization detector. Stage-3 of a three-stage pipeline (Detector → Localizer → Classifier). The localizer's job: given a 1-N-arthropod scene, return tight bboxes per individual that the classifier can crop to a square and identify to species.

This document describes (a) the dataset construction, (b) each model architecture compared, (c) the eval protocol and its known biases, and (d) all results to date. The companion CSV (`results.csv`) is the data table for Google Sheets.

---

## 1. Dataset

### 1.1 Training set: `butterflies-fg-train-2026-05`

- **Source**: 3 Fieldguide eval sets repartitioned into a single train+val split, seed=20260505
  - `leps-butterflies-500` (999 imgs, full distribution)
  - `leps-butterflies-small-1000` (999 imgs, bbox area frac < 0.10)
  - `leps-butterflies-medsmall-1000` (997 imgs, bbox area frac 0.10-0.25)
- **Total unique**: 2,862 images (133 cross-set duplicates dropped)
- **Split**: 2,572 train / 290 val (90/10)
- **Format**: YOLO labels at `/mnt/butterflies-fg-2026-05/yolo/`, single class `arthropod`
- **Bbox-area-fraction distribution**:
  - small (<0.10): well-covered (~970 imgs)
  - medsmall (0.10-0.25): well-covered (~970 imgs)
  - medlarge (0.25-0.50): **underrepresented** — only what the random sample contributed
  - large (>0.50): **underrepresented**

### 1.2 Available FG pool (Butterflies / Papilionoidea clade, snapshot 2026-04-27)

74,944 candidate FG photos with valid `crop_info` in the Butterflies (Papilionoidea, `552e76f291201b5ddbcbf77b`) clade. Bbox-area-fraction distribution:

| frac bucket | count | notes |
|---|---|---|
| 0.01-0.05 | 528 | tiny — hardest detector cases |
| 0.05-0.10 | 1,403 | small |
| 0.10-0.25 | 7,715 | medsmall |
| 0.25-0.50 | 22,714 | **medlarge — underrepresented in current train set** |
| 0.50-0.95 | 37,868 | large |
| ≥0.95 | 4,716 | skipped (subject crops to nearly full frame) |

Used in current train+val: ~2,862 photos (<4% of pool). Substantial headroom for enrichment runs.

The broader **Lepidoptera** (butterflies + moths, `5926f024fd89783b2a721ba8`) clade has 217,914 subtree categories — moths can be added to widen coverage if butterfly-only saturates.

### 1.3 Held-out test set: GAP (no FG test set reserved)

The original detector-dataset spec (2026-04-24) intended **all 4 datasets** (`leps-butterflies-500`, `-small-1000`, `-medsmall-1000`, Leeds) to be held out as eval-only forever. During stage-3 training, the 3 FG datasets were repartitioned into train+val seed=20260505. **Only Leeds remains held-out.**

Implications:
- No FG-style framing test set — can't measure quality on the actual production distribution
- Leeds is biased (see §1.4) so the only held-out metric we have is structurally penalizing correct models
- Comparing checkpoints across architectures relies on Leeds + visual gym inspection, no clean tiebreaker

**Fix when next data pull happens (#48)**: stratify pull by frac bucket, **reserve 1K as locked FG test** before training touches the rest.

### 1.4 Held-out eval set: Leeds-butterflies (832 images)

- Standard butterfly dataset, 10 species, single butterfly per image, mostly large/centered
- **Never seen during training of any model in this comparison**
- GT bboxes: see Bias section below

### 1.5 Bias note: Leeds GT framing

Leeds GT bboxes systematically clip the **antennae** (wing tips are intact). Tight crops cover head + full wing area but exclude antennae. This biases IoU/containment metrics **against** models that produce biologically correct boxes (which would include antennae). Discovered 2026-05-07 by visual inspection in the gym.

Implication: a model that wins on Leeds IoU may be losing for the downstream classifier (antennae are diagnostic). Don't treat Leeds recall@IoU as the ground truth ranking.

---

## 2. Architectures compared

### 2.1 YOLOv11s (Ultralytics, anchor-free, NMS-based)

- **Params**: 9.4M, **GFLOPs**: 21.5
- **Backbone**: CSPDarknet variant with C2f/C3k2 blocks
- **Head**: Decoupled cls/reg with DFL regression
- **Pretrained on**: COCO 80-class detection
- **Mobile/web export**: Native CoreML/TFLite/ONNX. NMS runs on-device.
- **Why included**: Reliable small-fast baseline. Drop-in via `from ultralytics import YOLO`.

### 2.2 YOLO26-s (Ultralytics, NMS-free, Jan 2026 release)

- **Params**: 9.95M, **GFLOPs**: 22.5
- **Backbone**: Refined CSP variant
- **Head**: NMS-free decoder with **STAL** (Self-Adversarial Training) and **ProgLoss** (progressive loss schedule for small objects)
- **Pretrained on**: COCO 80-class detection
- **Mobile/web export**: Native CoreML/TFLite/ONNX. **No on-device NMS needed** — cleaner export, smaller compute graph.
- **Why included**: Drop-in successor to v11s. Better small-object handling in theory. Native mobile target.

### 2.3 RT-DETR-l (Ultralytics, end-to-end DETR)

- **Params**: 33M, **GFLOPs**: 108
- **Backbone**: HGNetv2
- **Head**: Transformer decoder, query-based (300 queries default), end-to-end, **no NMS by design**
- **Pretrained on**: COCO 80-class detection
- **Mobile/web export**: ONNX works; CoreML works as of coremltools 8.x (multi-head attention native ops). Inference is 3-5x slower than YOLO on mobile NPU.
- **Why included**: Real-time DETR baseline. Best for accuracy when compute isn't constrained.

### 2.4 SAHI-tiled YOLOv11s (test-time augmentation)

- Wraps the v11s checkpoint at inference with **Slicing Aided Hyper Inference**
- Splits input image into overlapping tiles, runs detection on each, merges
- **No retraining** — same checkpoint as v11s
- **Why included**: Low-cost way to test if v11s misses small objects in raw inference

### 2.5 DEIM-D-FINE-S (planned, queued for training)

- **Params**: 10M, **GFLOPs**: 25
- **Backbone**: HGNetv2 (smaller than RT-DETR-l)
- **Head**: D-FINE Fine-grained Distribution Refinement + DEIM Improved Matching for faster convergence
- **Pretrained on**: COCO 80-class detection
- **License**: Apache-2.0 (Intellindust-AI-Lab/DEIM repo)
- **Mobile/web export**: ONNX native. CoreML via coremltools.convert(). DETR-class inference cost.
- **Why include**: Current SOTA real-time DETR (CVPR 2025). Smaller than RT-DETR-l but expected to match or beat on accuracy. Successor to RT-DETR-l in our server-side pipeline.

---

## 3. Training protocol

### 3.1 Common settings

- Hardware: Arbutus H100 24GB MIG VM (`ami-workspace-02-gpu`)
- Epochs: 80 (no early stopping triggered for any run)
- Batch: 16
- Patience: 20
- Seed: 20260505 (Ultralytics RNG — independent of split-assignment seed)
- AMP: enabled
- Optimizer: AdamW (Ultralytics auto-determined LR=0.002)
- Logger: Weights & Biases (project: `leps_localizer`)

### 3.2 Per-run hyperparameters

| Run | Arch | imgsz | copy_paste | mosaic | scale | Init |
|---|---|---|---|---|---|---|
| `yolov11s-fg-2026-05` (r1) | YOLOv11s | 1280 | 0.0 | 1.0 | 0.5 | yolo11s.pt (COCO) |
| `yolov11s-fg-2026-05-r2` | YOLOv11s | 1536 | 0.5 | 1.0 | 0.5 | yolo11s.pt (COCO) |
| `rtdetr-l-fg-2026-05` | RT-DETR-l | 640 | n/a | n/a | 0.5 | rtdetr-l.pt (COCO) |
| `yolo26s-fg-2026-05` | YOLO26-s | 1280 | 0.0 | 1.0 | 0.5 | yolo26s.pt (COCO) |
| `yolo26s-fg-2026-05-v2` | YOLO26-s | 1280 | 0.0 | 1.0 | 0.5 | yolo26s.pt (COCO) |
| `deimv2-s-fg-2026-05` (in-progress) | DEIMv2-S (DINOv3-distilled ViT-T) | 640 | n/a | n/a | n/a | deimv2_dinov3_s_coco.pth |

**imgsz=640 for RT-DETR** is intentional: RT-DETR's positional encodings were tuned at 640. Comparing at native imgsz per architecture, not identical-imgsz across.

**copy_paste=0.5 in r2** was a small-object augmentation experiment. Cuts patches from one image and pastes into another. Slightly hurt FG val mAP but slightly improved Leeds containment.

---

## 4. Eval protocol

Eval driver: `src/localization/eval_locked.py` (called by `research/leps_localizer/scripts/eval_on_locked.py` and `eval_sahi.py`).

Per Leeds image:
1. Load image, get GT bbox xyxy
2. Run model with `conf=0.25` (matches gym default behavior at non-tight settings)
3. For each predicted bbox, compute IoU and "containment" (`area(GT ∩ pred) / area(GT)`) vs GT
4. Take the **best-matching predicted bbox** by IoU
5. Record: best_iou, best_containment, n_preds

Aggregates:
- **recall (IoU)**: fraction of GT with at least one pred at IoU ≥ 0.5
- **recall (containment)**: fraction of GT with at least one pred at containment ≥ 0.5
- **mean best IoU**: average of best_iou across all GT
- **mean best containment**: same, for containment
- **missed completely**: GT images where the model produced zero predictions at conf=0.25
- **per-bucket recall**: same metrics broken down by bbox-area-fraction (small/medsmall/medlarge/large)

The recall@containment metric is generally more informative than recall@IoU for downstream-classification because the classifier needs the subject inside the crop, not necessarily in a tight box.

---

## 5. Results

### 5.1 Training-time metrics (FG val, 290 images)

Final-epoch metrics from `runs/<name>/results.csv`:

| Run | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| yolov11s r1 (imgsz=1280) | 0.892 | 0.551 | 0.877 | 0.834 |
| yolov11s r2 (imgsz=1536, cp=0.5) | 0.870 | 0.508 | 0.865 | 0.840 |
| rtdetr-l (imgsz=640) | 0.898 | 0.506 | 0.875 | 0.872 |
| yolo26s (imgsz=1280) | 0.896 | 0.554 | 0.867 | 0.879 |
| **yolo26s v2 (imgsz=1280, re-run)** | **0.939** | **0.637** | **0.896** | 0.871 |

YOLO26-s v2 (the re-run with the same recipe as v1) is the new best on FG val: **0.939 mAP50 / 0.637 mAP50-95**, +4-8pp over v1. Same epoch count, same imgsz, same init weights — improvement comes from training stochasticity / EMA divergence between runs. RT-DETR-l previously held best plain mAP50 (0.898); v2 now leads both metrics.

### 5.2 Held-out Leeds eval (832 images, conf=0.25)

| Run | recall@IoU0.5 | recall@cont0.5 | mean IoU | mean cont | missed |
|---|---|---|---|---|---|
| yolov11s r1 | 0.697 | 0.730 | 0.558 | 0.602 | 20 |
| yolov11s r2 | 0.703 | **0.785** | 0.532 | **0.635** | 53 |
| yolov11s r2 + SAHI 768/0.2 | **0.720** | 0.774 | 0.553 | 0.627 | 51 |
| yolov11s r2 + SAHI 1024/0.3 | **0.720** | 0.770 | 0.552 | 0.624 | 53 |
| rtdetr-l | 0.672 | 0.681 | **0.567** | 0.573 | **1** |
| yolo26s | 0.676 | 0.690 | 0.553 | 0.571 | 26 |
| **yolo26s v2** | **0.692** | **0.698** | **0.574** | **0.584** | **9** |

### 5.3 Read

- **By Leeds metrics**: v11s r2 + SAHI wins recall@IoU (0.720); v11s r2 wins recall@cont (0.785).
- **But**: Leeds GT clips antennae. Models that include antennae (RT-DETR, YOLO26) are penalized on Leeds metrics.
- **Per-model framing observed in gym**:
  - RT-DETR-l: most inclusive — antennae intact, full wings (rectangular boxes, biologically correct)
  - YOLO26-s: antennae intact, slight wing-tip clipping (tight rectangular)
  - YOLOv11s: clips wing tips, antennae intact (near-square boxes)
- **For server-side bulk annotation pipeline**: RT-DETR-l is the right choice — most inclusive boxes, downstream classifier gets antennae + full wings. Visual gym inspection confirms: RT-DETR clearly wins on smaller butterflies.
- **For mobile/web app**: YOLO26-s — NMS-free, native CoreML, similar accuracy to v11s, smaller compute graph.

### 5.4 The unbiased eval (#46, planned)

End-to-end pipeline eval — **localizer → square crop → classifier accuracy**. The classifier is the existing FG production classifier (truth set = Fieldguide tagged photos). This treats Leeds-style framing biases as noise: what matters is whether the classifier can identify the cropped subject, not whether the box matches Leeds GT. This will be the unbiased metric for ranking architectures.

---

## 6. Open questions and next steps

1. **Train DEIM-D-FINE-S** on current dataset (queued, task #51). DEIM v1 blocked on H100L MIG (NCCL + torchvision-v2-transforms); superseded by **DEIMv2-S** which is currently training (2026-05-09, wandb run `4rivhpkt` in `moth-ai/leps_localizer`). Compare DEIMv2-S vs RT-DETR-l vs YOLO26-s v2 on Leeds + E2E classifier-acc eval.
2. **Pull medlarge data** (#48): 1-2K more Lepidoptera images with bbox area fraction 0.25-0.50 to fill the distribution gap. Done in chroma-backend repo where Azure DB access lives.
3. **Repartition** the enriched set (#49) and retrain best-of-class on each tier (mobile + server).
4. **E2E classifier-acc eval** (#46): the unbiased ranking. Should override Leeds metrics.
5. **CoreML / ONNX exports** (#38): YOLO26-s native, RT-DETR/DEIM via ONNX → coremltools.

---

## 7. Sources / reference URLs

- Ultralytics docs: https://docs.ultralytics.com/
- DEIM (Intellindust-AI-Lab): https://github.com/Intellindust-AI-Lab/DEIM
- D-FINE (Peterande): https://github.com/Peterande/D-FINE
- DEIM paper: https://arxiv.org/abs/2412.04234
- SAHI: https://github.com/obss/sahi
- Leeds-butterflies dataset: J. Wang et al, 2009
- Gym: https://leps-localizer.dev.antenna.insectai.org

---

*Last updated: 2026-05-08*
