# NEXT SESSION — leps localizer

## Where we are (2026-05-08, evening)

**In-flight on workspace VM (`ami-workspace-02-gpu`):**
- YOLO26-s v2 retrain (`yolo26s-fg-2026-05-v2`), PID 30709 — epoch 57/80 at imgsz=1280, batch=16. ETA ~70 min. Source: `/mnt/butterflies-fg-2026-05/yolo/` (8,673-image set, supplement A merged).
- DEIM-D-FINE-S waiter, PID 31283 — sleeping in `bash /tmp/launch_deim_after_yolo.sh`. Launches when YOLO26 finishes.
- Gym (PID 23141) live since 06:00 with 3 models (v11s-r2, yolo26s, rtdetr-l) on port 7860.

**Verify before acting:** see `.claude/notes/workspace-vm.md` "Live state check" snippet. PIDs and progress drift fast.

**Pending plans (not yet started):**
- Mobile/JS export CLI (CoreML/ONNX/TFJS) at `research/leps_localizer/scripts/export_model.py` — task #38
- Batch BQ inference over 10.7M `training_images` writing to planned `localizer_eval_results` + MERGE into `training_images.primary_subject_bbox`. See `.claude/skills/bigquery-leps/`.

## New context for this branch (added 2026-05-08)

- **BigQuery is source-of-truth.** Training set: `leps-ai.global_butterflies_2604.localizer_training_images` (9,628 FG rows). Sibling: `training_images` (10.7M iNat/eButterfly/museum). Schema, MERGE, two-host workflow → `.claude/skills/bigquery-leps/SKILL.md`.
- **Local infra notes** at `.claude/notes/` (gitignored). Read these before asking about infra.
- **`gcloud` only on BEAST**, not workspace VM. Move files via rsync.

4 trained runs + 2 SAHI test-time configs. Leeds eval done. Gym live at https://leps-localizer.dev.antenna.insectai.org with 3 models (yolov11s r2, yolo26s, rtdetr-l).

Methods doc + CSV committed: `research/leps_localizer/METHODS_AND_RESULTS.md`, `research/leps_localizer/results.csv`.

### Headline finding
Leeds GT bboxes clip antennae+wingtips. v11s "wins" Leeds metrics by being wrong in the same direction as GT. RT-DETR-l produces biologically correct boxes but is penalized. **Leeds metrics ≠ downstream-classification quality.** Need E2E classifier-acc eval (#46) for unbiased ranking.

### Real gap
We violated the original locked-eval constraint — 3 of 4 FG eval sets got pulled into train+val. **No FG-style held-out test set exists.** Only Leeds is held-out. Fix when pulling more data: reserve 1K as locked test.

## Next concrete step: train DEIM-D-FINE-S (task #51)

Repo: `Intellindust-AI-Lab/DEIM` (Apache-2.0). NOT in Ultralytics — standalone repo.

Steps:
1. Clone DEIM repo on `ami-workspace-02-gpu`
2. Convert `/mnt/butterflies-fg-2026-05/yolo/` labels → COCO JSON. Write small converter (Ultralytics has `convert_coco` in reverse; or use `globox` / `pylabel`).
3. Custom config YAML based on `configs/deim_dfine/deim_hgnetv2_s_*.yml`:
   - `num_classes: 1`
   - `remap_mscoco_category: False` (CRITICAL)
   - point dataset paths at COCO-converted FG butterfly set
4. Download `deim_dfine_s_coco.pth` (Google Drive — link in repo README)
5. `torchrun --nproc_per_node=1 train.py -c <cfg> -t deim_dfine_s_coco.pth --use-amp --seed=0`
6. Estimate: ~1.5-3hr at bs=8/imgsz=640 on H100 24GB MIG. Try bs=16 first.
7. Eval on Leeds with same harness as RT-DETR-l (need DEIM predict_fn for `eval_locked.py`)
8. Add to gym (will need a new `_build_deim_predictor` since not in Ultralytics shared interface)

## After DEIM

- Pull 4-6K medlarge FG data + reserve 1K as locked FG test (#48 + new task)
- Repartition + retrain best-of-class (DEIM for server, YOLO26 for mobile)
- E2E classifier-acc eval (#46) — the unbiased ranking
- CoreML export YOLO26 (#38) — script done at `research/leps_localizer/scripts/export_model.py`, ONNX+CoreML verified on yolo26s v1 (2026-05-08). Re-run on v2 best.pt when training finishes.
- Classifier edge export — sibling script at `research/leps_localizer/scripts/export_classifier.py` (timm-based, not Ultralytics). Verified 2026-05-08 on `mohammedelabbas/global-butterflies-max1000img-512` (resnet50, 8851 classes, 512×512). Two output patterns: image-native (default — `ImageType` + `ClassifierConfig`, normalization baked) and `--raw-output` (LepsAI-compatible — `TensorType` "input"→"logits", app does norm). Sidecar files: `<model-id>-category-map.json` (enriched format) always emitted; `--ios-bundle` adds `<model-id>.model-info.json` template.
  - **Validation:** `research/leps_localizer/scripts/validate_export.py` — cross-format parity check (PyTorch ↔ ONNX ↔ TorchScript). Latest run on `--raw-output` export: ONNX max |Δlogit| = 3.05e-04, TorchScript = 0, top-1 agreement 8/8 vs PT (PASS gate 1e-3).
  - **mac VM CoreML smoke:** `Danaus plexippus` → 0.74 top-1, `Argynnis cybele` → 0.93 top-1. Inference 200–270 ms QEMU CPU (Neural Engine on real device much faster).
  - **Staging artifacts:** `/home/michael/Projects/LepsAI/models/staging/export-2026050823{4625,4922}Z/` (mlpackage 80 MB, ONNX 158 MB, TorchScript 159 MB, sidecars).
  - **iOS handoff prompt:** `~/Projects/LepsAI/docs/claude/prompts/global-butterflies-512-integration.md` covers bundling, Arbutus upload (`s3://ami-models/mobile/`), preprocessing change in `CoreMLClassifier.swift` (224 → use `ModelInfo.inputSize`), validation gates.
- DEIMv2 vs D-FINE comparison — current `/home/debian/Projects/DEIM` clone is v1 (`Intellindust-AI-Lab/DEIM`, arxiv 2412.04234). DEIMv2 is a separate project page (`intellindust-ai-lab.github.io/projects/DEIMv2/`). After v1 D-FINE-S baseline is trained on FG, repeat with DEIMv2 backbone + config and compare on Leeds + locked FG test.

## Candidate stack — multi-arthropod detector (post-leps)

Once leps single-class is solid, generalize to all-arthropod. Candidate pipeline:

- **Per-tile detector**: DEIMv2-S or DEIMv2-M (DINOv3-distilled ViT-Tiny+ backbone). Class-agnostic or coarse-arthropod-class.
- **Tiling/merging layer**: SAHI. 640×640 tiles, 20–25% overlap.
- **Inference batching**: async batched 8–16 tiles to saturate the H100 MIG slice.
- **Post-merge**: NMS across tile boundaries → crop list.
- **Species classifier**: BioCLIP 2.5 on each merged crop.

Open questions to settle before training: detector class granularity (1-class vs ~10 coarse orders), training-data source for non-leps arthropods, how SAHI's overlap interacts with DEIMv2's NMS-free head.

## Constraints
- Don't touch FG prod or prod DB
- Wait for chroma-backend repo to do data pulls (Azure access lives there)
- DEIM training compute on `ami-workspace-02-gpu` H100 MIG only

## Key files
- `research/leps_localizer/scripts/train_yolov11s.py`, `train_rtdetr.py`, `train_yolo26s.py` — Ultralytics drivers
- `research/leps_localizer/scripts/eval_on_locked.py`, `eval_sahi.py` — Leeds eval drivers
- `research/leps_localizer/scripts/serve_gym.py` — gradio gym
- `src/localization/eval_locked.py`, `metrics.py`, `leps_data.py` — eval core
- `research/leps_localizer/METHODS_AND_RESULTS.md` — full writeup
- `research/leps_localizer/results.csv` — results table

## Data
- Train+val: `/mnt/butterflies-fg-2026-05/yolo/` (2,572 train / 290 val, single class)
- Leeds: `/mnt/butterflies-fg-2026-05/datasets/leeds-butterflies/images/` + `metadata/leeds-index.jsonl`
- Trained checkpoints: `/mnt/butterflies-fg-2026-05/runs/{yolov11s-fg-2026-05,yolov11s-fg-2026-05-r2,rtdetr-l-fg-2026-05,yolo26s-fg-2026-05}/weights/best.pt`
- Eval results: `/mnt/butterflies-fg-2026-05/eval/*/{metrics.json,report.md,per_sample.jsonl}`

## FG pool (Butterflies / Papilionoidea, ~74k photos)
- frac 0.01-0.05: 528
- frac 0.05-0.10: 1,403
- frac 0.10-0.25: 7,715
- frac 0.25-0.50: 22,714 (where to pull next)
- frac 0.50-0.95: 37,868
- frac ≥0.95: 4,716 (skip)
