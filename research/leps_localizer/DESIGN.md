# Lepidoptera Single-Subject Localizer — Design

**Date:** 2026-05-05
**Branch:** `feat/leps-localizer-training`
**Status:** design draft (not yet implemented)
**Owner:** michael@mixedneeds.com (designer); handoff to another dev for Arbutus 2026 training

## Goal

Train a fast, high-accuracy single-class object detector for **butterflies** (Papilionoidea) that can run inference on ~10M Fieldguide photos with high recall, especially for small bounding boxes (subject occupies <10% of frame).

Eventual goals (out of scope for this stage):
1. Extend to all Lepidoptera, then all Arthropoda
2. Multi-object localization
3. iOS / on-device deployment

## Why this stage exists

We need an automated way to crop the primary subject from arbitrary user-uploaded photos. Off-the-shelf foundation detectors (Grounded-SAM, OWLv2, DINOv3+PCA) are candidates but unverified on FG distribution. A fine-tuned single-class detector is faster than open-vocab detectors and easier to deploy at scale.

Skipping the off-the-shelf eval stage (deferred spec at `~/Projects/Fieldguide/chroma-backend/.claude/worktrees/detector-training/docs/superpowers/specs/2026-04-27-arthropod-localizer-eval-design.md`) — go straight to training a butterfly model. The eval datasets become validation tooling during training.

## Held-out test set (locked, never train on)

Built in the sibling `detector_dataset/` project (Fieldguide chroma-backend worktree). Total ~3,800 images:

| Dataset | Count | Source | Notes |
|---|---|---|---|
| `leps-butterflies-500` | 999 | FG random butterflies | square gt (FG crops are square), full area-frac range |
| `leps-butterflies-small-1000` | 999 | FG, frac < 0.10 | hardest small-bbox cases, 446 species, 324 photographers |
| `leps-butterflies-medsmall-1000` | 997 | FG, frac 0.10–0.25 | 644 species, 337 photographers |
| `leeds-butterflies` | 832 | Leeds v1.0 (Zenodo) | tight rect from segmentation masks; reference dataset |

All four are COCO-format with single class `arthropod`. They live at `~/Projects/Fieldguide/chroma-backend/.claude/worktrees/detector-training/detector_dataset/datasets/`. Training data MUST exclude any `photo_id` present in these COCOs.

## Training data strategy

**Source:** Fieldguide production Postgres (`fieldguide.postgres.database.azure.com`, read-only) for metadata + bbox; FG image objects already live on the old-cloud `object-arbutus.cloud.computecanada.ca` bucket `fieldguide-production`.

**Destination (training-side store):** `s3://ami-trainingdata/ai-for-leps/localization/<dataset-name>/` on the **new** Arbutus 2026 endpoint `https://object-arbutus.alliancecan.ca`. The `ai-for-leps/` prefix mirrors the old-cloud Swift layout that will be rsynced into the new cloud (open follow-up #26 in `2026-04-28-object-store-fuse-mount-setup.md`); the `localization/` sub-prefix is new and reserved for detector training datasets (sibling to the existing `datasets/` prefix used for classification training). Training jobs read from the new endpoint.

For our first dataset, name it `butterflies-fg-2026-05`. Full path: `s3://ami-trainingdata/ai-for-leps/localization/butterflies-fg-2026-05/`.

**Layout in the bucket:**
```
s3://ami-trainingdata/ai-for-leps/localization/butterflies-fg-2026-05/
  index.parquet                  # one row per image, all metadata
  images/<sha256-prefix>/<sha256>.jpg
  manifest_train.txt             # newline-delimited image keys
  manifest_val.txt
  README.md                      # provenance, extract date, schema
```

**`index.parquet` schema** (one row per image; this replaces dataset-side stratification):

| Column | Type | Notes |
|---|---|---|
| `image_key` | str | S3 key relative to bucket root |
| `sha256` | str | content hash, also used as filename |
| `photo_id` | str | FG `photos.mongo_id` |
| `category_id` | str | FG `photos.category_mongo_id` (species-level) |
| `parents` | list[str] | FG `categories.parents` array (full taxonomy chain) |
| `user_id` | str | FG `photos.user_mongo_id` (photographer; for split-by-user if needed) |
| `width` | int | full image px |
| `height` | int | full image px |
| `bbox_xyxy` | list[float] | gt bbox in pixels |
| `bbox_area_fraction` | float | bbox area / image area (precomputed; train-time filter / weight) |
| `bbox_is_square` | bool | `abs(w-h) < 2 px` (FG square-crop signal) |
| `created_at` | str | ISO timestamp from FG (cutoff filter) |
| `split` | str | `"train"` or `"val"` (90/10 random, fixed seed) |

Training jobs query the parquet at runtime to: filter eval-set `photo_id`s, choose subsets (e.g. only small bboxes for a hardness-focused run), apply per-row weights, etc. **No extract-time bucket stratification.**

**Scope:** Butterflies clade only (`552e76f291201b5ddbcbf77b`). Pool: 74,944 candidate photos with valid `crop_info`.

**Target size:** 20,000 images for first run, randomly drawn from the pool with eval-set `photo_id`s excluded. Distribution of `bbox_area_fraction` follows the natural pool — overweight by bucket happens later via parquet filters if needed.

**Pull pipeline:** extend `detector_dataset/` (in chroma-backend worktree) with:
1. `--exclude-photo-ids-from <coco>` flag pointing at the 4 eval COCOs
2. New stage `assemble-trainset`: writes `index.parquet` + uploads images directly to `s3://ami-trainingdata/ai-for-leps/localization/<name>/` on the new arbutus endpoint, instead of writing local COCO

**Train/val split:** 90/10 random, fixed seed, recorded in `index.parquet.split`. Distinct from the locked eval set; used only for early-stopping during training.

**Final reporting:** mAP@50, mAP@50-95, recall stratified by `bbox_area_fraction` bucket on the 4 locked eval datasets.

## Model architecture

Three candidates, run sequentially, compared against the same locked eval set:

1. **Ultralytics YOLOv11s (or YOLOv8s)** — first model. Fastest training and inference. AGPL-3.0 (matches ami-ml's recent license switch). Trivial COCO ingest. Single-GPU 1×H100/24GB sufficient. Expected: 1–3 days end-to-end.
2. **RT-DETRv2 (small)** — second model. Transformer detector, no NMS, often stronger small-object recall. Slightly heavier training but comparable inference speed.
3. **ami-ml `src/localization/` torchvision (Faster R-CNN ResNet50 FPN v2 or RetinaNet)** — third model. Reuses existing ami-ml code largely unmodified. Establishes a baseline using the project's incumbent stack.

All trained as single-class `arthropod`.

**Inference target:** server-side batch on Arbutus 2026 GPU VMs initially. iOS deployment is a separate later stage and may favour YOLO-nano or a distilled student.

## Code layout

Branch: `feat/leps-localizer-training` on `RolnickLab/ami-ml`.

| Path | Purpose |
|---|---|
| `research/leps_localizer/DESIGN.md` | this document |
| `research/leps_localizer/README.md` | quick-start for the next dev |
| `research/leps_localizer/configs/` | YOLO `.yaml` data configs, RT-DETR configs, FRCNN args |
| `research/leps_localizer/scripts/` | shell wrappers for VM jobs (data pull, train, eval) |
| `research/leps_localizer/notebooks/` | data exploration, results visualisation |
| `src/localization/training.py` | existing torchvision trainer — reuse as-is for variant 3 |
| `src/localization/leps_data.py` | new: parquet-index adapter, fixed train/val split read from `index.parquet` |
| `src/localization/eval_on_locked.py` | new: run a saved checkpoint against the 4 eval COCOs, log per-bucket recall and IoU + containment |
| `src/localization/yolo_train.py` | new: thin Ultralytics wrapper — calls `YOLO(...).train(data=...)` with our COCO + reporting hooks |
| `src/localization/rtdetr_train.py` | new: thin RT-DETR wrapper |

Code added under `src/localization/` follows existing ami-ml conventions: `src` package, click for CLI, wandb for tracking, env-var-driven paths.

## Arbutus 2026 VM: `ami-workspace-02-gpu`

Cluster: `192.168.129.0/24`. OpenStack project `rpp-drolnick`, user `mihow`. Bastion: `ami-arbutus-bastion` (134.87.8.160), key `~/.ssh/ami2026.pem`.

**Use the existing `ami-workspace-02-gpu` box.** No provisioning needed.

| Setting | Value |
|---|---|
| SSH alias | `ami-workspace-02-gpu` (also `ami-arbutus-workspace-02-gpu`) |
| IP | `192.168.129.83` (private; jump via `ami-arbutus-bastion`) |
| Flavor | `g1-24gb-c6-70gb-250` (H100 24GB vGPU, 6 vCPU, 70GB RAM) |
| Disk | 246 GB ephemeral `/mnt` |
| User | `debian` |
| Keypair | `ami2026` |

SSH config block already in `~/Projects/AMI/ami-devops/ssh/arbutus2026_connections`.

### Already installed and configured

- NVIDIA driver, `nvidia-smi` working (H100 24GB vGPU)
- `awscli`, `rclone`, `squashfuse`, `fuse3`, `mount-s3`
- `~debian/.aws/{credentials,config}` with `[ami]` profile pointing at `https://object-arbutus.cloud.computecanada.ca`
- `~debian/ami-devops` cloned
- systemd units for FUSE-mounting `ami-trainingdata` S3 bucket and squashfs shards (`ami-trainingdata-s3.service`, `ami-squashfs@N.service`)

### Existing data on the box

`s3://ami-trainingdata/ai-for-leps/datasets/global_butterflies_2604/sqfs/` is FUSE-mounted as 10 squashfs shards at `/mnt/squash-0..9` (each ~1.18 TB JPEGs, total ~11.8 TB). Currently only `@0` enabled by default — `sudo systemctl enable --now ami-squashfs@N.service` to mount more.

This is a separate dataset from the FG butterflies pull described above. **Decision for first run:** ignore `global_butterflies_2604` and train on the FG-derived 20k. Revisit using the larger squashfs as additional training data after the first model lands and we have a baseline. Provenance, taxonomy, and bbox availability of `global_butterflies_2604` need to be confirmed before relying on it.

### What still needs setup on the box

1. Install `uv` if not present (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
2. Clone `ami-ml` at `~/ami-ml` on the `feat/leps-localizer-training` branch
3. `uv sync --extra dev --extra research` inside the repo
4. FUSE-mount `s3://ami-trainingdata/` at `/mnt/s3-trainingdata/` (analogous to the existing `/mnt/s3` for old-cloud). Training data and eval set both live under it.
5. `wandb login` with the kalpa/RolnickLab account
6. Add Ultralytics + RT-DETR deps as new optional extras under `pyproject.toml` (e.g. `--extra detection`)

### Anomaly to be aware of

`ssh -A ami-workspace-02-gpu 'ssh git@github.com'` authenticates as `adityajain07`, not the local user (per `2026-04-28-object-store-fuse-mount-setup.md` § Anomaly). Forwarded agent key is registered to Aditya's GitHub account. Repo writes from this box will appear under his name unless `GIT_AUTHOR_*` / `GIT_COMMITTER_*` are set explicitly.

## Data flow

```
FG Postgres + old-cloud S3 (fieldguide-production)
        │  (extract via detector_dataset/, on a laptop or beast)
        ▼
new-cloud s3://ami-trainingdata/ai-for-leps/localization/butterflies-fg-2026-05/
        │  (training reads here)
        ▼
ami-workspace-02-gpu  →  cache misses fault to S3, hot data lands in /mnt cache
```

Two read modes for the trainer, decide at implementation time:

1. **Stream from S3** via `webdataset` (ami-ml's existing convention) or `mountpoint-s3` FUSE-mount of `ami-trainingdata` at `/mnt/s3-trainingdata/`, similar to the existing setup at `/mnt/s3` for the old cloud. No bulk copy. Lowest disk pressure on the 246 GB ephemeral.
2. **Bulk copy to `/mnt`** via `s5cmd` or `rclone` for the active dataset shard. Predictable I/O, good for repeated runs. ~50–100 GB for 20k JPEGs.

Recommendation: start with FUSE-mount streaming. Switch to bulk copy if dataloader I/O caps GPU utilization.

**Credentials.** Two endpoints involved:
- New cloud (`object-arbutus.alliancecan.ca`, `ami-trainingdata`) — already configured under `~debian/.aws/[ami]` profile on the box (per `2026-04-28-object-store-fuse-mount-setup.md`). Confirm at session start; rotate if leaked.
- Old cloud (`object-arbutus.cloud.computecanada.ca`, `fieldguide-production`) — only needed by the **extract** step on a laptop/beast, never on the training VM. Live at `~/Projects/Fieldguide/prism/credentials-ami-cc-prism.json` `fieldguide` entry. Do **not** commit credentials.

**Eval datasets** (the locked 4): sync once to `s3://ami-trainingdata/ai-for-leps/localization/eval-locked/` so they live alongside the training data. The training VM reads them via the same FUSE mount.

## Training loop (YOLOv11s, first model)

```
data:
  path: /mnt/s3-trainingdata/ai-for-leps/localization/butterflies-fg-2026-05
  train: manifest_train.txt
  val:   manifest_val.txt
  names: ['arthropod']

model: yolo11s.pt   # COCO-pretrained
imgsz: 640          # also try 1024 for small-bbox recall
batch: 32
epochs: 100
patience: 20        # early stop on val mAP@50
optimizer: AdamW
lr0: 0.001
cos_lr: true
warmup_epochs: 3
augment: mosaic + hsv + flip; mixup off (single-subject)
device: 0
workers: 6
project: leps_localizer
name: yolo11s_butterflies_v0
```

Checkpoints to `/mnt/runs/leps_localizer/yolo11s_butterflies_v0/` (ephemeral). Best-by-val-mAP50 mirrors to:
- wandb artefact
- `s3://ami-trainingdata/ai-for-leps/localization/runs/yolo11s_butterflies_v0/` (persistent, survives box reshelving)
- `/media/michael/ZWEIBEL/MODELS/leps_localizer/` on `beast` (local mirror via rsync at end of run)

After training: run `eval_on_locked.py` against the 4 locked eval datasets, log:
- mAP@50, mAP@50-95 overall
- Recall@IoU≥0.5 per area-frac bucket (4 buckets)
- IoU + containment per dataset (containment = pred ∩ gt / area(gt), critical for the FG square-gt vs YOLO-tight-rect mismatch)
- Best-of-N reduction: any predicted box near gt counts as a hit (single-subject task)

Same protocol for RT-DETRv2 and Faster R-CNN runs.

## Eval considerations: square vs tight bboxes

FG ground-truth bboxes come from user-set crop info — almost all are **square** (the `crop_info` is the crop the user chose). YOLO/RT-DETR/FRCNN all predict tight rectangles. Pure IoU underrates predictions that correctly localise the subject inside a square gt.

Mitigations:
1. Report **both IoU and containment** (already in eval set design).
2. During training: optionally augment YOLO targets to be the smallest enclosing **square** of the subject. Decide after first run — may help the model match FG's user-curation prior, but may also hurt recall on Leeds (tight gt) and on real downstream cropping.
3. First run: train on YOLO-tight targets derived from the FG square gt as-is. The square gt is a strict over-bound on the subject — model still learns useful localisation.

## Reporting

Per architecture, produce:
1. wandb run with all hyperparams, train/val curves
2. A markdown report at `research/leps_localizer/reports/<arch>-<date>.md` summarising the eval-set numbers
3. Saved checkpoint + ONNX export under `/media/michael/ZWEIBEL/MODELS/leps_localizer/<arch>-<date>/`

## Open questions for the next dev

1. **`global_butterflies_2604` dataset**: confirm provenance (source, license), bbox availability, taxonomy mapping. If it has bboxes that aren't square-cropped FG-style and the species distribution overlaps butterflies, this is a 12 TB pretraining or co-training opportunity.
2. **Disk strategy**: 246 GB ephemeral `/mnt` does not survive shelving. Authoritative dataset and run artefacts live on `s3://ami-trainingdata/ai-for-leps/localization/`; treat the VM disk as a cache. If FUSE-streaming caps GPU utilization, attach a persistent volume or warm `/mnt` from S3 at job start.
3. **Square-target augmentation**: implement variant or skip for first run? Recommendation: skip first run, evaluate, decide based on per-dataset IoU vs containment gap.
4. **YOLO version**: YOLOv11s vs YOLOv8s. YOLOv11 is newer (2026) and Ultralytics-recommended. Default to v11 unless stability issues.
5. **Image size**: 640 default vs 1024 for small-bbox recall. Run both as separate runs if budget allows.
6. **Git author identity on the box**: agent-forwarded SSH writes commits as `adityajain07`. Set `GIT_AUTHOR_*` / `GIT_COMMITTER_*` per session, or push from a laptop only.

## Definition of done (this stage)

- [ ] 20k butterfly training set extracted, eval `photo_id`s excluded, uploaded to `s3://ami-trainingdata/ai-for-leps/localization/butterflies-fg-2026-05/` with `index.parquet`
- [ ] YOLOv11s training run completed, eval-set report produced
- [ ] RT-DETRv2 training run completed, eval-set report produced
- [ ] Faster R-CNN ResNet50 FPN v2 training run completed, eval-set report produced
- [ ] Side-by-side comparison report at `research/leps_localizer/reports/comparison-2026-MM-DD.md`
- [ ] Best model exported to ONNX and saved under `/media/michael/ZWEIBEL/MODELS/leps_localizer/`
- [ ] Inference benchmark on `beast` 2× RTX 3090 for the best model: img/s, GPU util, peak VRAM

## Out of scope

- Fine-tuning multi-class arthropod detector
- Multi-object detection
- iOS / CoreML export
- Auto-labelling pipeline using the trained model
- Inference-side production wiring (FG batch crop service)

These come after this stage's best model demonstrates ≥0.85 mAP@50 + ≥0.70 mAP@50-95 on the locked eval set.

## References

- Existing ami-ml localizer: `src/localization/training.py`, `src/localization/utils.py`, `src/localization/data/custom_datasets.py`
- Detector dataset project: `~/Projects/Fieldguide/chroma-backend/.claude/worktrees/detector-training/detector_dataset/`
- Locked eval spec (deferred): `~/Projects/Fieldguide/chroma-backend/.claude/worktrees/detector-training/docs/superpowers/specs/2026-04-27-arthropod-localizer-eval-design.md`
- Eval dataset spec: `~/Projects/Fieldguide/chroma-backend/.claude/worktrees/detector-training/docs/superpowers/specs/2026-04-24-arthropod-detector-eval-dataset-design.md`
- Workspace VM setup: `~/Projects/AMI/ami-devops/docs/claude/sessions/2026-04-28-object-store-fuse-mount-setup.md`
- Arbutus 2026 GPU provisioning runbook: `~/Projects/AMI/ami-devops/docs/claude/sessions/2026-04-20-ami-gpu-04-provisioning.md`
- Arbutus 2026 SSH config: `~/Projects/AMI/ami-devops/ssh/arbutus2026_connections`
- Arbutus infrastructure overview: `~/Projects/AMI/ami-devops/docs/claude/INFRASTRUCTURE.md`
