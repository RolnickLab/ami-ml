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

Source: Fieldguide production Postgres (`fieldguide.postgres.database.azure.com`, read-only) + Arbutus S3 (`object-arbutus.cloud.computecanada.ca`, bucket `fieldguide-production`).

Pull pipeline: extend `detector_dataset/` to support an `--exclude-photo-ids-from <coco>` flag pointing at the 4 eval COCOs.

**Scope:** Butterflies clade only (`552e76f291201b5ddbcbf77b`). Pool: 74,944 candidate photos with valid `crop_info`.

**Target size:** 20,000 images for first run. Buckets:

| Area-fraction bucket | Pool | Sample target | Why |
|---|---|---|---|
| 0.01–0.05 (tiny) | 528 | up to all (~500) | rarest, hardest, overweight |
| 0.05–0.10 (small) | 1,403 | ~1,400 | hardest at training scale |
| 0.10–0.25 | 7,715 | 4,000 | medium-hard |
| 0.25–0.50 | 22,714 | 6,000 | typical |
| 0.50–0.95 | 37,868 | 8,000 | easy, but reflects upload distribution |

Approximate stratification — exact counts decided at extract time. Excludes photos in any eval COCO. Uses the same `pi.sort_order = 0` filter and FG `photo_images.crop_info` JSON field for bbox.

**Train/val split:** 90/10 random within the 20k pull. Validation is held-out from training but **distinct from the locked eval set** — used only for early-stopping signal, not final reporting.

**Final reporting:** mAP@50, mAP@50-95, recall by area-frac bucket, computed on the 4 locked eval datasets.

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
| `src/localization/leps_data.py` | new: COCO-format adapter, area-frac stratified val split |
| `src/localization/eval_on_locked.py` | new: run a saved checkpoint against the 4 eval COCOs, log per-bucket recall and IoU + containment |
| `src/localization/yolo_train.py` | new: thin Ultralytics wrapper — calls `YOLO(...).train(data=...)` with our COCO + reporting hooks |
| `src/localization/rtdetr_train.py` | new: thin RT-DETR wrapper |

Code added under `src/localization/` follows existing ami-ml conventions: `src` package, click for CLI, wandb for tracking, env-var-driven paths.

## Arbutus 2026 VM setup

Cluster: `192.168.129.0/24`. OpenStack project `rpp-drolnick`, user `mihow`. Bastion: `ami-arbutus-bastion` (134.87.8.160), key `~/.ssh/ami2026.pem`.

**Provision a new training VM** (separate from the production ADC inference workers `ami-arbutus-gpu-01..04`). Suggested name `ami-arbutus-train-01`.

| Setting | Value | Notes |
|---|---|---|
| Flavor | `g1-24gb-c6-70gb-250` (H100 24GB vGPU, 6 vCPU, 70GB RAM, 20GB root + 250GB ephemeral `/mnt`) | Same as `ami-gpu-04`. If a 2× H100 flavor exists in the project's quota, prefer it for parallel architecture training. |
| Image | `ami-base-debian13-2026-04` | Same as gpu-03/04. NVIDIA driver `580.105.08` already in DKMS. |
| Disk | 250 GB ephemeral `/mnt`, plus an attached **persistent volume** for datasets and checkpoints | 20 GB root is tight but fine if Docker root + uv cache live on `/mnt`. Persistent volume so data survives shelving. |
| Network | default + internal SGs. No floating IP needed (jump via bastion). | |
| Keypair | `ami2026` | |
| User | `debian` | |

**Provisioning runbook:** follow `~/Projects/AMI/ami-devops/docs/claude/sessions/2026-04-20-ami-gpu-04-provisioning.md`, replacing the ADC-worker supervisor configs with a training environment (uv + ami-ml + datasets). Reboot once after driver install before declaring `nvidia-smi` healthy.

**Add SSH config** to `~/Projects/AMI/ami-devops/ssh/arbutus2026_connections`:
```
Host ami-arbutus-train-01
  Hostname 192.168.129.<TBD>
  User debian
  ProxyJump ami-arbutus-bastion
  IdentityFile ~/.ssh/ami2026.pem
  IdentitiesOnly yes
  ForwardAgent yes
```

## Data flow to the VM

The VM is in the **same Arbutus region** as `fieldguide-production` Arbutus S3 — fast intra-region transfer.

Two options, decide at implementation time:

1. **Bulk copy to local volume.** `s5cmd` or `rclone copy s3:fieldguide-production/<keys>` from a manifest file → `/mnt/data/leps_localizer/images/`. Predictable training I/O, good for repeated runs. ~50–100 GB for 20k images.
2. **Stream from S3 via webdataset.** ami-ml's existing convention. No local copy. Best for the first exploratory training run.

Manifests (image keys + bboxes) come from the `detector_dataset/` extract pipeline, packaged as a single tarball or COCO JSON synced to the VM via `scp`.

**Credentials:** Arbutus S3 keys live in `~/Projects/Fieldguide/prism/credentials-ami-cc-prism.json` `fieldguide` entry. Copy to the VM's `~/.aws/credentials` or as `AWS_*` env vars. Do **not** commit credentials.

**Eval datasets** (the locked 4) sync to `/mnt/data/leps_localizer/eval/` once at provision time. Always re-evaluated from there.

## Training loop (YOLOv11s, first model)

```
data:
  path: /mnt/data/leps_localizer/dataset
  train: images/train.txt
  val: images/val.txt
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

Checkpoints to `/mnt/data/leps_localizer/runs/yolo11s_butterflies_v0/`. Best-by-val-mAP50 mirrors to wandb artefact and to `/media/michael/ZWEIBEL/MODELS/leps_localizer/` on `beast` (rsync at end of run).

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

1. **GPU flavor**: confirm `g1-24gb-c6-70gb-250` (H100 24GB) is available under the project quota. If a 2× GPU flavor is available, prefer it (parallel training across architectures).
2. **Persistent volume size**: 200–500 GB. Depends on whether you bulk-copy images or stream from S3.
3. **Square-target augmentation**: implement variant or skip for first run? Recommendation: skip first run, evaluate, decide based on per-dataset IoU vs containment gap.
4. **YOLO version**: YOLOv11s vs YOLOv8s. YOLOv11 is newer (2026) and Ultralytics-recommended. Default to v11 unless stability issues.
5. **Image size**: 640 default vs 1024 for small-bbox recall. Run both as separate runs if budget allows.

## Definition of done (this stage)

- [ ] 20k butterfly training set extracted, eval `photo_id`s excluded, on Arbutus VM
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
- Arbutus 2026 GPU provisioning: `~/Projects/AMI/ami-devops/docs/claude/sessions/2026-04-20-ami-gpu-04-provisioning.md`
- Arbutus 2026 SSH config: `~/Projects/AMI/ami-devops/ssh/arbutus2026_connections`
- Arbutus infrastructure overview: `~/Projects/AMI/ami-devops/docs/claude/INFRASTRUCTURE.md`
