# Lepidoptera Single-Subject Localizer

Train a fast butterfly detector for use on ~10M Fieldguide photos.

**Status:** design phase. See `DESIGN.md`.

**Branch:** `feat/leps-localizer-training`

## Quick start (for the next dev)

1. Read `DESIGN.md` end-to-end.
2. Read the references at the bottom of `DESIGN.md`. Especially `2026-04-28-object-store-fuse-mount-setup.md` (the workspace VM is already set up).
3. SSH to the existing training box: `ssh ami-workspace-02-gpu`. H100 24GB, FUSE-mounted butterfly squashfs already in place. No provisioning needed.
4. Pull training data from FG via the `detector_dataset/` project — extend it with `--exclude-photo-ids-from <coco>` first. The 4 eval COCO files are at:
   ```
   ~/Projects/Fieldguide/chroma-backend/.claude/worktrees/detector-training/detector_dataset/datasets/
   ```
5. Upload extracted dataset to `s3://ami-trainingdata/ai-for-leps/localization/butterflies-fg-2026-05/` (new arbutus 2026 endpoint `object-arbutus.alliancecan.ca`). FUSE-mount it on the VM at `/mnt/s3-trainingdata/`.
6. Train YOLOv11s first (smallest scope, fastest feedback). Then RT-DETRv2. Then ami-ml's existing torchvision FRCNN.
7. After each run, evaluate against the 4 locked eval COCOs. Record results at `reports/<arch>-<date>.md`.

## Layout

```
research/leps_localizer/
├── DESIGN.md          # full design spec
├── README.md          # this file
├── configs/           # YOLO data .yaml, RT-DETR config, FRCNN args (TBD)
├── scripts/           # shell wrappers for VM jobs (TBD)
├── notebooks/         # data exploration, results viz (TBD)
└── reports/           # one .md per training run + a comparison report (TBD)
```

Code lives under `src/localization/` (existing module) — see `DESIGN.md` § "Code layout".

## Don't train on the eval set

Anything in these 4 COCOs is **forever held out**:
- `leps-butterflies-500` (999 random)
- `leps-butterflies-small-1000` (frac < 0.10)
- `leps-butterflies-medsmall-1000` (frac 0.10–0.25)
- `leeds-butterflies` (832 reference)

The training data extraction script must take an `--exclude-photo-ids-from <coco>` argument.
