# Lepidoptera Single-Subject Localizer

Train a fast butterfly detector for use on ~10M Fieldguide photos.

**Status:** design phase. See `DESIGN.md`.

**Branch:** `feat/leps-localizer-training`

## Quick start (for the next dev)

1. Read `DESIGN.md` end-to-end.
2. Read the references at the bottom of `DESIGN.md`. Especially the Arbutus 2026 GPU provisioning runbook.
3. Provision `ami-arbutus-train-01` per `DESIGN.md` § "Arbutus 2026 VM setup".
4. Add SSH config block to `~/Projects/AMI/ami-devops/ssh/arbutus2026_connections`.
5. Pull training data from FG via the `detector_dataset/` project — extend it with `--exclude-photo-ids-from <coco>` first. The 4 eval COCO files are at:
   ```
   ~/Projects/Fieldguide/chroma-backend/.claude/worktrees/detector-training/detector_dataset/datasets/
   ```
6. Sync data + eval set to the VM's persistent volume at `/mnt/data/leps_localizer/`.
7. Train YOLOv11s first (smallest scope, fastest feedback). Then RT-DETRv2. Then ami-ml's existing torchvision FRCNN.
8. After each run, evaluate against the 4 locked eval COCOs. Record results at `reports/<arch>-<date>.md`.

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
