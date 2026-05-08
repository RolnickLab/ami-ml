# NEXT SESSION — leps localizer

## Where we are (2026-05-08)

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
- CoreML export YOLO26 (#38)

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
