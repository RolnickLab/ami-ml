# Geo-prior pipeline

Trains and evaluates a **geographic prior** for global butterflies: a model of
`p(species | latitude, longitude, date)` that re-ranks / gates the image
classifier's predictions. The network is a SINR-style **FCNet** (no images —
only coordinates, dates, and species labels).

This package is self-contained: BigQuery → training data → trained model, with
all configuration in `.env` and the model code kept in-tree.

---

## Directory layout

```
src/geoprior/
├── README.md                       ← you are here
├── config.py                       ← central config (reads .env); nothing hardcoded
├── .env.sample                     ← copy to .env and fill in (real .env is gitignored)
├── requirements.txt                ← all dependencies
├── geoprior_categ_map.json         ← FROZEN species→class_id artifact (12,317 classes)
├── geoprior_categ_map.PROVENANCE.md← where the frozen map comes from
├── build_geoprior_categ_map.py     ← Stage 0: BQ → category map (regenerate/verify)
├── build_geoprior_json.py          ← Stage 1: BQ + splits + map → train/val/test.json
├── train_geoprior.py               ← Stage 2: JSON → trained FCNet (+ wandb)
├── predict_geoprior.py             ← Stage 3: per-occurrence priors for fusion
├── fusion_eval_top5.py             ← Stage 4: geo-prior + classifier top-5 eval
└── geoprior_fagner/                ← FCNet network, in-tree (frozen @ upstream commit)
    ├── README.md, models.py, losses.py, dataloader.py
```

All scripts are run **as modules from the repo root** so package imports
(`from src.geoprior…`) resolve:

```bash
python -m src.geoprior.<script_name> [flags]
```

---

## Setup

```bash
pip install -r src/geoprior/requirements.txt      # see notes for the CUDA torch wheel
cp src/geoprior/.env.sample src/geoprior/.env      # then edit .env
```

### Configuration (`.env`)

Every path / identifier / secret lives in `src/geoprior/.env` (loaded by
`config.py`). See `.env.sample` for the annotated list. Summary:

| Variable | Purpose | Default |
|---|---|---|
| `GEOPRIOR_BQ_PROJECT` / `GEOPRIOR_BQ_DATASET` | BigQuery source | `leps-ai` / `global_butterflies_2604` |
| `GOOGLE_APPLICATION_CREDENTIALS` | BQ auth (service-account key) | unset → uses gcloud ADC |
| `GEOPRIOR_CATEG_MAP` | frozen category map | in-repo `geoprior_categ_map.json` |
| `GEOPRIOR_DATA_DIR` | generated train/val/test.json | `<repo>/data/geoprior` |
| `GEOPRIOR_SPLITS_DIR` | vision `val.csv`/`test.csv` (hold-out gbif_ids) | `<repo>/data/splits` |
| `GEOPRIOR_MODEL_DIR` | checkpoint output | `<repo>/models/geoprior` |
| `WANDB_ENTITY` / `WANDB_PROJECT` | W&B logging | `moth-ai` / `Global-Butterfly` |
| `WANDB_API_KEY` | **secret** W&B key | — (or use `--wandb_offline`) |

**Secrets** (`WANDB_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`) are never
hardcoded and never committed — `.env` is gitignored.

---

## Pipeline

### Stage 0 — Category map (frozen)
The `species → class_id` map is the **frozen class-space contract**: the trained
model's output indices are bound to its alphabetical ordering. It is committed
(`geoprior_categ_map.json`) and only regenerated/verified, never silently
overwritten. See `geoprior_categ_map.PROVENANCE.md`.

```bash
# verify the committed map still matches BigQuery (no writes)
python -m src.geoprior.build_geoprior_categ_map
# (re)materialise all artifacts
python -m src.geoprior.build_geoprior_categ_map --write --out-dir "$GEOPRIOR_DATA_DIR"
```

### Stage 1 — Build training JSON
Pulls geocoded occurrences from BigQuery, excludes the vision `val`/`test`
gbif_ids from train (prevents leakage), maps species via the frozen map, and
writes COCO-style `train/val/test.json` to `GEOPRIOR_DATA_DIR`.

```bash
python -m src.geoprior.build_geoprior_json
```

### Stage 2 — Train
```bash
python -m src.geoprior.train_geoprior \
    --train_data_json "$GEOPRIOR_DATA_DIR/train.json" \
    --model_save_path "$GEOPRIOR_MODEL_DIR" \
    --epochs 30 --batch_size 1024 --embed_dim 256 \
    --max_instances_per_class 100        # add --wandb_offline to skip W&B
```
Saves a checkpoint after every epoch plus `model_final_*.pth`.

### Stage 3 — Predict (for fusion)
```bash
python -m src.geoprior.predict_geoprior \
    --test_data_json "$GEOPRIOR_DATA_DIR/val.json" \
    --model_path "$GEOPRIOR_MODEL_DIR/model_final_*.pth" \
    --results_dir "$GEOPRIOR_DATA_DIR/preds/val"
```

### Stage 4 — Fusion eval (top-5)
```bash
python -m src.geoprior.fusion_eval_top5      # paths come from .env
```

---

## Inputs you must provide

- **BigQuery access** to `GEOPRIOR_BQ_PROJECT.GEOPRIOR_BQ_DATASET`
  (`gbif_inat_occurrences`, `gbif_occurrence_location`), via ADC or a
  service-account key.
- **Vision split CSVs** `val.csv` / `test.csv` in `GEOPRIOR_SPLITS_DIR`
  (produced by the vision pipeline's `src/dataset_tools/bq_squashfs/split.py`).
  Only their `gbif_id` column is used.
- A **GPU** is optional — the model is ~3.7 M params; CPU works.

The category map is already provided (committed, frozen).

---

## The model

- **Network:** `FCNet` (4× residual blocks over a coordinate/date encoding),
  kept in-tree in `geoprior_fagner/`, from Fagner Cunha's lepsAI (Apache-2.0).
- **Class space:** 12,317 species (every species with ≥1 geocoded occurrence in
  the `public_gbif_2026-05` snapshot).
- **Inputs:** 6 features (cos/sin of lat, lon, day-of-year).
- **Reference run** `geoprior-fcnet-global-12317cls-v1`: 30 epochs, batch 1024,
  lr 5e-4 (decay 0.98/epoch), embed_dim 256, BalancedSampler cap 100/class;
  final loss ≈ 0.15; checkpoint ≈ 14.7 MB.

## Reproducibility notes

- The frozen category map + the in-tree `geoprior_fagner` (pinned at an upstream
  commit) together fix the model's architecture/class contract. Changing either
  means retraining.
- The build is deterministic given the same BigQuery snapshot, splits, and map.
