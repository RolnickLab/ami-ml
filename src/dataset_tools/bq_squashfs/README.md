# BQ/SquashFS Pipeline

End-to-end pipeline for building a WebDataset training set from the BigQuery
`training_images` table on the **fir** cluster (Compute Canada / DRAC).

## Overview

```
[BigQuery: training_images]
         │
         ▼ Stage 1: download
   task_0.sqfs … task_9.sqfs       ← all downloaded images, split into 10 chunks
         │
         ▼ Stage 2: bq_export
   global_min25occ.csv             ← metadata for qualifying images (species, paths, taxon IDs)
         │
         ▼ Stage 3: split
   splits/train.csv
   splits/val.csv
   splits/test.csv
         │
         ▼ Stage 4: webdataset
   global_wds/{train,val,test}/*.tar + class_map.json
         │
         ▼ Stage 5: train
   models/global_wds/*.pt
```

---

## Data Source: BigQuery `training_images`

All images originate from the `leps-ai.global_butterflies_2604.training_images`
BigQuery table. Each row represents one image with:

- `photo_id` — iNaturalist photo ID (integer)
- `gbif_id` — GBIF occurrence ID (multiple images can share the same occurrence)
- `relative_local_path` — path of the image file within its SquashFS chunk
- `inat_taxon_id` — iNaturalist taxon ID (used as the class label)
- `fetch_status` — `'downloaded'` once the image has been fetched to disk

Only rows with `fetch_status = 'downloaded'` are used in the pipeline.

---

## SquashFS Chunks: task_0 … task_9

Images on disk are stored in 10 SquashFS archives (`task_0.sqfs` … `task_9.sqfs`),
located at `/project/rrg-bengioy-ad/melabbas/` and `/project/6068129/melabbas/`.

The task assignment is deterministic:

```
task_id = photo_id % 10
```

So `task_3.sqfs` contains all images whose `photo_id` ends in 3. The
`relative_local_path` column in BigQuery is the path of the image *within* its
sqfs archive. This split allows parallel downloading across 10 SLURM array tasks
and efficient batched processing during WebDataset creation.

---

## Stage 1 — Download

**SLURM job:** `scripts/job_bq_download.sh`
**Python:** `download_images.py`

Runs as a SLURM array job (10 tasks). Each task handles `photo_id % 10 == task_id`
and performs the full download-verify-pack loop:

1. Queries BQ for pending images assigned to this task, skipping any already
   recorded in `training_images_downloads` — fully resumable on resubmit.
2. Downloads images in parallel (32 workers) from `absolute_url`.
3. Verifies each image with PIL (width, height, corruption check).
4. Writes results back to the `training_images_downloads` BQ table
   (`fetch_status`: `downloaded`, `failed`, or `corrupted`).
5. Every 10,000 images, packs the staging dir into a `chunk_NNNN.sqfs` file
   using `mksquashfs`, then deletes the raw images to keep inode usage low.

After all 10 tasks complete, the per-chunk sqfs files in each staging dir are
merged into the final `task_N.sqfs` archives by the pack job.

```bash
sbatch scripts/job_bq_download.sh
```

**Output:** `task_0.sqfs` … `task_9.sqfs`

---

## Stage 2 — BQ Export

**SLURM job:** `scripts/job_bq_export.sh`
**Python:** `bq_export.py`

Runs a SQL query against BigQuery and streams the results to a CSV file on
Lustre. The query joins `training_images` with `inat_taxa` to attach species
names and taxonomy, and filters to images with `fetch_status = 'downloaded'`.

Two queries are available under `queries/`:

| Query file | Filter | Expected rows |
|---|---|---|
| `global_min25occ.sql` | Species with ≥ 25 distinct GBIF occurrences | ~10.6M images, ~4,704 species |
| `global_max2000img.sql` | Cap at 2,000 images per species | smaller subset |

```bash
# Default (global_min25occ.csv)
sbatch --dependency=afterok:<download_job_id> scripts/job_bq_export.sh

# Custom query
sbatch --export=QUERY_FILE=queries/global_max2000img.sql,OUTPUT=global_max2000img.csv \
       --dependency=afterok:<download_job_id> scripts/job_bq_export.sh
```

**Input:** BigQuery `training_images` + `inat_taxa` tables
**Output:** `data/global_min25occ.csv` (or custom filename)

---

## Stage 3 — Split

**SLURM job:** `scripts/job_bq_split.sh`
**Python:** `split.py`

Splits the BQ-exported CSV into `train.csv`, `val.csv`, and `test.csv` using
stratified sampling so every species is proportionally represented in all three
sets.

Key behaviours:
- **Split by occurrence** — images that share the same `gbif_id` (same field
  observation) are always kept in the same split, preventing data leakage.
- **Max instances** — caps images per species at 1,000 for train (proportional
  for val/test) to avoid class imbalance dominating training.
- **Min instances** — species with fewer than 5 training images are dropped.

```bash
sbatch --dependency=afterok:<export_job_id> scripts/job_bq_split.sh

# Custom input CSV
sbatch --export=CSV=global_max2000img.csv \
       --dependency=afterok:<export_job_id> scripts/job_bq_split.sh
```

**Input:** `data/<CSV>` (default: `global_min25occ.csv`)
**Output:** `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`

---

## Stage 4 — WebDataset

**SLURM job:** `scripts/job_bq_webdataset.sh`
**Python:** `create_webdataset.py` or `create_webdataset_generic.py`

Packs images from the SquashFS archives into WebDataset tar shards, organised
into train/val/test splits. Each shard contains ~1,000 images. Every image is
stored as a triplet inside the tar:

- `<md5key>.jpg` — image bytes
- `<md5key>.cls` — integer class ID (text)
- `<md5key>.json` — metadata (`class_id`, `relative_local_path`, `task_id`, species fields)

A `class_map.json` is also written to the output root, mapping sequential class
IDs to `inat_taxon_id` values.

```bash
sbatch --dependency=afterok:<split_job_id> scripts/job_bq_webdataset.sh
```

**Input:** `task_0.sqfs` … `task_9.sqfs`, `data/splits/{train,val,test}.csv`
**Output:** `/scratch/melabbas/global_wds/{train,val,test}/*.tar`, `class_map.json`

### Two versions of `create_webdataset`

| Script | When to use |
|---|---|
| `create_webdataset.py` | On **fir** — copies each sqfs to NVMe (`$SLURM_TMPDIR`), mounts with `squashfuse`, scatters images to NVMe shard dirs, packs to Lustre tars. Two-batch strategy keeps NVMe usage under 7 TB. |
| `create_webdataset_generic.py` | On **any machine** where sqfs are already mounted. Takes a parent directory containing `task_0/` … `task_9/` subdirectories. Reads images directly from the mounted dirs and streams them into tar shards — no NVMe copy, no squashfuse calls. |

The fir-specific version (`create_webdataset.py`) exists because the SquashFS
files are ~1 TB each and reading them directly from Lustre during scatter is
too slow — copying to NVMe first gives ~10× better I/O throughput. On machines
where the sqfs are already mounted (or images are on a fast local filesystem),
the generic version is simpler and produces identical output.

To use the generic version, mount each sqfs into a parent directory first:

```bash
mkdir -p /mnt/images/task_{0..9}
for T in $(seq 0 9); do
    squashfuse /path/to/task_${T}.sqfs /mnt/images/task_${T}
done

python create_webdataset_generic.py \
    --images-dir   /mnt/images \
    --split-csvs   train:splits/train.csv val:splits/val.csv test:splits/test.csv \
    --output-dir   /output/global_wds
```

---

## Stage 5 — Train

**SLURM job:** `scripts/job_bq_train.sh`

Trains a ResNet-50 classifier on the WebDataset shards using the
`ami-classification train-model` CLI. Shard counts and number of classes are
resolved dynamically from the output directory. Automatically resumes from the
latest checkpoint if one exists in the models directory.

```bash
sbatch --dependency=afterok:<webdataset_job_id> scripts/job_bq_train.sh
```

**Input:** `/scratch/melabbas/global_wds/{train,val,test}/*.tar`, `class_map.json`
**Output:** `/project/6068129/melabbas/ami-ml/models/global_wds/*.pt`

---

## Chaining the Full Pipeline

```bash
DOWNLOAD_JOB=$(sbatch --parsable scripts/job_bq_download.sh)
EXPORT_JOB=$(sbatch --parsable --dependency=afterok:$DOWNLOAD_JOB scripts/job_bq_export.sh)
SPLIT_JOB=$(sbatch --parsable --dependency=afterok:$EXPORT_JOB scripts/job_bq_split.sh)
WDS_JOB=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB scripts/job_bq_webdataset.sh)
sbatch --dependency=afterok:$WDS_JOB scripts/job_bq_train.sh
```

---

## File Reference

| File | Role |
|---|---|
| `download_images.py` | Stage 1 — fetch images from iNaturalist, verify with PIL, write results back to BQ (`training_images_downloads`), pack into per-chunk sqfs files |
| `bq_export.py` | Stage 2 — export BigQuery query results to CSV |
| `split.py` | Stage 3 — stratified train/val/test split with occurrence-level grouping |
| `create_webdataset.py` | Stage 4 — fir-specific, NVMe-optimised WebDataset packer |
| `create_webdataset_generic.py` | Stage 4 — generic WebDataset packer for pre-mounted image directories |
| `queries/global_min25occ.sql` | BQ query — species with ≥ 25 occurrences (~10.6M images) |
| `queries/global_max2000img.sql` | BQ query — capped at 2,000 images per species |
