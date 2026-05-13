# Session prompt — bulk localize 10.7M `training_images` → BQ

## Goal

Run **YOLO26-s v2** (current best on the real ranking signal — small/mid-ratio
recall, METHODS_AND_RESULTS §5.3) over every row in
`leps-ai.global_butterflies_2604.training_images` (~10.7M iNat / eButterfly /
museum photos), write per-image inference rows to the planned
`localizer_eval_results` table, and MERGE the winning bbox into
`training_images.primary_subject_bbox`. **DEIMv2-S** is a follow-up second
run; downstream MERGE then switches from "latest run wins" to "best score
across models."

The pipeline (worklist → parquet → rsync → batch_localize → NDJSON → bq load
→ MERGE) is fully designed already in
`.claude/skills/bigquery-leps/procedures/`. **This session's job is to fill
in the 4 missing implementation pieces (BQ table, BQ column patch,
`batch_localize.py`, schema JSON), validate on a 1K canary, then kick off the
full run.**

## State summary

| Component | Status | Where |
|---|---|---|
| Pipeline design (worklist → infer → load → MERGE) | done | `.claude/skills/bigquery-leps/procedures/batch-inference-worklist.md` |
| Cost shape (≤$1/run for 10.7M) | done | `.claude/skills/bigquery-leps/SKILL.md` §"Cost shape" |
| Resume / idempotency (shard NDJSON + `bbox_inference_run_id` filter) | done | `procedures/batch-inference-worklist.md` §"Resume / idempotency" |
| Two-host pattern (BEAST=bq, workspace=GPU+`/mnt/squash-0`) | done | `SKILL.md` §"Two-host workflow" |
| Path translation BQ → on-disk (`/mnt/squash-0/{bucket:03d}/<uuid>.jpg`) | done | `.claude/notes/arbutus-storage.md` §"Path translation" |
| **`localizer_eval_results` BQ table** | **missing** | this session — Step A |
| **`training_images.primary_subject_bbox` + 3 sibling cols** | **missing** | this session — Step B |
| **`research/leps_localizer/scripts/batch_localize.py`** | **missing** | this session — Step C |
| **`schemas/localizer_eval_results_schema.json`** | **missing** | this session — Step A |
| **Canary run (LIMIT 1000)** | **not started** | this session — Step D |
| Choice of first model | done | YOLO26-s v2 (justified §"First model" below) |

## First model to run

**YOLO26-s v2.** Wins small/mid-ratio recall per METHODS_AND_RESULTS §5.3
(0.328 small / 0.629 mid vs DEIMv2-S's 0.281 / 0.573). iNat / eButterfly /
museum photos are not framing-bias-friendly the way Leeds is — most subjects
will land in small/mid bins, so small-ratio recall dominates deployment
value. DEIMv2-S wins large-bin recall + "finds something" rate, so it's a
strong **second** run for combined-best-score MERGE.

- **Weights:** `/mnt/butterflies-fg-2026-05/runs/yolo26s-v2/weights/best.pt`
  (note: verify exact path on the workspace VM — the run dir was auto-suffixed
  `yolo26s-fg-2026-05-v2-2` per NEXT_SESSION.md 2026-05-09; the symlink at
  `runs/yolo26s-v2/` may or may not exist. `ls /mnt/butterflies-fg-2026-05/runs/`
  on workspace before launching.)
- **Inference imgsz:** 1280 (training imgsz; downscaling at inference hurts
  small-bin recall, which is the whole point of picking this model).
- **`--score-thresh` for "primary subject":** 0.25 (matches gym + Leeds eval
  default).
- **Run-id:** `2026-05-12-yolo26s-v2-fullbq` (date = day the run actually
  launches; bump if it slips).
- **Model-id:** `yolo26s_v2`.

DEIMv2-S follow-up:
- Weights: `/mnt/butterflies-fg-2026-05/runs/deimv2-s-fg-2026-05/best_stg1.pth`
- Config: `/home/debian/Projects/DEIMv2/configs/deimv2_butterflies_s.yml`
- Venv: `/home/debian/Projects/DEIMv2/.venv/` (DEIMv2's own torch 2.5.1 /
  torchvision 0.20.1 pin — do NOT use ami-ml's venv for DEIMv2)
- Run-id: `2026-05-XX-deimv2-s-fullbq`, model-id `deimv2_s`, arch
  `deimv2-s`, imgsz 640.

---

## Step A — Create `localizer_eval_results` and its schema JSON

Per `SKILL.md` §"Schema evolution rules", schema lives in a committed JSON
file, never as inline `bq` flags. Write to:

`research/leps_localizer/scripts/schemas/localizer_eval_results_schema.json`

```json
[
  {"name": "dataset_source_uuid", "type": "STRING", "mode": "REQUIRED"},
  {"name": "model_id",            "type": "STRING", "mode": "REQUIRED"},
  {"name": "run_id",              "type": "STRING", "mode": "REQUIRED"},
  {"name": "run_at",              "type": "TIMESTAMP", "mode": "REQUIRED"},
  {"name": "arch",                "type": "STRING", "mode": "NULLABLE"},
  {"name": "imgsz",               "type": "INTEGER", "mode": "NULLABLE"},
  {"name": "predicted_bbox_xyxy", "type": "FLOAT",  "mode": "REPEATED"},
  {"name": "score",               "type": "FLOAT",  "mode": "NULLABLE"},
  {"name": "image_width",         "type": "INTEGER", "mode": "NULLABLE"},
  {"name": "image_height",        "type": "INTEGER", "mode": "NULLABLE"}
]
```

Notes:
- `predicted_bbox_xyxy` is `REPEATED FLOAT` of length 4 on hit, **length 0 on
  zero-detection** (BQ has no fixed-length array type; the consumer trusts
  length 4 or 0 by convention).
- `score` is **NULL on zero-detection** (per
  `procedures/merge-eval-into-training.md` §"Sanity check after MERGE" lines
  93–95: zero-detection still gets a row so the MERGE can distinguish "no
  detection" from "not processed").
- `image_width` / `image_height` are **denormalized** from `training_images`
  so the MERGE's `area_fraction = (x2-x1)*(y2-y1) / (image_width*image_height)`
  doesn't need a second join.
- `arch` + `imgsz` are debug metadata — cheap to store, expensive to lose if
  we ever want to compare YOLO26 @ 1280 vs YOLO26 @ 640 retrospectively.

Create the table from BEAST:

```bash
bq mk --project_id=leps-ai \
    --table \
    --time_partitioning_field=run_at \
    --time_partitioning_type=DAY \
    --clustering_fields=model_id,dataset_source_uuid \
    leps-ai:global_butterflies_2604.localizer_eval_results \
    research/leps_localizer/scripts/schemas/localizer_eval_results_schema.json

# Sanity check
bq show --schema --format=prettyjson \
    leps-ai:global_butterflies_2604.localizer_eval_results
```

Verify partition + clustering picked up:

```bash
bq show --format=prettyjson \
    leps-ai:global_butterflies_2604.localizer_eval_results \
    | jq '{timePartitioning, clustering}'
```

## Step B — Add 4 columns to `training_images`

Per `procedures/merge-eval-into-training.md` lines 11–25 + `SKILL.md`
§"Schema evolution rules" ("Never `bq update --replace_schema`"). Write the
patch JSON to:

`research/leps_localizer/scripts/schemas/training_images_bbox_patch.json`

```json
[
  {"name": "primary_subject_bbox", "type": "RECORD", "mode": "NULLABLE",
   "fields": [
     {"name": "x1", "type": "FLOAT"},
     {"name": "y1", "type": "FLOAT"},
     {"name": "x2", "type": "FLOAT"},
     {"name": "y2", "type": "FLOAT"},
     {"name": "score", "type": "FLOAT"},
     {"name": "area_fraction", "type": "FLOAT"}
   ]},
  {"name": "bbox_model_id",         "type": "STRING",    "mode": "NULLABLE"},
  {"name": "bbox_inference_run_id", "type": "STRING",    "mode": "NULLABLE"},
  {"name": "bbox_inferred_at",      "type": "TIMESTAMP", "mode": "NULLABLE"}
]
```

Apply via `--add_column` (this is the only safe path — `--replace_schema`
drops nullable columns silently):

```bash
# bq's --add_column wants the FULL new schema = existing + additions.
# Easiest path: dump existing schema, append the 4 new fields, then update.

bq show --schema --format=prettyjson \
    leps-ai:global_butterflies_2604.training_images \
    > /tmp/training_images_current_schema.json

jq -s '.[0] + .[1]' \
    /tmp/training_images_current_schema.json \
    research/leps_localizer/scripts/schemas/training_images_bbox_patch.json \
    > /tmp/training_images_new_schema.json

bq update --project_id=leps-ai \
    leps-ai:global_butterflies_2604.training_images \
    /tmp/training_images_new_schema.json

# Verify all 4 columns landed
bq show --schema --format=prettyjson \
    leps-ai:global_butterflies_2604.training_images \
    | jq '.[] | select(.name | test("bbox"))'
```

Expected output: `primary_subject_bbox` RECORD + `bbox_model_id` +
`bbox_inference_run_id` + `bbox_inferred_at`, all NULLABLE, all empty for
every existing row.

## Step C — Write `batch_localize.py`

Path: `research/leps_localizer/scripts/batch_localize.py`

Templates: copy the `_build_yolo_predictor` pattern from
`research/leps_localizer/scripts/eval_on_locked.py` (~line 34) and the
`build_deimv2_predictor` factory from
`research/leps_localizer/scripts/eval_deimv2_on_locked.py` (~line 71).

### CLI

```
python research/leps_localizer/scripts/batch_localize.py \
    --shard <path/to/shard-NNNN.parquet> \
    --out   <path/to/shard-NNNN.ndjson> \
    --weights <path/to/best.pt> \
    --run-id   <run-id-string> \
    --model-id <model-id-string> \
    --arch     <yolo26s | deimv2-s> \
    --imgsz    <int>     # 1280 for yolo26s, 640 for deimv2-s
    --batch    <int>     # 32 default, 64 on H100L MIG if VRAM allows
    --image-root /mnt/squash-0 \
    --score-thresh 0.25  \
    --device cuda:0      \
    [--deimv2-root /home/debian/Projects/DEIMv2]   # only required for deimv2-s
    [--deimv2-config <path/to/deimv2_butterflies_s.yml>]
```

### Required behavior

1. **Read parquet shard** with `pyarrow.parquet` (or `polars` — either is fine,
   parquet is ~5MB per shard, no streaming needed).
2. **Resume rule:** if `--out` already exists, count its lines. If line count
   == shard row count, print "already done, skipping" and exit 0. (Per
   `procedures/batch-inference-worklist.md` §"Resume / idempotency".)
3. **Path resolve per row** — exact code from `.claude/notes/arbutus-storage.md`
   §"Path translation":
   ```python
   on_disk = Path(args.image_root) / row["relative_local_path"]
   ```
   If `on_disk` is missing or not readable: emit a zero-detection NDJSON row
   (empty `predicted_bbox_xyxy`, `score=None`) **and continue**. Do NOT crash
   the whole shard on one missing image. Log to stderr.
4. **Batch images through the model.** Build the predictor once outside the
   loop. For YOLO use ultralytics' `model.predict([img1, img2, ...], ...)`
   list form (it handles batching internally). For DEIMv2 stack tensors into
   a `(B, 3, imgsz, imgsz)` and run a single forward.
5. **Top-1 detection per image:** highest-score box. If no boxes clear
   `--score-thresh`, emit zero-detection. **Always emit one row per image**
   — this is what lets the MERGE distinguish "no detection" from "not yet
   processed".
6. **NDJSON output, one row per image:**
   ```json
   {
     "dataset_source_uuid": "<from parquet>",
     "model_id": "yolo26s_v2",
     "run_id": "2026-05-12-yolo26s-v2-fullbq",
     "run_at": "2026-05-12T03:14:22Z",
     "arch": "yolo26s",
     "imgsz": 1280,
     "predicted_bbox_xyxy": [x1, y1, x2, y2],
     "score": 0.87,
     "image_width": 4032,
     "image_height": 3024
   }
   ```
   For zero-detection rows: `"predicted_bbox_xyxy": []`, `"score": null`.

7. **`--arch` dispatch:**
   - `--arch yolo26s` → `from ultralytics import YOLO` (lazy import — workspace
     has it). Use the same `model.predict(..., conf=score_thresh,
     imgsz=imgsz, device=device, verbose=False)` call pattern as
     `eval_on_locked.py`.
   - `--arch deimv2-s` → call `build_deimv2_predictor()` factored out of
     `eval_deimv2_on_locked.py`. Requires DEIMv2 venv at runtime; the script
     itself just `sys.path.insert(0, args.deimv2_root)` and imports
     `engine.core.YAMLConfig`. **Must be launched with DEIMv2's `.venv`
     python**, not ami-ml's.

### Output dir layout

```
/mnt/butterflies-fg-2026-05/inference/<run-id>/
  shard-0000.ndjson
  shard-0001.ndjson
  ...
  shard-NNNN.ndjson
```

### Two arch verifications before full run

```bash
# YOLO26-s smoke (one parquet shard, one image, on workspace)
.venv/bin/python research/leps_localizer/scripts/batch_localize.py \
    --shard /mnt/butterflies-fg-2026-05/worklists/canary/shard-0000.parquet \
    --out /tmp/canary-yolo.ndjson \
    --weights /mnt/butterflies-fg-2026-05/runs/yolo26s-v2/weights/best.pt \
    --run-id smoke --model-id yolo26s_v2 --arch yolo26s \
    --imgsz 1280 --batch 8 --image-root /mnt/squash-0

# Inspect
head -3 /tmp/canary-yolo.ndjson | jq .
```

## Step D — Canary on 1000-row shard

Before kicking off the full 10.7M run, validate end-to-end on a small sample.

### Build a 1000-row canary worklist (on BEAST)

```bash
RUN_ID=canary-2026-05-12-yolo26s-v2

bq query --project_id=leps-ai --use_legacy_sql=false \
    --destination_table=tmp_dataset.tmp_canary --replace \
    --allow_large_results "
SELECT dataset_source_uuid, source, photo_id, relative_local_path,
       image_width, image_height
FROM \`leps-ai.global_butterflies_2604.training_images\`
WHERE fetch_status = 'downloaded'
  AND COALESCE(corrupted, FALSE) = FALSE
  AND image_size > 10000
ORDER BY FARM_FINGERPRINT(dataset_source_uuid)
LIMIT 1000
"

# Note: the `bbox_inference_run_id != '$RUN_ID'` filter in the full-run query
# isn't needed here yet — the column will be empty for all rows until the
# first MERGE.

bq extract --project_id=leps-ai \
    --destination_format=PARQUET \
    "leps-ai:tmp_dataset.tmp_canary" \
    "gs://leps-ai-tmp/worklist/$RUN_ID/shard-*.parquet"

gsutil -m cp -r "gs://leps-ai-tmp/worklist/$RUN_ID" /tmp/worklist/
rsync -avz "/tmp/worklist/$RUN_ID/" \
    ami-workspace-02-gpu:/mnt/butterflies-fg-2026-05/worklists/$RUN_ID/
```

### Inference (on workspace VM)

```bash
ssh ami-workspace-02-gpu
cd /home/debian/Projects/ami-ml
RUN_ID=canary-2026-05-12-yolo26s-v2
WORKLIST=/mnt/butterflies-fg-2026-05/worklists/$RUN_ID
OUT=/mnt/butterflies-fg-2026-05/inference/$RUN_ID
mkdir -p "$OUT"

for shard in "$WORKLIST"/shard-*.parquet; do
    name=$(basename "$shard" .parquet)
    .venv/bin/python research/leps_localizer/scripts/batch_localize.py \
        --shard "$shard" \
        --out "$OUT/${name}.ndjson" \
        --weights /mnt/butterflies-fg-2026-05/runs/yolo26s-v2/weights/best.pt \
        --run-id "$RUN_ID" \
        --model-id yolo26s_v2 \
        --arch yolo26s \
        --imgsz 1280 \
        --batch 32 \
        --image-root /mnt/squash-0 \
        --score-thresh 0.25 \
        --device cuda:0
done
```

### Load + MERGE (back on BEAST)

```bash
RUN_ID=canary-2026-05-12-yolo26s-v2
rsync -avz ami-workspace-02-gpu:/mnt/butterflies-fg-2026-05/inference/$RUN_ID/ \
    /tmp/inference/$RUN_ID/

bq load --project_id=leps-ai \
    --source_format=NEWLINE_DELIMITED_JSON --noreplace \
    leps-ai:global_butterflies_2604.localizer_eval_results \
    "/tmp/inference/$RUN_ID/*.ndjson" \
    research/leps_localizer/scripts/schemas/localizer_eval_results_schema.json

# Now MERGE — exact SQL from procedures/merge-eval-into-training.md §"latest run wins"
bq query --project_id=leps-ai --use_legacy_sql=false \
    --parameter="run_id::$RUN_ID" "
MERGE \`leps-ai.global_butterflies_2604.training_images\` T
USING (
  SELECT dataset_source_uuid,
         predicted_bbox_xyxy[OFFSET(0)] AS x1,
         predicted_bbox_xyxy[OFFSET(1)] AS y1,
         predicted_bbox_xyxy[OFFSET(2)] AS x2,
         predicted_bbox_xyxy[OFFSET(3)] AS y2,
         score, model_id, run_id, run_at
  FROM \`leps-ai.global_butterflies_2604.localizer_eval_results\`
  WHERE run_id = @run_id
    AND ARRAY_LENGTH(predicted_bbox_xyxy) = 4
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY dataset_source_uuid ORDER BY score DESC
  ) = 1
) S
ON T.dataset_source_uuid = S.dataset_source_uuid
WHEN MATCHED THEN UPDATE SET
  primary_subject_bbox = STRUCT(
    S.x1, S.y1, S.x2, S.y2, S.score,
    SAFE_DIVIDE(
      (S.x2 - S.x1) * (S.y2 - S.y1),
      T.image_width * T.image_height
    )
  ),
  bbox_model_id = S.model_id,
  bbox_inference_run_id = S.run_id,
  bbox_inferred_at = S.run_at
"
```

(Note: I added `AND ARRAY_LENGTH(predicted_bbox_xyxy) = 4` to the source
filter so zero-detection rows are loaded into `localizer_eval_results` for
audit but skipped in the MERGE — `OFFSET(0)` on an empty array errors. The
canonical SQL in `procedures/merge-eval-into-training.md` doesn't have this
guard yet; add it there in a followup commit.)

### Canary sanity checks

Run the per-bin breakdown from `procedures/merge-eval-into-training.md` lines
82–91. **Expected ranges** (with reasons):

| Metric | Expected range | Why |
|---|---|---|
| `no_detection` (zero-bbox rows) | **<5%** | Most rows are real photos; YOLO26 missed-completely on Leeds was 9/832 ≈ 1.1%. iNat is harder framing but not 10× harder. >5% suggests path-resolution bugs (path translation wrong, `/mnt/squash-0` not mounted, etc). |
| `low_conf` (score<0.3) | **<20%** | Leeds saw ~95% conf-≥0.25 detections clear 0.3. iNat will be harder — some single-pixel-of-butterfly camera-trap-style shots — but >30% means the model is mis-firing on something systemic. |
| `area_fraction` distribution | **right-shifted** (most >0.10) | Per METHODS_AND_RESULTS §1.6 framing-skew: training data + Leeds both lean large-centered. iNat photos will skew similarly. If the distribution is uniformly small (<0.05 dominant), the model is finding leaves/background instead of butterflies. |

```sql
-- per-bin canary breakdown
SELECT
  CASE
    WHEN primary_subject_bbox IS NULL THEN 'no_detection'
    WHEN primary_subject_bbox.area_fraction < 0.01 THEN 'micro <0.01'
    WHEN primary_subject_bbox.area_fraction < 0.05 THEN '<0.05'
    WHEN primary_subject_bbox.area_fraction < 0.10 THEN '<0.10'
    WHEN primary_subject_bbox.area_fraction < 0.25 THEN '<0.25'
    WHEN primary_subject_bbox.area_fraction < 0.50 THEN '<0.50'
    ELSE '>=0.50'
  END AS bucket,
  COUNT(*) AS n,
  AVG(primary_subject_bbox.score) AS avg_score
FROM `leps-ai.global_butterflies_2604.training_images`
WHERE bbox_inference_run_id = 'canary-2026-05-12-yolo26s-v2'
GROUP BY 1 ORDER BY bucket;
```

### Manual audit of one low-conf image

Pick one `dataset_source_uuid` with `score < 0.2`, resolve to `/mnt/squash-0/{bucket:03d}/<uuid>.jpg`,
run the model interactively, confirm the low confidence is **about the
image**, not a script bug:

```bash
# On workspace VM
ssh ami-workspace-02-gpu
.venv/bin/python -c "
from pathlib import Path
from ultralytics import YOLO
m = YOLO('/mnt/butterflies-fg-2026-05/runs/yolo26s-v2/weights/best.pt')
r = m.predict('/mnt/squash-0/<bucket>/<uuid>.jpg', imgsz=1280, conf=0.05)[0]
print(r.boxes.conf, r.boxes.xyxy)
"
```

If the result matches the canary NDJSON for that row → script is correct,
low conf is real. If it differs → debug the dispatch / preprocessing in
`batch_localize.py`.

### Canary gate

All three: **proceed with full run** if `no_detection<5%`, `low_conf<20%`,
area_fraction distribution right-shifted, **and** one manual audit
matches. Any of those fail → diagnose before full run.

---

## Full run protocol

Recipe verbatim from
`.claude/skills/bigquery-leps/procedures/batch-inference-worklist.md` — only
the wrapping differs from canary:

1. **On BEAST**: build worklist with `bq query` filter, `bq extract` to
   parquet shards (~10–50 shards depending on partition), `gsutil cp` to
   `/tmp/worklist/`, `rsync` to workspace.
2. **On workspace**: loop over shards, `batch_localize.py` each. NDJSON
   shards land in `/mnt/butterflies-fg-2026-05/inference/<run-id>/`.
3. **Back on BEAST**: `rsync` NDJSON back, `bq load` to
   `localizer_eval_results`, MERGE to `training_images`.

### Wall-time estimates

Rough — these are extrapolations, not measurements:

| Arch | imgsz | H100L MIG throughput (single proc) | 10.7M wall time |
|---|---|---|---|
| YOLO26-s | 1280 | ~50–80 img/s (estimate from FG val bench) | **37–60 hours** |
| DEIMv2-S | 640 | ~150 img/s (estimate, DETR @ smaller imgsz) | **~20 hours** |

Measure on canary (1000 / wall-seconds → img/s) and re-estimate before
committing to the full launch.

### Launch in a way that survives SSH disconnect

Per memory `feedback_ssh_background_proc_setsid.md`: `nohup CMD &` over ssh
**dies on disconnect**. Wrap with `setsid bash -c "nohup ... &"`:

```bash
# On workspace VM
RUN_ID=2026-05-12-yolo26s-v2-fullbq

setsid bash -c "
  cd /home/debian/Projects/ami-ml
  nohup bash -c '
    for shard in /mnt/butterflies-fg-2026-05/worklists/$RUN_ID/shard-*.parquet; do
      name=\$(basename \$shard .parquet)
      .venv/bin/python research/leps_localizer/scripts/batch_localize.py \
        --shard \$shard \
        --out /mnt/butterflies-fg-2026-05/inference/$RUN_ID/\${name}.ndjson \
        --weights /mnt/butterflies-fg-2026-05/runs/yolo26s-v2/weights/best.pt \
        --run-id $RUN_ID \
        --model-id yolo26s_v2 \
        --arch yolo26s \
        --imgsz 1280 \
        --batch 32 \
        --image-root /mnt/squash-0 \
        --score-thresh 0.25 \
        --device cuda:0
    done
  ' > /mnt/butterflies-fg-2026-05/inference/$RUN_ID/run.log 2>&1 < /dev/null &
"
```

**Belt + suspenders:** also run inside `tmux new -s bulk-localize` on the
workspace VM. Either alone is enough; both is cheap insurance against a 60-hour
job losing its parent process.

### Sharding rule

Process shards in **alphabetical order**. The `bbox_inference_run_id != '$RUN_ID'`
filter (in the worklist query, see `procedures/batch-inference-worklist.md`
line 33) means if the run is interrupted and you re-extract the worklist,
already-MERGEd rows are filtered out — so resume is automatic. The
per-shard `len(NDJSON) == len(parquet)` check inside `batch_localize.py` is
the **finer-grain** resume (don't re-process shards that finished).

### Multi-GPU sharding (if 2× 3090 NVLink is available)

Per `procedures/batch-inference-worklist.md` §"Multi-GPU sharding": split
shards by first hex char of md5(filename). Two parallel `for` loops on the
two GPUs. No coordination — disjoint shard names → disjoint output files.
Cuts wall time roughly in half.

---

## Verification gates

| Stage | Pass criteria |
|---|---|
| Canary 1000 | `no_detection<5%`, `low_conf<20%`, area_fraction right-shifted, one manual audit matches |
| Full run (in-flight) | Every parquet shard has a matching NDJSON file with row count = parquet row count |
| Post-`bq load` | `SELECT COUNT(*) FROM localizer_eval_results WHERE run_id=@run_id` ≈ 10.7M (minus shards skipped by worklist filter) |
| Post-MERGE | `SELECT COUNT(*) FROM training_images WHERE primary_subject_bbox IS NULL` should be small — limited to failed-download/corrupted rows and the zero-detection NDJSON rows. **Not** limited to "model missed it" since zero-detection rows DON'T MERGE (per the `ARRAY_LENGTH = 4` guard); decide whether to surface those separately or leave `primary_subject_bbox=NULL` as "either no image or no detection." |
| Spot-check | Pick 3 `dataset_source_uuid` rows post-MERGE: pull image, render bbox, visually verify it tracks the subject. |

---

## Followups (defer until both YOLO26 + DEIMv2 full runs complete)

1. **Filter micro-detections** before classifier retraining. Use
   `primary_subject_bbox.area_fraction < 0.005` as a strong signal that the
   model didn't actually find the subject (a 0.005 box on a 4000×3000 image
   is ~280×210 px — at that ratio you're past the small bin into noise).
   Cut these from the classifier training pool.
2. **Re-crop training images** around the primary bbox + retrain classifier
   on consistent framing. Padding strategy = pad to square with the bbox
   centered (matches `_pad_to_square` in
   `src/classification/dataloader.py:18-29` but bbox-aware).
3. **DEIMv2 vs YOLO26 head-to-head on the full 10.7M.** Currently we have
   832 Leeds + 290 FG val to rank these two. With 10.7M images each given a
   prediction by both models, we can compute true cross-model agreement,
   disagreement-stratified accuracy, and the "DEIMv2 finds something where
   YOLO26 doesn't" rate at scale. This is the real version of §5.4's E2E
   classifier-acc eval.
4. **Promote** `batch_localize.py` into `src/localization/` once stable —
   per `CLAUDE.md` §"Code consolidation plan", new work lands in
   `research/leps_localizer/scripts/` and graduates after interfaces settle.

---

## Key references

Read first:
- `/home/michael/Projects/AMI/ami-ml/.claude/skills/bigquery-leps/SKILL.md` —
  two-host workflow, schema-evolve rules, cost shape
- `/home/michael/Projects/AMI/ami-ml/.claude/skills/bigquery-leps/procedures/batch-inference-worklist.md` —
  pipeline recipe (worklist → parquet → rsync → infer → NDJSON → load)
- `/home/michael/Projects/AMI/ami-ml/.claude/skills/bigquery-leps/procedures/merge-eval-into-training.md` —
  MERGE SQL, both "latest-wins" and "best-score-across-models" variants
- `/home/michael/Projects/AMI/ami-ml/.claude/notes/arbutus-storage.md` —
  path translation `relative_local_path` → `/mnt/squash-0/{bucket:03d}/<uuid>.jpg`

Templates to copy from:
- `/home/michael/Projects/AMI/ami-ml/research/leps_localizer/scripts/eval_on_locked.py` —
  YOLO predictor pattern (`_build_yolo_predictor`)
- `/home/michael/Projects/AMI/ami-ml/research/leps_localizer/scripts/eval_deimv2_on_locked.py` —
  DEIMv2 predictor factory (`build_deimv2_predictor`)

Context:
- `/home/michael/Projects/AMI/ami-ml/research/leps_localizer/NEXT_SESSION.md` —
  task list (this prompt = line 63: "Batch BQ inference over 10.7M…")
- `/home/michael/Projects/AMI/ami-ml/research/leps_localizer/METHODS_AND_RESULTS.md` —
  §5.3 model ranking on small/mid bins (justifies YOLO26-s v2 as first model);
  §1.6 framing-skew (sanity-check expectation for area_fraction distribution)
- `/home/michael/.claude/projects/-home-michael-Projects-AMI-ami-ml/memory/feedback_leeds_easy_mode.md` —
  why small-ratio recall is the real signal
- `/home/michael/.claude/projects/-home-michael-Projects-AMI-ami-ml/memory/feedback_ssh_background_proc_setsid.md` —
  `setsid bash -c "nohup ... &"` for ssh-disconnect survival

Weights:
- YOLO26-s v2: `/mnt/butterflies-fg-2026-05/runs/yolo26s-v2/weights/best.pt`
  (workspace VM)
- DEIMv2-S: `/mnt/butterflies-fg-2026-05/runs/deimv2-s-fg-2026-05/best_stg1.pth`
- DEIMv2 config: `/home/debian/Projects/DEIMv2/configs/deimv2_butterflies_s.yml`
- DEIMv2 venv (must use for DEIMv2 inference):
  `/home/debian/Projects/DEIMv2/.venv/`
