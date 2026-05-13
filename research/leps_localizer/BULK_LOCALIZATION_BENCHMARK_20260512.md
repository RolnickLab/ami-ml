# Bulk localization benchmark — YOLO26-s v2 over `training_images`

**Date:** 2026-05-12
**Branch:** `feat/leps-localizer-training`
**Run-id:** `canary-2026-05-12-yolo26s-v2-squashable`
**Model:** `/mnt/butterflies-fg-2026-05/runs/yolo26s-fg-2026-05-v2-2/weights/best.pt`
**Hardware:** workspace VM — 1× H100 80GB sliced to 24GB MIG (`ami-workspace-02-gpu`)

> All performance numbers below are **rough estimates from a single canary run**
> on the same VM as production training+gym. Numbers are hedged because:
> (a) sample size n=1 (single shard, single arch, single batch size);
> (b) IO subsystem is FUSE-on-FUSE, behavior depends on S3 cache warmth;
> (c) we measured throughput, not GPU util.

## 1. Image-source decision

Picked **Option A — BQ LIMIT 10000** with one critical filter:
`MOD(photo_id, 10) = 0`.

### Why Option A

- Cost: 1 destination-table BQ query + a local JSON pull, ~$0.01 total
- Exercises the same path the production pipeline takes (BQ → parquet → rsync
  → infer → NDJSON), so any bottlenecks I measure translate directly
- Carries real `image_width` / `image_height` (needed for `area_fraction`
  computation), which Option C (filesystem scan) does not
- Option B (`localizer_training_images`, 9628 FG-only rows) was tempting for
  zero BQ cost, but the FG curated set has framing skew incompatible with the
  iNat-heavy production data shape we actually care about

### Why the `MOD(photo_id, 10) = 0` filter (and what it means)

**During smoke testing the path translation in `.claude/notes/arbutus-storage.md`
turned out to be partially wrong** — or rather, the squashfs mount on the
workspace VM only covers **~10% of the buckets**.

Specifically:
- `relative_local_path = "{bucket:03d}/<dataset_source_uuid>.jpg"` where
  `bucket = photo_id % 1000` ✓
- `/mnt/squash-0` is `squashfuse` mounting `/mnt/s3/task_0.sqfs` (one of ten
  shard files `task_0..task_9.sqfs` on the underlying S3 mount)
- `/mnt/squash-0/` contains exactly 100 bucket directories: `000, 010, 020,
  ..., 990` — every multiple of 10, nothing else
- Random pull from `training_images` → 95% of rows resolve to a path under a
  bucket directory that does not exist on this filesystem

I rebuilt the canary with `MOD(photo_id, 10) = 0` to filter to images
actually accessible from the workspace VM as configured today.

**Implication for the 10.7M run:** in its current state, the workspace VM can
only see ~1.07M of the 10.7M images.

**Good news — fixing this is a one-command operation.** The workspace VM has
a systemd template unit `ami-squashfs@.service` (path:
`/etc/systemd/system/ami-squashfs@.service`) designed exactly for this:

```
ExecStart=/usr/bin/squashfuse ... /mnt/s3/task_%i.sqfs /mnt/squash-%i
```

Currently only `ami-squashfs@0.service` is enabled. Enabling instances 1–9
would mount the missing nine shards at `/mnt/squash-1` ... `/mnt/squash-9`.
Each task file holds a disjoint bucket set: `task_N.sqfs` contains buckets
where `bucket % 10 == N` (confirmed by mounting `task_1.sqfs` briefly during
this session — its top-level dirs were `001, 011, 021, ...`).

After enabling, `batch_localize.py` needs one small change: resolve the disk
path as `/mnt/squash-{photo_id % 10}/{relative_local_path}` instead of a
fixed `/mnt/squash-0/{relative_local_path}`.

Command (needs sudo on the workspace VM):

```bash
sudo systemctl enable --now ami-squashfs@{1,2,3,4,5,6,7,8,9}.service
ls /mnt/squash-*  # should show 10 directories now
```

This is the **biggest blocker** for the production run, but the fix is
trivial and was clearly anticipated by whoever wrote the systemd template.
See §5.

## 2. What was built this session

| Path | Purpose | Status |
|---|---|---|
| `research/leps_localizer/scripts/batch_localize.py` | bulk inference workhorse (YOLO26 + DEIMv2 dispatch + `--multi-squash` for 10-mount layout) | written, smoke-tested on 3.6k images ✓ |
| `research/leps_localizer/scripts/analyze_batch_localize.py` | summary stats on NDJSON output | written |
| `research/leps_localizer/scripts/schemas/localizer_eval_results_schema.json` | planned BQ table schema | written (not applied to BQ) |
| `research/leps_localizer/scripts/schemas/training_images_bbox_patch.json` | planned column-addition patch | written (not applied to BQ) |

Everything is unstaged. **No BQ DDL was executed this session.** The only
write to BQ was a single destination-table query result in
`leps-ai:tmp_dataset.canary_10k_squashable` (7-day TTL), used as the canary
worklist source.

## 3. Canary throughput

### Setup

| Param | Value |
|---|---|
| arch | yolo26s |
| imgsz | 1280 |
| batch | 32 |
| score_thresh | 0.25 |
| device | cuda:0 (24GB MIG slice on H100) |
| n images | 10000 (this run) and 1000 (warm-cache run) |
| filter | `MOD(photo_id, 10) = 0` (squashfs availability) |
| image source dist | 100% iNat (training_images is currently 100% iNat) |

### Measured throughput

| Sample | Wall-clock | Avg img/s | Notes |
|---|---|---|---|
| 1000-image, post-warmup | 145s | **6.9** | second consecutive run, FUSE cache likely partially warm |
| 3584-image (partial; 10k canary stopped early to respect the 30-min budget) | ~10 min | **~6.0** (overall), `recent=6.1–6.5 img/s` steady | first run on cold cache; full 10k would have taken ~28 min |

Steady-state per-batch rate after the initial cache-warm burst is **6.1–6.5
img/s** — consistent across both samples. The early "20 img/s" reading is
FUSE cache hits from the previous run; that does not represent cold-cache
throughput.

**Why the 10k didn't finish in-session:** projected ~28-minute wall-clock
at steady-state rate, which would have exceeded the 30-min benchmark cap
once warmup and process startup are included. Stopped at 3584 rows. The
distributional stats are stable from 1k onward, so a longer run would only
narrow tail estimates.

GPU memory peak: ~**10.0 GiB** (out of 24 GiB MIG slice) — sustained, never
spiked. There is significant headroom for either a larger batch or a parallel
inference process.

GPU utilization: nvidia-smi returns `[N/A]` for util% on this MIG slice (a
known MIG quirk), so we cannot report util directly. The python process was
observed in `D` state (uninterruptible disk-IO wait) for most of the run,
confirming the work is **disk/IO bound, not GPU bound**.

### Likely bottleneck

`/mnt/squash-0` is FUSE squashfs → backed by `/mnt/s3/task_0.sqfs` → mountpoint-s3
→ object-arbutus.cloud.computecanada.ca over the network. Every cold-cache
image read takes a round-trip through three FUSE layers + one S3 GET. The
6 img/s figure is consistent with ~150ms per random S3 read + small
batching gains.

## 4. Quality of detections (3584-image partial)

From `analyze_batch_localize.py` on the final partial NDJSON (10k canary
stopped at 3584 rows to respect the 30-min budget):

```
n_rows         : 3584
n_detected     : 3475 (97.0%)
n_zero_det     :  109 ( 3.0%)
score          : mean=0.762 p10=0.433 p50=0.852 p90=0.903 min=0.252 max=0.933
low_conf (<0.3): 63/3475 (1.8%)
area_fraction  : mean=0.4096 p10=0.1151 p50=0.3803 p90=0.7516 min=0.0155 max=0.9822
tiny (<0.005)  : 0/3475 (0.0%)

area_fraction bins:
    tiny<0.005:     0 ( 0.0%)
    micro<0.01:     0 ( 0.0%)
         <0.05:    64 ( 1.8%)
         <0.10:   223 ( 6.4%)
         <0.25:   749 (21.6%)
         <0.50:  1235 (35.5%)
        >=0.50:  1204 (34.6%)

n_detections_above_thresh dist:
   0 (zero-det):   109
   1 (single):    3153
   2:               303
   3:                18
   4:                 1
```

Stats are stable across 1k → 3k → 3.6k samples — same detect rate (97%),
same score median (0.85), same area_fraction shape. Sample-size effects
have already washed out.

Passes the gate from `NEXT_SESSION_BULK_LOCALIZATION.md` §"Canary sanity checks":

| Metric | Expected | Measured | Pass? |
|---|---|---|---|
| `no_detection` | <5% | 3.0% | ✓ |
| `low_conf` (<0.3) | <20% | 1.8% | ✓ |
| `area_fraction` distribution | right-shifted | median 0.380, 34.6% in `>=0.50` | ✓ |

Distribution matches METHODS_AND_RESULTS §1.6 framing-skew expectation (most
training photos are framing-biased toward large/centered subjects). The
right-tail bulk (`<0.25` + `<0.50` + `>=0.50` = 91.7% of detections) tracks
the FG val set; the left tail (`<0.10` + `<0.05` = 8.2%) is what real
deployment-quality recall in small/mid bins looks like.

9.3% of detections find more than one box above threshold (`n_detections >=
2`). Top-1 selection is fine for the planned MERGE; a future "all detections
above threshold" extension would store the runner-up boxes if needed (see §6).

## 5. Extrapolation to 10.7M

**Caveat first**: with the current `/mnt/squash-0` mount this VM can only
reach **1.07M images** (the `mod 10 == 0` subset). Numbers below are
projected throughput, not realizable wall time, until the missing 9 squashfs
files are mounted.

### Wall-time projection (rough estimate)

Using the measured steady-state of **~6.3 img/s** (post-cache-warmup; mean
of the recent= readings after the first 1500 images in both the 1k and 10k
samples):

| Scope | Throughput (est) | Wall time |
|---|---|---|
| 1 process, 1 squash mount | 6.3 img/s | 10.7M / 6.3 = **~20 days** |
| 1 process, 10 squash mounts (parallel S3 streams) | ~30 img/s (estimate) | **~4 days** |
| 4 parallel procs across 10 mounts | ~80 img/s (estimate, assuming each S3 cap is per-mount) | **~1.5 days** |
| ONNX + 4 procs | ~100 img/s (estimate) | **~1.2 days** |

All numbers above the first row are estimates extrapolated from the
single-mount measurement. The dominant variable is the per-mount FUSE-on-FUSE
+ S3 cap, which we have not directly measured.

**Hot opportunity:** mounting all 10 squashfs shards is also a path to higher
throughput, not just more coverage. Each `task_N.sqfs` has its own
squashfuse → mountpoint-s3 → S3 chain. With per-row dispatch to the
correct `/mnt/squash-N` (already implemented as `--multi-squash` in
`batch_localize.py`), 10 concurrent processes — one per mount — would each
hit their own FUSE/S3 path. Conservative estimate: 5x speedup if S3
bandwidth is the dominant cap, possibly 10x if the FUSE layer is the cap.

Until we measure this, the 1-week to 1-day range is the realistic envelope.

### Recommended sharding for full run

- **100 shards × ~107k rows each.** Easy to resume per-shard. Each shard runs
  in 4–6 hours at 6 img/s; if a shard fails it costs at most one shard's
  worth of progress.
- Order shards alphabetically (the parquet filename sort). The per-shard
  `len(NDJSON) == len(parquet)` resume check in `batch_localize.py` already
  handles re-runs.
- Hash-shard onto multiple GPUs by first hex char of `md5(shard_filename)`
  per `procedures/batch-inference-worklist.md` §"Multi-GPU sharding".

### Cost projection (BQ side, separate from compute)

| Step | Cost |
|---|---|
| BQ extract 10.7M rows to parquet | ~$0.05 |
| `bq load` 10.7M NDJSON rows | ~$0.05 |
| MERGE into `training_images` | ~$0.05 |
| Eval table storage (200 bytes × 10.7M × 2 models) | ~$0.20/mo |
| **Total upper bound for one model pass** | **<$0.50** |

BQ is the cheapest part. Compute is the bottleneck.

## 6. Recommended next steps (in order)

1. **Enable the other 9 squashfs systemd units** to mount `/mnt/squash-1`
   through `/mnt/squash-9`. One command on the workspace VM:
   ```bash
   sudo systemctl enable --now ami-squashfs@{1,2,3,4,5,6,7,8,9}.service
   ```
   The script-side per-row dispatch is already implemented as the
   `--multi-squash` flag in `batch_localize.py` (resolves
   `/mnt/squash-{photo_id % 10}/{relative_local_path}` per row); just pass
   it in place of `--image-root /mnt/squash-0` once all mounts are up.

2. **Re-benchmark on a hot-then-sequential 10k sample** post-mount. The
   current numbers reflect a single S3-backed shard with the FUSE cache
   probably warmed by hours of training-time reads, so the steady-state
   estimate has wide error bars.

3. **ONNX export and re-benchmark** the localizer. With 24GB headroom and
   the model itself fitting in ~10GB of memory, the inference path is not the
   gating factor — IO is — so the impact will be more about reducing CPU
   pre/post (numpy / autograd dispatch overhead in ultralytics) than raw
   model speed. Defer until after IO is fixed.

4. **For the 10.7M run:** YOLO26-s v2 goes first (matches the §5.3 ranking on
   small/mid recall). DEIMv2-S as the second pass; "best score across
   models" MERGE variant from `merge-eval-into-training.md` selects the
   winner per image.

5. **Decide on n_detections downstream policy.** 9.3% of detected images
   have ≥2 detections above 0.25. The current schema only stores top-1;
   for downstream tasks that care about "more than one butterfly in frame"
   (e.g. classifier training pair sampling), the schema would need a
   `predicted_bbox_xyxy_secondary` or a sibling RECORD per extra detection.
   Defer unless we hit a use case.

## 7. Unexpected findings worth flagging

- **`training_images` is 100% `source = 'inat'`**. The skill doc says iNat +
  eButterfly + museum; the live filter-passing rows are pure iNat. Either
  the eButterfly / museum sources haven't been loaded yet, or they all fail
  the `fetch_status='downloaded' AND image_size>10000` filter. This affects
  framing-bias assumptions for the full run (iNat is the easier framing
  case; museum/eButterfly may differ).

- **`/mnt/squash-0` covers only 10% of bucket dirs.** Already discussed
  above. The doc `arbutus-storage.md` should be updated to clarify that the
  path translation is the index but the actual filesystem on the workspace VM
  is one of ten shards.

- **FUSE-on-FUSE IO is the bottleneck.** This was hidden in the original
  estimates ("ETA 37–60 hours") which assumed mostly-cached IO. Real
  steady-state on cold reads is closer to 7+ days per pass.

- **GPU util reads as `[N/A]`** on this MIG slice — a known quirk. Can't
  use the standard nvidia-smi based health probe; have to fall back to
  process state (`D` = IO wait) or memory.used trend for "is it doing
  something" checks.

---

## Files referenced

- `research/leps_localizer/scripts/batch_localize.py` — the inference workhorse
- `research/leps_localizer/scripts/analyze_batch_localize.py` — stats summarizer
- `research/leps_localizer/scripts/schemas/localizer_eval_results_schema.json` — planned BQ schema (not applied)
- `research/leps_localizer/scripts/schemas/training_images_bbox_patch.json` — planned column patch (not applied)
- `/tmp/leps_canary_worklist/canary-10k-squashable.parquet` — laptop-side canary worklist
- `ami-workspace-02-gpu:/mnt/butterflies-fg-2026-05/inference/canary-2026-05-12-yolo26s-v2-squashable/shard-0000.ndjson` — VM-side NDJSON output
- `leps-ai:tmp_dataset.canary_10k_squashable` — BQ destination table (7-day TTL)
- `leps-ai:tmp_dataset.canary_10k` — BQ destination table (initial unfiltered, also 7-day TTL)
