#!/usr/bin/env python3
"""
Download images from training_images BQ table to a local staging directory,
record results in training_images_downloads, merge status back into
training_images, then pack images into SquashFS chunks.

Images are split across parallel jobs using MOD(photo_id, num_jobs) = task_id
so each job handles a balanced, non-overlapping subset of images.

Resumable: already-attempted images are skipped by LEFT JOINing with
training_images_downloads. Re-running the same task_id is safe.

NOTE — mid-chunk restart behaviour: if the job dies after downloading a chunk
but before the BQ write, those images are on disk but unrecorded. On resume
the LEFT JOIN will re-queue them and they will be re-downloaded. No data is
lost but ~chunk_size images are downloaded twice. This is acceptable given
the low probability and low cost of a single chunk redo.

Usage (single job / test):
    python download_images.py \\
        --staging-dir /scratch/$USER/staging \\
        --num-jobs 1 --task-id 0 \\
        --limit 50 --table-prefix test_

Usage (SLURM array):
    python download_images.py \\
        --staging-dir /scratch/$USER/staging \\
        --num-jobs 10 --task-id $SLURM_ARRAY_TASK_ID

After all array tasks finish, run job_bq_pack_per_task.sh to merge
chunk sqfs files into the final task_N.sqfs archives.
"""

# ─────────────────────────────────────────────────────────────────────────────
# COST NOTE FOR AI ASSISTANTS (and humans) — READ BEFORE ADDING BQ CALLS HERE
#
# This script runs at scale against large tables. Know the sizes before you
# add or change any BigQuery statement:
#
#   training_images           ~24M rows  /  ~7 GB   (global_all_leps_2605)
#   training_images_downloads grows to a comparable size as downloads complete
#
# BigQuery on-demand pricing is $5 per TB *scanned* (not per row written).
# Two rules that matter most in this file:
#
#   1. A MERGE / UPDATE / DELETE is billed mainly for scanning the *target*
#      table, regardless of how few rows the source touches. Running one inside
#      the per-chunk loop re-scans all ~7 GB on every chunk — at chunk_size
#      10000 that is ~2400 full-table scans (~17 TB, ~$85) for a single pass.
#      Do DML *once per run*, not once per chunk. See
#      merge_downloads_into_training_images() below.
#
#   2. Appending rows via load_table_from_dataframe() (batch load) is free and
#      parallel-safe — prefer it over streaming inserts or per-row DML. The
#      downloads table is the source of truth for progress; status only needs
#      to land in training_images once, at the end.
#
# Before adding a query: estimate bytes scanned (a full SELECT * over
# training_images is ~7 GB = ~$0.035 each — cheap once, expensive in a loop),
# filter/project columns, and never put unbounded DML inside a loop. To verify
# real spend, check INFORMATION_SCHEMA.JOBS_BY_PROJECT (total_bytes_billed).
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import PIL
import requests
from PIL import Image
from google.api_core import exceptions as google_exceptions
from google.cloud import bigquery

Image.MAX_IMAGE_PIXELS = None

BQ_PROJECT = "leps-ai"
BQ_DEFAULT_DATASET = "global_butterflies_2604"

# Retry config for HTTP downloads
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES    = 5
_BACKOFF_BASE   = 2.0   # seconds
_BACKOFF_MAX    = 60.0  # seconds cap
_MERGE_MAX_RETRIES = 10  # BQ MERGE serialization conflicts (concurrent tasks)

# Warn if this many chunk sqfs files accumulate (pack job falling behind)
_CHUNK_ACCUMULATION_WARN = 20

DOWNLOADS_SCHEMA = [
    bigquery.SchemaField("dataset_source_uuid", "STRING"),
    bigquery.SchemaField("fetch_status",        "STRING"),
    bigquery.SchemaField("image_width",         "INTEGER"),
    bigquery.SchemaField("image_height",        "INTEGER"),
    bigquery.SchemaField("image_size",          "INTEGER"),
    bigquery.SchemaField("corrupted",           "BOOLEAN"),
]

# Thread-local storage for per-thread requests sessions
_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Return a per-thread requests.Session with a single keep-alive connection."""
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=1,
            max_retries=0,  # retries handled manually below
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _thread_local.session = s
    return _thread_local.session


def _fetch_with_retry(url: str, dest: Path) -> None:
    """Download url → dest with exponential backoff + jitter.

    Retries on rate-limit (429), transient server errors (5xx),
    connection errors (including Errno 16 — too many open sockets),
    and timeouts.  Raises on permanent client errors (4xx except 429)
    or after exhausting all retries.
    """
    session = _get_session()
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=30, stream=True)
            if resp.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES:
                delay = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_MAX)
                delay += random.uniform(0, delay * 0.25)
                print(f"  HTTP {resp.status_code} {url} — retry {attempt+1}/{_MAX_RETRIES} "
                      f"in {delay:.1f}s", flush=True)
                time.sleep(delay)
                continue
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return
        except requests.exceptions.ConnectionError as e:
            if attempt < _MAX_RETRIES:
                delay = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_MAX)
                delay += random.uniform(0, delay * 0.25)
                print(f"  ConnectionError {url} — retry {attempt+1}/{_MAX_RETRIES} "
                      f"in {delay:.1f}s: {e}", flush=True)
                time.sleep(delay)
            else:
                raise
        except requests.exceptions.Timeout:
            if attempt < _MAX_RETRIES:
                delay = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_MAX)
                delay += random.uniform(0, delay * 0.25)
                print(f"  Timeout {url} — retry {attempt+1}/{_MAX_RETRIES} "
                      f"in {delay:.1f}s", flush=True)
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Exhausted {_MAX_RETRIES} retries for {url}")


def download_and_verify(row: dict, staging_dir: Path) -> dict:
    """Download one image, verify with PIL, return result dict."""
    url      = row["absolute_url"]
    rel_path = row["relative_local_path"]
    dest     = staging_dir / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "dataset_source_uuid": row["dataset_source_uuid"],
        "fetch_status": None,
        "image_width":  None,
        "image_height": None,
        "image_size":   None,
        "corrupted":    None,
    }

    try:
        _fetch_with_retry(url, dest)
    except Exception as e:
        print(f"  Failed {url}: {e}", flush=True)
        result["fetch_status"] = "failed"
        return result

    try:
        with Image.open(dest) as img:
            img.convert("RGB")
            result["image_width"], result["image_height"] = img.size
        result["image_size"]   = dest.stat().st_size
        result["corrupted"]    = False
        result["fetch_status"] = "downloaded"
    except (PIL.UnidentifiedImageError, OSError) as e:
        print(f"  Corrupted {url}: {e}", flush=True)
        result["corrupted"]    = True
        result["fetch_status"] = "corrupted"

    return result


def ensure_downloads_table(client: bigquery.Client, downloads_table: str) -> None:
    """Create the downloads table if it doesn't exist."""
    try:
        client.get_table(downloads_table)
    except Exception:
        table = bigquery.Table(downloads_table, schema=DOWNLOADS_SCHEMA)
        table.description = (
            "Download results for training_images. One row per download attempt. "
            "Appended to by parallel download jobs. Used to track fetch progress "
            "without DML updates on the base training_images table."
        )
        client.create_table(table)
        print(f"Created table {downloads_table}", flush=True)


def write_results_to_bq(
    client: bigquery.Client,
    results: list[dict],
    downloads_table: str,
    max_retries: int = 3,
) -> None:
    """Append download results to the downloads table via batch load (free tier).

    Multiple parallel tasks can safely append simultaneously.
    Retries up to max_retries times on transient BQ errors.
    """
    df = pd.DataFrame(results)
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=DOWNLOADS_SCHEMA,
    )
    for attempt in range(max_retries):
        try:
            job = client.load_table_from_dataframe(df, downloads_table, job_config=job_config)
            job.result()
            return
        except Exception as e:
            if attempt < max_retries - 1:
                delay = 30 * (attempt + 1)
                print(f"  BQ write failed (attempt {attempt+1}/{max_retries}): {e} "
                      f"— retrying in {delay}s", flush=True)
                time.sleep(delay)
            else:
                raise


def merge_downloads_into_training_images(
    client: bigquery.Client,
    training_table: str,
    downloads_table: str,
) -> int:
    """MERGE all recorded download results into training_images, once per run.

    Updates fetch_status (and dims) for every outcome:
      downloaded — fetch_status='downloaded', dims and corrupted populated
      corrupted  — fetch_status='corrupted',  corrupted=True, dims NULL
      failed     — fetch_status='failed',     all fields NULL

    Permanently failed images (404, 403, exhausted retries) are marked
    fetch_status='failed' so they are excluded from future re-runs via
    the WHERE fetch_status='pending' clause — no wasted retry attempts.
    Retrying can still be done intentionally via retry_failed_downloads.py.

    Source is the downloads table, which already holds every result (appended
    per chunk via the free batch-load tier). Because a MERGE is billed mainly
    for scanning the *target* table, running it once per run instead of once
    per chunk turns N full-table scans into a single one. The source is
    deduplicated per dataset_source_uuid (the table is append-only and
    --force-redownload can append a second row), preferring a successful
    outcome over a failed one.

    Only updates rows still 'pending' — safe to re-run and safe from parallel
    tasks. Returns the number of rows updated.
    """
    merge_sql = f"""
        MERGE `{training_table}` T
        USING (
          SELECT
            dataset_source_uuid,
            ARRAY_AGG(
              STRUCT(fetch_status, image_width, image_height, image_size, corrupted)
              ORDER BY CASE fetch_status
                         WHEN 'downloaded' THEN 0
                         WHEN 'corrupted'  THEN 1
                         ELSE 2
                       END
              LIMIT 1
            )[OFFSET(0)] AS r
          FROM `{downloads_table}`
          WHERE fetch_status IN ('downloaded', 'corrupted', 'failed')
          GROUP BY dataset_source_uuid
        ) S
          ON T.dataset_source_uuid = S.dataset_source_uuid
        WHEN MATCHED AND (T.fetch_status = 'pending' OR T.fetch_status IS NULL) THEN UPDATE SET
          T.fetch_status  = S.r.fetch_status,
          T.image_width   = S.r.image_width,
          T.image_height  = S.r.image_height,
          T.image_size    = S.r.image_size,
          T.corrupted     = S.r.corrupted
        """
    # Concurrent MERGEs from parallel tasks can collide with
    # "Could not serialize access ... due to concurrent update" (400).
    # BQ docs recommend retrying — back off with jitter until a slot frees.
    for attempt in range(_MERGE_MAX_RETRIES + 1):
        try:
            job = client.query(merge_sql)
            job.result()
            return job.dml_stats.updated_row_count
        except google_exceptions.BadRequest as e:
            if "serialize" not in str(e).lower() or attempt >= _MERGE_MAX_RETRIES:
                raise
            delay = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_MAX)
            delay += random.uniform(0, delay * 0.5)
            print(f"  MERGE serialization conflict — retry "
                  f"{attempt+1}/{_MERGE_MAX_RETRIES} in {delay:.0f}s", flush=True)
            time.sleep(delay)
    # Unreachable: the loop either returns or raises on the final attempt.
    return 0


def get_pending_rows(
    client: bigquery.Client,
    training_table: str,
    downloads_table: str,
    num_jobs: int,
    task_id: int,
    limit: int | None = None,
    force_redownload: bool = False,
) -> list[dict]:
    """Query pending images for this task, skipping already-attempted ones."""
    limit_clause = f"LIMIT {limit}" if limit else ""
    if force_redownload:
        query = f"""
        SELECT dataset_source_uuid, absolute_url, relative_local_path
        FROM `{training_table}`
        WHERE (fetch_status = 'pending' OR fetch_status IS NULL)
          AND MOD(photo_id, {num_jobs}) = {task_id}
        {limit_clause}
        """
    else:
        query = f"""
        SELECT ti.dataset_source_uuid, ti.absolute_url, ti.relative_local_path
        FROM `{training_table}` ti
        LEFT JOIN `{downloads_table}` d
          ON ti.dataset_source_uuid = d.dataset_source_uuid
        WHERE (ti.fetch_status = 'pending' OR ti.fetch_status IS NULL)
          AND MOD(ti.photo_id, {num_jobs}) = {task_id}
          AND d.dataset_source_uuid IS NULL
        {limit_clause}
        """
    return [dict(r) for r in client.query(query).result()]


def pack_chunk_to_sqfs(staging_dir: Path, chunk_num: int, num_workers: int = 4) -> Path | None:
    """Pack downloaded images into a per-chunk SquashFS file.

    Uses bucket subdirs (000/, 001/, ...) directly so paths inside the archive
    are clean: 000/abc123.jpg rather than staging_dir/000/abc123.jpg.
    Raises RuntimeError if mksquashfs fails so the SLURM task is marked failed.
    """
    bucket_dirs = sorted(d for d in staging_dir.iterdir() if d.is_dir())
    if not bucket_dirs:
        print(f"  No images in staging dir — skipping sqfs pack for chunk {chunk_num}", flush=True)
        return None

    chunk_sqfs = staging_dir / f"chunk_{chunk_num:04d}.sqfs"
    cmd = [
        "mksquashfs",
        *[str(d) for d in bucket_dirs],
        str(chunk_sqfs),
        "-noappend",
        "-no-xattrs",
        "-comp", "zstd",
        "-Xcompression-level", "3",
        "-processors", str(num_workers),
    ]
    print(f"  Packing {len(bucket_dirs)} bucket dirs → {chunk_sqfs.name}...", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"mksquashfs failed with exit code {result.returncode} for chunk {chunk_num}. "
            f"Staging dir preserved for inspection: {staging_dir}"
        )
    size_mb = chunk_sqfs.stat().st_size / (1024 ** 2)
    print(f"  Packed: {chunk_sqfs.name} ({size_mb:.1f} MB)", flush=True)
    return chunk_sqfs


def clear_staging(staging_dir: Path) -> None:
    """Remove all image files from staging dir, preserving .sqfs chunk files."""
    for f in staging_dir.rglob("*"):
        if f.is_file() and f.suffix != ".sqfs":
            f.unlink()
    for d in sorted(staging_dir.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass


def warn_chunk_accumulation(staging_dir: Path) -> None:
    """Warn if too many chunk sqfs files have built up in staging."""
    count = len(list(staging_dir.glob("chunk_*.sqfs")))
    if count >= _CHUNK_ACCUMULATION_WARN:
        print(
            f"  WARNING: {count} chunk sqfs files in {staging_dir} — "
            f"pack job may be falling behind or a previous run left chunks behind.",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging-dir", required=True,
                        help=(
                            "Local directory where images are downloaded before packing. "
                            "Use scratch (e.g. /scratch/$USER/staging), not home — home has "
                            "a 500k inode quota and each image counts as one inode. "
                            "chunk_NNNN.sqfs files accumulate here until job_bq_pack_per_task.sh "
                            "merges them into the final task_N.sqfs."
                        ))
    parser.add_argument("--num-jobs", type=int, required=True,
                        help=(
                            "Total number of parallel download tasks. Images are partitioned "
                            "by MOD(photo_id, num_jobs) so each task gets a non-overlapping "
                            "subset. Must match the SLURM --array range: --num-jobs 10 requires "
                            "--array=0-9 in the job script. Typical value: 10."
                        ))
    parser.add_argument("--task-id", type=int, required=True,
                        help=(
                            "Index of this task (0 to num_jobs-1). In a SLURM array job set "
                            "this to $SLURM_ARRAY_TASK_ID. This task will download all images "
                            "where MOD(photo_id, num_jobs) == task_id."
                        ))
    parser.add_argument("--num-workers", type=int, default=32,
                        help=(
                            "Number of parallel download threads per task (default: 32). "
                            "With 10 tasks running simultaneously this means up to 320 "
                            "concurrent connections to iNaturalist S3. At scale this caused "
                            "Errno 16 (too many open sockets) — the retry logic handles it "
                            "but reducing to 16-24 workers per task lowers the error rate."
                        ))
    parser.add_argument("--chunk-size", type=int, default=10000,
                        help=(
                            "Number of images to download before packing into a sqfs chunk "
                            "and clearing the staging dir (default: 10000). Lower values "
                            "reduce peak inode usage in staging but produce more chunk files "
                            "for the pack job to merge. Each chunk becomes one "
                            "chunk_NNNN.sqfs file in --staging-dir."
                        ))
    parser.add_argument("--limit", type=int, default=None,
                        help=(
                            "Cap the total number of images queried from BQ. Only for "
                            "small-scale tests — omit for production runs. "
                            "Example: --limit 50 --table-prefix test_ for a quick smoke test."
                        ))
    parser.add_argument("--force-redownload", action="store_true",
                        help=(
                            "Ignore existing records in training_images_downloads and "
                            "re-download all images for this task. Use when staging files "
                            "were deleted after a failed pack job and you need to rebuild "
                            "the chunks from scratch. Without this flag, already-attempted "
                            "images are skipped via LEFT JOIN."
                        ))
    parser.add_argument("--dataset", default=BQ_DEFAULT_DATASET,
                        help=(
                            f"BigQuery dataset name within the leps-ai project "
                            f"(default: {BQ_DEFAULT_DATASET}). "
                            f"Example: --dataset global_all_leps_2605"
                        ))
    parser.add_argument("--table-prefix", default="",
                        help=(
                            "BQ table name prefix for testing without touching production. "
                            "Example: --table-prefix test_ reads from test_training_images "
                            "and writes to test_training_images_downloads. "
                            "Create test tables first with create_test_tables.py."
                        ))
    args = parser.parse_args()

    training_table  = f"{BQ_PROJECT}.{args.dataset}.{args.table_prefix}training_images"
    downloads_table = f"{BQ_PROJECT}.{args.dataset}.{args.table_prefix}training_images_downloads"

    client      = bigquery.Client(project=BQ_PROJECT)
    staging_dir = Path(args.staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== download_images task={args.task_id}/{args.num_jobs} ===", flush=True)
    print(f"training table  : {training_table}", flush=True)
    print(f"downloads table : {downloads_table}", flush=True)
    print(f"staging dir     : {staging_dir}", flush=True)
    print(f"workers         : {args.num_workers}  chunk_size={args.chunk_size}", flush=True)
    print(flush=True)

    ensure_downloads_table(client, downloads_table)
    warn_chunk_accumulation(staging_dir)

    print(f"Querying pending rows (force_redownload={args.force_redownload})...", flush=True)
    rows = get_pending_rows(
        client, training_table, downloads_table,
        args.num_jobs, args.task_id,
        limit=args.limit, force_redownload=args.force_redownload,
    )
    print(f"{len(rows):,} pending images to download", flush=True)

    total_downloaded = total_failed = total_corrupted = 0

    for chunk_start in range(0, len(rows), args.chunk_size):
        chunk        = rows[chunk_start : chunk_start + args.chunk_size]
        chunk_num    = chunk_start // args.chunk_size + 1
        total_chunks = (len(rows) + args.chunk_size - 1) // args.chunk_size
        print(f"\n[Task {args.task_id}] Chunk {chunk_num}/{total_chunks} "
              f"({len(chunk):,} images)...", flush=True)

        # Download in parallel
        results  = []
        t0       = time.perf_counter()
        n_ok = n_fail = n_corrupt = 0

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(download_and_verify, row, staging_dir): row
                       for row in chunk}
            for i, future in enumerate(as_completed(futures)):
                r = future.result()
                results.append(r)
                if r["fetch_status"] == "downloaded":
                    n_ok += 1
                elif r["fetch_status"] == "failed":
                    n_fail += 1
                elif r["fetch_status"] == "corrupted":
                    n_corrupt += 1
                if (i + 1) % 1000 == 0:
                    elapsed = time.perf_counter() - t0
                    print(f"  {i+1:,}/{len(chunk):,}  "
                          f"downloaded={n_ok:,} failed={n_fail:,} corrupted={n_corrupt:,}  "
                          f"({(i+1)/elapsed:.0f} img/s)", flush=True)

        elapsed = time.perf_counter() - t0
        total_downloaded += n_ok
        total_failed     += n_fail
        total_corrupted  += n_corrupt
        print(f"  Chunk done in {elapsed:.0f}s ({len(chunk)/elapsed:.0f} img/s)  "
              f"downloaded={n_ok:,} failed={n_fail:,} corrupted={n_corrupt:,}", flush=True)

        # Append to downloads table (batch load — free tier, parallel-safe).
        # This is the source of truth for progress: get_pending_rows() resumes
        # by LEFT JOINing against it, so status is durable without any DML on
        # training_images mid-run. The MERGE into training_images runs once at
        # the end of the run (see after the loop).
        write_results_to_bq(client, results, downloads_table)
        print(f"  Written to {downloads_table}", flush=True)

        # Pack images into chunk sqfs then clear raw files to keep inode usage low
        pack_chunk_to_sqfs(staging_dir, chunk_num, num_workers=4)
        clear_staging(staging_dir)
        warn_chunk_accumulation(staging_dir)
        print(f"  Staging cleared (chunk sqfs kept)", flush=True)

    # One MERGE per run instead of one per chunk. A MERGE is billed mainly for
    # scanning the target table, so the per-chunk version re-scanned the whole
    # training_images table on every chunk (~N full scans per pass). Sourcing
    # from the already-populated downloads table lets us do it in a single scan.
    print(f"\nMerging download status into {training_table}...", flush=True)
    n_updated = merge_downloads_into_training_images(
        client, training_table, downloads_table
    )
    print(f"  Merged {n_updated:,} rows into {training_table}", flush=True)

    print(f"\n[Task {args.task_id}] Done.  "
          f"downloaded={total_downloaded:,} failed={total_failed:,} "
          f"corrupted={total_corrupted:,}", flush=True)


if __name__ == "__main__":
    main()
