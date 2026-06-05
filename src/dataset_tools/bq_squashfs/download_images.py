#!/usr/bin/env python3
"""
Download images from training_images BQ table to a local staging directory,
record results in training_images_downloads BQ table, then pack into SquashFS.

Images are split across parallel jobs using MOD(photo_id, num_jobs) = task_id
so each job handles a balanced, non-overlapping subset of images.

Resumable: already-attempted images are skipped by LEFT JOINing with
training_images_downloads. Re-running the same task_id is safe.

Usage (single job):
    python download_images.py \
        --staging-dir /localscratch/$USER/staging \
        --num-jobs 1 \
        --task-id 0

Usage (one task in a SLURM array):
    python download_images.py \
        --staging-dir /localscratch/$USER/staging \
        --num-jobs 10 \
        --task-id $SLURM_ARRAY_TASK_ID

After all array tasks finish, run job_bq_pack_squashfs.sh to merge
all staging directories into a single SquashFS archive.
"""

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import PIL
import requests
from PIL import Image
from google.cloud import bigquery

Image.MAX_IMAGE_PIXELS = None

BQ_PROJECT = "leps-ai"
BQ_DATASET = "global_butterflies_2604"
TRAINING_TABLE = f"{BQ_PROJECT}.{BQ_DATASET}.training_images"
DOWNLOADS_TABLE = f"{BQ_PROJECT}.{BQ_DATASET}.training_images_downloads"

DOWNLOADS_SCHEMA = [
    bigquery.SchemaField("dataset_source_uuid", "STRING"),
    bigquery.SchemaField("fetch_status", "STRING"),
    bigquery.SchemaField("image_width", "INTEGER"),
    bigquery.SchemaField("image_height", "INTEGER"),
    bigquery.SchemaField("image_size", "INTEGER"),
    bigquery.SchemaField("corrupted", "BOOLEAN"),
]


def download_and_verify(row: dict, staging_dir: Path) -> dict:
    """Download one image from absolute_url, verify with PIL, return result."""
    url = row["absolute_url"]
    rel_path = row["relative_local_path"]
    dest = staging_dir / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "dataset_source_uuid": row["dataset_source_uuid"],
        "fetch_status": None,
        "image_width": None,
        "image_height": None,
        "image_size": None,
        "corrupted": None,
    }

    # Download
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"Failed {url}: {e}", flush=True)
        result["fetch_status"] = "failed"
        return result

    # Verify with PIL
    try:
        with Image.open(dest) as img:
            img.convert("RGB")
            result["image_width"], result["image_height"] = img.size
        result["image_size"] = dest.stat().st_size
        result["corrupted"] = False
        result["fetch_status"] = "downloaded"
    except (PIL.UnidentifiedImageError, OSError) as e:
        print(f"Corrupted {url}: {e}", flush=True)
        result["corrupted"] = True
        result["fetch_status"] = "corrupted"

    return result


def ensure_downloads_table(client: bigquery.Client) -> None:
    """Create training_images_downloads table if it doesn't exist."""
    try:
        client.get_table(DOWNLOADS_TABLE)
    except Exception:
        table = bigquery.Table(DOWNLOADS_TABLE, schema=DOWNLOADS_SCHEMA)
        table.description = (
            "Download results for training_images. One row per download attempt. "
            "Appended to by parallel download jobs. Used to track fetch progress "
            "without DML updates on the base training_images table."
        )
        client.create_table(table)
        print(f"Created table {DOWNLOADS_TABLE}", flush=True)


def write_results_to_bq(client: bigquery.Client, results: list[dict]) -> None:
    """
    Append download results to training_images_downloads via batch load job.
    Uses load_table_from_dataframe which does not require DML billing.
    Multiple parallel jobs can safely append to the same table simultaneously.
    """
    df = pd.DataFrame(results)
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=DOWNLOADS_SCHEMA,
    )
    job = client.load_table_from_dataframe(df, DOWNLOADS_TABLE, job_config=job_config)
    job.result()


def get_pending_rows(
    client: bigquery.Client, num_jobs: int, task_id: int,
    limit: int | None = None, force_redownload: bool = False
) -> list[dict]:
    """
    Query training_images for rows assigned to this task (MOD split),
    excluding images already attempted in training_images_downloads.
    Pass force_redownload=True to ignore existing download records (e.g. to
    re-download images whose staging files were deleted).
    """
    limit_clause = f"LIMIT {limit}" if limit else ""
    if force_redownload:
        query = f"""
        SELECT
            ti.dataset_source_uuid,
            ti.absolute_url,
            ti.relative_local_path
        FROM `{TRAINING_TABLE}` ti
        WHERE ti.fetch_status = 'pending'
          AND MOD(ti.photo_id, {num_jobs}) = {task_id}
        {limit_clause}
        """
    else:
        query = f"""
        SELECT
            ti.dataset_source_uuid,
            ti.absolute_url,
            ti.relative_local_path
        FROM `{TRAINING_TABLE}` ti
        LEFT JOIN `{DOWNLOADS_TABLE}` d
            ON ti.dataset_source_uuid = d.dataset_source_uuid
        WHERE ti.fetch_status = 'pending'
          AND MOD(ti.photo_id, {num_jobs}) = {task_id}
          AND d.dataset_source_uuid IS NULL
        {limit_clause}
        """
    rows = list(client.query(query).result())
    return [dict(r) for r in rows]


def pack_chunk_to_sqfs(staging_dir: Path, chunk_num: int, num_workers: int = 4) -> Path | None:
    """Pack downloaded images in staging_dir into a per-chunk SquashFS file.

    Passes bucket dirs (000/, 001/, ...) directly to mksquashfs so paths inside
    the archive are clean: 000/abc123.jpg — not staging_dir/000/abc123.jpg.

    Returns the path to the created .sqfs file, or None if staging_dir is empty.
    """
    bucket_dirs = sorted(d for d in staging_dir.iterdir() if d.is_dir())
    if not bucket_dirs:
        print(f"  No images in staging dir, skipping sqfs pack for chunk {chunk_num}", flush=True)
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
    subprocess.run(cmd, check=True)
    size_mb = chunk_sqfs.stat().st_size / (1024 ** 2)
    print(f"  Packed: {chunk_sqfs.name} ({size_mb:.1f} MB)", flush=True)
    return chunk_sqfs


def clear_staging(staging_dir: Path) -> None:
    """Remove all image files from staging dir; preserve .sqfs chunk files."""
    for f in staging_dir.rglob("*"):
        if f.is_file() and f.suffix != ".sqfs":
            f.unlink()
    for d in sorted(staging_dir.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()  # only removes empty dirs; bucket dirs with no images will be gone
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-dir",  required=True,
                        help="Local directory to download images into")
    parser.add_argument("--num-jobs",     type=int, required=True,
                        help="Total number of parallel jobs (used for MOD split)")
    parser.add_argument("--task-id",      type=int, required=True,
                        help="This job's task ID (0 to num_jobs-1)")
    parser.add_argument("--num-workers",  type=int, default=32,
                        help="Parallel download workers")
    parser.add_argument("--chunk-size",   type=int, default=10000,
                        help="Images per chunk before writing to BQ")
    parser.add_argument("--limit",        type=int, default=None,
                        help="Cap total images queried (for small-scale tests)")
    parser.add_argument("--force-redownload", action="store_true",
                        help="Re-download all images for this task, ignoring existing BQ records")
    args = parser.parse_args()

    client = bigquery.Client(project=BQ_PROJECT)
    staging_dir = Path(args.staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    ensure_downloads_table(client)

    print(f"Task {args.task_id}/{args.num_jobs}: querying pending rows "
          f"(force_redownload={args.force_redownload})...", flush=True)
    rows = get_pending_rows(client, args.num_jobs, args.task_id,
                            limit=args.limit, force_redownload=args.force_redownload)
    print(f"Task {args.task_id}/{args.num_jobs}: {len(rows):,} pending images", flush=True)

    total_downloaded = 0
    total_failed = 0
    total_corrupted = 0

    for chunk_start in range(0, len(rows), args.chunk_size):
        chunk = rows[chunk_start : chunk_start + args.chunk_size]
        chunk_num = chunk_start // args.chunk_size + 1
        total_chunks = (len(rows) + args.chunk_size - 1) // args.chunk_size
        print(f"\n[Task {args.task_id}] Chunk {chunk_num}/{total_chunks} "
              f"({len(chunk)} images)...", flush=True)

        # Download in parallel
        results = []
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(download_and_verify, row, staging_dir): row
                for row in chunk
            }
            for i, future in enumerate(as_completed(futures)):
                results.append(future.result())
                if (i + 1) % 1000 == 0:
                    print(f"  {i+1}/{len(chunk)} done", flush=True)

        # Count results
        for r in results:
            if r["fetch_status"] == "downloaded":
                total_downloaded += 1
            elif r["fetch_status"] == "failed":
                total_failed += 1
            elif r["fetch_status"] == "corrupted":
                total_corrupted += 1

        print(f"  downloaded={total_downloaded} failed={total_failed} "
              f"corrupted={total_corrupted}", flush=True)

        # Write results to BQ (free batch load, no DML)
        write_results_to_bq(client, results)
        print(f"  Results written to BQ", flush=True)

        # Pack images into a per-chunk sqfs, then delete raw files.
        # This keeps peak inode usage at ~chunk_size per task (well under quota)
        # rather than accumulating all images on disk until the pack job runs.
        pack_chunk_to_sqfs(staging_dir, chunk_num, num_workers=4)
        clear_staging(staging_dir)
        print(f"  Staging cleared (chunk sqfs kept)", flush=True)

    print(f"\n[Task {args.task_id}] Done. "
          f"downloaded={total_downloaded} failed={total_failed} "
          f"corrupted={total_corrupted}", flush=True)


if __name__ == "__main__":
    main()
