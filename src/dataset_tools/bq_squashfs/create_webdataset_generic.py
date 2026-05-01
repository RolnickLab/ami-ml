#!/usr/bin/env python3
"""
Build a WebDataset from pre-mounted task directories and per-split CSVs.

Assumes images are mounted under a parent directory with one subdirectory per
task, named task_0/ through task_9/.  The task that owns each image is derived
from its photo_id: task_id = photo_id % 10.  relative_local_path in the CSV is
the path of the image *within* its task directory.

Full image path: <images-dir>/task_<photo_id % 10>/<relative_local_path>

This is a generic alternative to create_webdataset.py that does not require
SquashFS, squashfuse, or NVMe scratch space.  It produces identical output tar
shards and is compatible with the same training scripts.

Usage:
    python create_webdataset_from_dir.py \\
        --images-dir    /mnt/images \\
        --split-csvs    train:train.csv val:val.csv test:test.csv \\
        --output-dir    /scratch/global_wds \\
        --images-per-shard 1000
"""

import argparse
import json
import math
import random
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

_t_start = time.perf_counter()


def log(msg: str) -> None:
    elapsed = (time.perf_counter() - _t_start) / 3600
    print(f"[{elapsed:5.2f}h] {msg}", flush=True)


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_split_csvs(
    split_csvs: list[tuple[str, str]],
    taxon_id_column: str,
    images_per_shard: int,
) -> tuple[dict[str, pd.DataFrame], dict[int, int]]:
    dfs: dict[str, pd.DataFrame] = {}
    for name, path in split_csvs:
        df = pd.read_csv(path, dtype={"photo_id": "Int64"})
        missing = {"photo_id", "relative_local_path"} - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")
        assert taxon_id_column in df.columns, (
            f"{path}: required label column '{taxon_id_column}' not found "
            f"(available: {list(df.columns)})"
        )
        df["task_id"] = (df["photo_id"] % 10).astype(int)
        df["row_idx"] = range(len(df))
        df["shard_id"] = df["row_idx"] // images_per_shard
        dfs[name] = df
        log(f"  {name}: {len(df):,} rows  {df['shard_id'].max() + 1} shards  ({path})")

    all_df = pd.concat(dfs.values(), ignore_index=True)
    class_map: dict[int, int] = {}
    if "class_id" not in all_df.columns:
        taxon_ids = sorted(all_df[taxon_id_column].dropna().unique())
        id_map = {int(tid): idx for idx, tid in enumerate(taxon_ids)}
        for split_name, df in dfs.items():
            dfs[split_name]["class_id"] = df[taxon_id_column].map(id_map).astype("Int64")
        log(f"  Assigned class_ids for {len(taxon_ids):,} taxon IDs (column='{taxon_id_column}')")
        class_map = {idx: int(tid) for tid, idx in id_map.items()}
    else:
        class_map = (
            all_df[["class_id", taxon_id_column]]
            .dropna()
            .drop_duplicates("class_id")
            .set_index("class_id")[taxon_id_column]
            .astype(int)
            .to_dict()
        )

    return dfs, class_map


def save_class_map(class_map: dict[int, int], output_dir: Path) -> None:
    dest = output_dir / "class_map.json"
    if dest.exists():
        log(f"class_map.json already exists — skipping ({dest})")
        return
    with open(dest, "w") as f:
        json.dump({str(k): v for k, v in sorted(class_map.items())}, f, indent=2)
    log(f"Saved class_map.json ({len(class_map):,} classes → {dest})")


# ── Build lookup ──────────────────────────────────────────────────────────────

def build_lookup(
    dfs: dict[str, pd.DataFrame],
    meta_columns: list[str],
) -> dict[str, dict]:
    """Build {relative_local_path: {split, shard_id, task_id, class_id, meta}}."""
    lookup: dict[str, dict] = {}
    for split, df in dfs.items():
        for row in df.itertuples(index=False):
            meta = {
                col: getattr(row, col)
                for col in meta_columns
                if hasattr(row, col) and not pd.isna(getattr(row, col))
            }
            lookup[row.relative_local_path] = {
                "split":    split,
                "shard_id": int(row.shard_id),
                "task_id":  int(row.task_id),
                "class_id": int(row.class_id),
                "meta":     meta,
            }
    log(f"Lookup built: {len(lookup):,} entries across {len(dfs)} splits")
    return lookup


# ── Walk image dirs and collect per-shard path lists ─────────────────────────

def collect_shards(
    images_dir: Path,
    lookup: dict[str, dict],
) -> tuple[dict[tuple[str, int], list[tuple[Path, str, dict]]], int]:
    """
    Walk task_0/ … task_9/ under images_dir.
    For each image found in the lookup, append (abs_path, rel_path, entry)
    to the appropriate (split, shard_id) bucket.

    Returns (shard_buckets, n_missing).
    """
    shard_buckets: dict[tuple[str, int], list] = defaultdict(list)
    found = missing = skipped = 0
    t0 = time.perf_counter()

    for task_dir in sorted(images_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue

        log(f"  Walking {task_dir.name} ...")
        for img_path in task_dir.rglob("*"):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            rel = str(img_path.relative_to(task_dir))
            entry = lookup.get(rel)
            if entry is None:
                skipped += 1
                continue

            key = (entry["split"], entry["shard_id"])
            shard_buckets[key].append((img_path, rel, entry))
            found += 1

            if found % 500_000 == 0:
                elapsed = time.perf_counter() - t0
                log(f"    {found:,} found  {skipped:,} skipped  {found/elapsed:.0f} img/s")

    missing = len(lookup) - found
    elapsed = time.perf_counter() - t0
    log(f"Walk complete: {found:,} found  {skipped:,} not-in-csv  "
        f"{missing:,} in-csv-not-found  {elapsed:.0f}s")
    return dict(shard_buckets), missing


# ── Pack shards to tar ────────────────────────────────────────────────────────

def pack_shards(
    shard_buckets: dict[tuple[str, int], list],
    output_dir: Path,
    splits: list[str],
    pack_workers: int,
    seed: int,
) -> None:
    rng = random.Random(seed)

    for split in splits:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

    items = list(shard_buckets.items())
    log(f"Packing {len(items):,} shards ({pack_workers} workers) ...")
    t0 = time.perf_counter()
    total_bytes = total_images = 0

    def pack_one(item: tuple) -> tuple[int, int]:
        (split, shard_id), entries = item
        rng.shuffle(entries)
        tar_path = output_dir / split / f"{split}-{shard_id:06d}.tar"
        nb = ni = 0
        with tarfile.open(tar_path, "w") as tf:
            for img_path, rel, entry in entries:
                import hashlib
                key = hashlib.md5(f"{entry['task_id']}:{rel}".encode()).hexdigest()
                ext = img_path.suffix.lower()

                img_bytes = img_path.read_bytes()
                cls_bytes = str(entry["class_id"]).encode()
                meta_bytes = json.dumps({
                    "class_id": entry["class_id"],
                    "relative_local_path": rel,
                    "task_id": entry["task_id"],
                    **entry["meta"],
                }).encode()

                for suffix, data in [(ext, img_bytes), (".cls", cls_bytes), (".json", meta_bytes)]:
                    import io
                    buf = io.BytesIO(data)
                    info = tarfile.TarInfo(name=f"{key}{suffix}")
                    info.size = len(data)
                    tf.addfile(info, buf)

                nb += len(img_bytes)
                ni += 1
        return nb, ni

    with ThreadPoolExecutor(max_workers=pack_workers) as pool:
        for nb, ni in pool.map(pack_one, items):
            total_bytes += nb
            total_images += ni

    elapsed = time.perf_counter() - t0
    log(f"Packed {total_images:,} images  {total_bytes/1024**3:.1f} GB  "
        f"{elapsed:.0f}s  ({total_bytes/1024**2/elapsed:.0f} MB/s)")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_split_csvs(values: list[str]) -> list[tuple[str, str]]:
    return [(v.split(":", 1)[0].strip(), v.split(":", 1)[1].strip()) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--images-dir",      required=True,
                        help="Parent dir containing task_0/ … task_9/ subdirectories")
    parser.add_argument("--split-csvs",      nargs="+", required=True,
                        help="Per-split CSV files as 'name:path' (e.g. train:train.csv)")
    parser.add_argument("--output-dir",      required=True,
                        help="Output directory; split subdirs and class_map.json written here")
    parser.add_argument("--images-per-shard", type=int, default=1000,
                        help="Images per shard, assigned by CSV row order (default: 1000)")
    parser.add_argument("--taxon-id-column", default="inat_taxon_id",
                        help="CSV column used as label (default: inat_taxon_id)")
    parser.add_argument("--pack-workers",    type=int, default=16,
                        help="Parallel workers for tar packing (default: 16)")
    parser.add_argument("--seed",            type=int, default=42,
                        help="Seed for within-shard shuffle (default: 42)")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_csvs = parse_split_csvs(args.split_csvs)
    splits = [name for name, _ in split_csvs]

    log("=== create_webdataset_from_dir ===")
    log(f"Images dir : {images_dir}")
    log(f"Splits     : {splits}")
    log(f"Output     : {output_dir}")
    print()

    log("Loading split CSVs ...")
    dfs, class_map = load_split_csvs(split_csvs, args.taxon_id_column, args.images_per_shard)
    save_class_map(class_map, output_dir)
    print()

    structural = {"photo_id", "relative_local_path", "class_id", "task_id", "shard_id", "row_idx"}
    all_cols = set().union(*(df.columns for df in dfs.values()))
    meta_columns = [c for c in all_cols if c not in structural]

    log("Building image lookup ...")
    lookup = build_lookup(dfs, meta_columns)
    del dfs
    print()

    log("Collecting images from task directories ...")
    shard_buckets, n_missing = collect_shards(images_dir, lookup)
    if n_missing > 0:
        log(f"WARNING: {n_missing:,} images are in the CSVs but were not found on disk")
    print()

    log("Packing shards → output dir ...")
    pack_shards(shard_buckets, output_dir, splits, args.pack_workers, args.seed)

    elapsed = (time.perf_counter() - _t_start) / 3600
    log(f"\nDone in {elapsed:.2f}h")


if __name__ == "__main__":
    main()
