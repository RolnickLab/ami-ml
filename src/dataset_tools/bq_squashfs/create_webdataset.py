#!/usr/bin/env python3
"""
Build a WebDataset from per-split CSVs and SquashFS files.

Workflow:
  1. Load per-split CSVs (produced by split_csv.py)
  2. Assign class IDs alphabetically from species_name if class_id absent
  3. Assign each image to a shard within its split
     — hash-based (--n-shards) or CSV-order-based (--images-per-shard)
  4. For each sqfs: copy to NVMe → mount → scatter images to split/shard dirs
  5. Pack each split's shard dirs → Lustre tar files (shuffled within each shard)

CSV required columns (same in all split files):
  photo_id             — integer; task_id = photo_id % 10 (which sqfs)
  relative_local_path  — path of the image within the sqfs

CSV class column (one of):
  class_id             — integer; used directly if present
  species_name         — used to assign class_id alphabetically if class_id absent

Any other CSV columns are passed through into per-sample .json metadata.

Typical usage — two batches for global WDS (stays under 7 TB NVMe peak):

  Batch 1 — sqfs 0–4, create new tars:
    python create_webdataset.py \\
        --split-csvs train:train.csv val:val.csv test:test.csv \\
        --sqfs-paths task_0.sqfs ... task_4.sqfs \\
        --sqfs-start-idx 0 \\
        --n-shards 10700 \\
        --nvme-dir $SLURM_TMPDIR \\
        --output-dir /scratch/melabbas/global_wds \\
        --tar-mode w

  Batch 2 — sqfs 5–9, append to existing tars:
    python create_webdataset.py \\
        --split-csvs train:train.csv val:val.csv test:test.csv \\
        --sqfs-paths task_5.sqfs ... task_9.sqfs \\
        --sqfs-start-idx 5 \\
        --n-shards 10700 \\
        --nvme-dir $SLURM_TMPDIR \\
        --output-dir /scratch/melabbas/global_wds \\
        --tar-mode a

Shard assignment is deterministic (hash of rel_path + seed) — the same image
always maps to the same shard, so --tar-mode a is safe to retry.
"""

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

_t_start = time.perf_counter()


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    elapsed = (time.perf_counter() - _t_start) / 3600
    print(f"[{elapsed:5.2f}h] {msg}", flush=True)


# ── Disk helpers ──────────────────────────────────────────────────────────────

def disk_free_tb(path: str) -> float:
    r = subprocess.run(["df", "--output=avail", "-k", path],
                       capture_output=True, text=True)
    return int(r.stdout.strip().split()[-1]) / 1024**3


# ── squashfuse helpers ────────────────────────────────────────────────────────

def sqfs_mount(sqfs_path: Path, mnt_dir: Path) -> None:
    r = subprocess.run(["squashfuse", str(sqfs_path), str(mnt_dir)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"squashfuse failed for {sqfs_path}: {r.stderr.strip()}")


def sqfs_unmount(mnt_dir: Path) -> None:
    subprocess.run(["fusermount", "-u", str(mnt_dir)], capture_output=True)


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_split_csvs(
    split_csvs: list[tuple[str, str]],
    taxon_id_column: str = "inat_taxon_id",
) -> tuple[dict[str, pd.DataFrame], dict[int, int]]:
    """Load per-split CSVs, assign class_ids if absent, return (dfs, class_map).

    If 'class_id' is absent, taxon_id_column is required and must be present in
    every split CSV. Taxon IDs are sorted numerically and mapped to sequential
    class IDs. class_map.json stores {sequential_id: taxon_id}.
    """
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
        df["row_idx"] = range(len(df))   # position in this split's CSV (used by --images-per-shard)
        dfs[name] = df
        log(f"  {name}: {len(df):,} rows from {path}")

    # Assign class_ids across all splits together so IDs are consistent
    all_df = pd.concat(dfs.values(), ignore_index=True)
    class_map: dict[int, int] = {}
    if "class_id" not in all_df.columns:
        taxon_ids = sorted(all_df[taxon_id_column].dropna().unique())
        id_map    = {int(tid): idx for idx, tid in enumerate(taxon_ids)}
        for split_name, df in dfs.items():
            dfs[split_name]["class_id"] = df[taxon_id_column].map(id_map).astype("Int64")
        log(f"  Assigned class_ids for {len(taxon_ids):,} taxon IDs "
            f"(column='{taxon_id_column}', sorted numerically)")
        class_map = {idx: int(tid) for tid, idx in id_map.items()}
    else:
        # class_id already in CSV — build reverse map from taxon_id
        class_map = (
            all_df[["class_id", taxon_id_column]]
            .dropna()
            .drop_duplicates("class_id")
            .set_index("class_id")[taxon_id_column]
            .astype(int)
            .to_dict()
        )

    return dfs, class_map


def save_class_map(class_map: dict[int, str], output_dir: Path) -> None:
    dest = output_dir / "class_map.json"
    if dest.exists():
        log(f"class_map.json already exists — skipping ({dest})")
        return
    with open(dest, "w") as f:
        json.dump({str(k): v for k, v in sorted(class_map.items())}, f, indent=2)
    log(f"Saved class_map.json ({len(class_map):,} classes → {dest})")


# ── Shard assignment ──────────────────────────────────────────────────────────



# ── Build per-task lookup ─────────────────────────────────────────────────────

def build_lookup(
    dfs: dict[str, pd.DataFrame],
    task_ids: list[int],
    meta_columns: list[str],
    images_per_shard: int,
) -> dict[int, dict[str, dict]]:
    """Build {task_id: {rel_path: {split, shard_id, class_id, meta}}}.

    Shard assignment: CSV-order — row_idx // images_per_shard.
    First N rows → shard 0, next N → shard 1, etc.
    """
    lookup: dict[int, dict[str, dict]] = {tid: {} for tid in task_ids}

    for split, df in dfs.items():
        subset = df[df["task_id"].isin(task_ids)]
        for row in subset.itertuples(index=False):
            rel = row.relative_local_path
            tid = row.task_id
            meta = {
                col: getattr(row, col)
                for col in meta_columns
                if hasattr(row, col) and not pd.isna(getattr(row, col))
            }
            lookup[tid][rel] = {
                "split":    split,
                "shard_id": int(row.row_idx) // images_per_shard,
                "class_id": int(row.class_id),
                "meta":     meta,
            }

    for tid in task_ids:
        log(f"  task_{tid}: {len(lookup[tid]):,} images in lookup")
    return lookup


# ── Scatter ───────────────────────────────────────────────────────────────────

def scatter_sqfs(
    sqfs_path: Path,
    sqfs_idx: int,
    nvme_dir: Path,
    nvme_mnt: Path,
    nvme_shards: Path,
    task_lookup: dict[str, dict],
    no_nvme_copy: bool = False,
    limit: int = 0,
) -> tuple[int, int]:
    """Copy sqfs to NVMe, mount, scatter images + labels to split/shard dirs.

    --no-nvme-copy: mount sqfs directly from source (skips copy; for testing).
    --limit N:      stop after N images per sqfs (for smoke tests).
    """
    if no_nvme_copy:
        mount_target = sqfs_path
        log(f"  [{sqfs_path.name}] mounting directly (--no-nvme-copy)")
    else:
        mount_target = nvme_dir / sqfs_path.name
        log(f"  [{sqfs_path.name}] copying to NVMe  "
            f"(NVMe free: {disk_free_tb(str(nvme_dir)):.2f} TB)")
        t0 = time.perf_counter()
        shutil.copy2(str(sqfs_path), str(mount_target))
        gb      = mount_target.stat().st_size / 1024**3
        elapsed = time.perf_counter() - t0
        log(f"  [{sqfs_path.name}] copied {gb:.1f} GB in {elapsed:.0f}s  "
            f"({gb * 1024 / elapsed:.0f} MB/s)")

    sqfs_mount(mount_target, nvme_mnt)
    try:
        if limit:
            # Don't materialise the full list — stop enumerating once limit is hit
            all_paths = (
                p for p in nvme_mnt.rglob("*")
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )
        else:
            # Sort for sequential reads → squashfuse block cache reuse
            all_paths = sorted(
                p for p in nvme_mnt.rglob("*")
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )
        cap = f"  (capped at {limit})" if limit else ""
        log(f"  [{sqfs_path.name}] scattering{cap}  "
            f"(lookup: {len(task_lookup):,} entries)...")

        written = skipped = 0
        total_bytes = 0
        t0 = time.perf_counter()

        for img_path in all_paths:
            if limit and written >= limit:
                break

            rel   = str(img_path.relative_to(nvme_mnt))
            entry = task_lookup.get(rel)
            if entry is None:
                skipped += 1
                continue

            split     = entry["split"]
            shard_id  = entry["shard_id"]
            class_id  = entry["class_id"]
            meta      = entry["meta"]
            key       = hashlib.md5(f"{sqfs_idx}:{rel}".encode()).hexdigest()
            shard_dir = nvme_shards / split / f"shard_{shard_id:06d}"
            ext       = img_path.suffix.lower()

            img_bytes = img_path.read_bytes()
            (shard_dir / f"{key}{ext}").write_bytes(img_bytes)
            (shard_dir / f"{key}.cls").write_bytes(str(class_id).encode())
            (shard_dir / f"{key}.json").write_bytes(
                json.dumps({
                    "class_id": class_id,
                    "relative_local_path": rel,
                    "sqfs_idx": sqfs_idx,
                    **meta,
                }).encode()
            )

            total_bytes += len(img_bytes)
            written += 1
            if written % 100_000 == 0:
                elapsed = time.perf_counter() - t0
                log(f"    {written:,}/{len(all_paths):,}  "
                    f"{total_bytes/1024**3:.1f} GB  {written/elapsed:.0f} img/s")

        elapsed = time.perf_counter() - t0
        log(f"  [{sqfs_path.name}] scattered {written:,} in {elapsed:.0f}s  "
            f"({written/elapsed:.0f} img/s)  skipped={skipped:,}")

        if not limit:
            missing = len(task_lookup) - written
            if missing > 0:
                log(f"  WARNING: {missing:,} images are in the CSV but were not found in "
                    f"{sqfs_path.name} — they will be missing from the dataset")
    finally:
        sqfs_unmount(nvme_mnt)

    if not no_nvme_copy:
        mount_target.unlink()
        log(f"  [{sqfs_path.name}] deleted  NVMe free: {disk_free_tb(str(nvme_dir)):.2f} TB")
    return written, skipped


# ── Pack ──────────────────────────────────────────────────────────────────────

def pack_split(
    split: str,
    nvme_shards: Path,
    lustre_split_dir: Path,
    n_shards: int,
    tar_mode: str,
    pack_workers: int,
    seed: int,
) -> int:
    """Pack one split's NVMe shard dirs → Lustre tar files. Returns total bytes."""
    split_shards = nvme_shards / split
    non_empty = [
        split_shards / f"shard_{s:06d}"
        for s in range(n_shards)
        if (split_shards / f"shard_{s:06d}").exists()
        and any((split_shards / f"shard_{s:06d}").iterdir())
    ]
    log(f"  [{split}] packing {len(non_empty):,} non-empty shards  "
        f"(mode='{tar_mode}', {pack_workers} workers)")

    rng = random.Random(seed)

    def pack_one(shard_dir: Path) -> int:
        shard_id = int(shard_dir.name.split("_")[1])
        tar_path = lustre_split_dir / f"{split}-{shard_id:06d}.tar"
        files    = list(shard_dir.iterdir())
        if not files:
            return 0

        existing_keys: set[str] = set()
        if tar_mode == "a" and tar_path.exists():
            with tarfile.open(tar_path, "r") as tf:
                existing_keys = {m.name for m in tf.getmembers()}

        # Group into per-sample triplets (.cls .jpg .json), then shuffle samples
        key_groups: dict[str, list[Path]] = defaultdict(list)
        for p in files:
            key_groups[p.stem].append(p)
        keys = list(key_groups)
        rng.shuffle(keys)

        total = 0
        with tarfile.open(tar_path, tar_mode) as tf:
            for key in keys:
                for p in sorted(key_groups[key]):   # consistent order within sample
                    if p.name not in existing_keys:
                        tf.add(p, arcname=p.name)
                        total += p.stat().st_size
                    p.unlink()
        return total

    t0 = time.perf_counter()
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=pack_workers) as pool:
        for nb in pool.map(pack_one, non_empty, chunksize=32):
            total_bytes += nb
    elapsed = time.perf_counter() - t0
    log(f"  [{split}] packed {total_bytes/1024**3:.1f} GB in {elapsed:.0f}s  "
        f"({total_bytes/1024**2/elapsed:.0f} MB/s)")
    return total_bytes


def pack_to_lustre(
    splits: list[str],
    shards_per_split: dict[str, int],
    nvme_shards: Path,
    output_dir: Path,
    tar_mode: str,
    pack_workers: int,
    seed: int,
) -> None:
    for split in splits:
        lustre_split_dir = output_dir / split
        lustre_split_dir.mkdir(parents=True, exist_ok=True)
        pack_split(split, nvme_shards, lustre_split_dir,
                   shards_per_split[split], tar_mode, pack_workers, seed)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_split_csvs(values: list[str]) -> list[tuple[str, str]]:
    """Parse ['train:train.csv', 'val:val.csv', ...] → [('train', 'train.csv'), ...]"""
    result = []
    for v in values:
        name, path = v.split(":", 1)
        result.append((name.strip(), path.strip()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-csvs",     nargs="+", required=True,
                        help="Per-split CSV files as 'name:path' (e.g. train:train.csv)")
    parser.add_argument("--sqfs-paths",     nargs="+", required=True,
                        help="Ordered sqfs paths for this batch")
    parser.add_argument("--sqfs-start-idx", type=int, required=True,
                        help="Global index of first sqfs (task_id = photo_id %% 10)")
    parser.add_argument("--images-per-shard", type=int, default=1000,
                        help="Images per shard (default: 1000). Row 0–(N-1) in the CSV → shard 0, "
                             "rows N–(2N-1) → shard 1, etc.")
    parser.add_argument("--nvme-dir",       required=True,
                        help="NVMe scratch root ($SLURM_TMPDIR)")
    parser.add_argument("--output-dir",     required=True,
                        help="Lustre output dir; split subdirs and class_map.json written here")
    parser.add_argument("--pack-workers",   type=int, default=16,
                        help="Parallel workers for tar packing")
    parser.add_argument("--tar-mode",       choices=["w", "a"], default="w",
                        help="'w' create new tars, 'a' append (idempotent on retry)")
    parser.add_argument("--taxon-id-column", default="inat_taxon_id",
                        help="CSV column to use as label (default: inat_taxon_id). "
                             "Taxon IDs are sorted numerically and mapped to sequential "
                             "class IDs. Ignored if 'class_id' is already in the CSV.")
    parser.add_argument("--seed",           type=int, default=42,
                        help="Seed for shard assignment and within-shard shuffle")
    parser.add_argument("--no-nvme-copy",   action="store_true",
                        help="Mount sqfs directly from source, skip copy to NVMe. "
                             "Use for smoke-testing scatter/pack logic without needing "
                             "full NVMe space.")
    parser.add_argument("--limit",          type=int, default=0,
                        help="Stop scatter after N images per sqfs (0 = no limit). "
                             "Use with --no-nvme-copy for a fast end-to-end smoke test.")
    args = parser.parse_args()


    nvme_dir    = Path(args.nvme_dir)
    nvme_mnt    = nvme_dir / "mnt"
    nvme_shards = nvme_dir / "shards"
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_csvs = parse_split_csvs(args.split_csvs)
    task_ids   = list(range(args.sqfs_start_idx,
                            args.sqfs_start_idx + len(args.sqfs_paths)))

    log("=== create_webdataset ===")
    log(f"Splits:  {[n for n,_ in split_csvs]}")
    log(f"Tasks:   {task_ids}  sqfs: {len(args.sqfs_paths)}")
    log(f"Output:  {output_dir}")
    if args.no_nvme_copy:
        log("Mode:    --no-nvme-copy (smoke test — sqfs mounted in place)")
    if args.limit:
        log(f"Mode:    --limit {args.limit} images per sqfs (smoke test)")
    print()

    # ── NVMe preflight ────────────────────────────────────────────────────────
    # Peak NVMe usage occurs during scatter of the last sqfs in the batch:
    #   peak = batch_size × avg_scatter_per_sqfs + largest_sqfs_file
    # scatter_per_sqfs ≈ sqfs_size (images decompressed into shard dirs, ~3 files
    # per image but .cls/.json are tiny vs JPEG, so ≈ sqfs size)
    # This matches the analysis in job_build_wds_global.sh:
    #   N=4 (5th sqfs): 5×1.07 + 1.1 = 6.45 TB  (safe under 7 TB with BATCH_SIZE=5)
    if not args.no_nvme_copy:
        sqfs_sizes_gb = [Path(p).stat().st_size / 1024**3 for p in args.sqfs_paths]
        n             = len(sqfs_sizes_gb)
        avg_scatter   = sum(sqfs_sizes_gb) / n
        peak_gb       = n * avg_scatter + max(sqfs_sizes_gb)
        nvme_free_gb  = disk_free_tb(str(nvme_dir)) * 1024
        log(f"NVMe preflight: {n} sqfs  avg={avg_scatter:.1f} GB  "
            f"peak estimate={peak_gb:.0f} GB  ({n}×{avg_scatter:.1f} + {max(sqfs_sizes_gb):.1f})  "
            f"NVMe free={nvme_free_gb:.0f} GB")
        if peak_gb > nvme_free_gb * 0.9:
            log(f"WARNING: estimated peak ({peak_gb:.0f} GB) exceeds 90% of NVMe "
                f"({nvme_free_gb:.0f} GB) — reduce --batch-size or request more --tmp")
        print()

    # ── Load CSVs ─────────────────────────────────────────────────────────────
    log("Loading split CSVs ...")
    dfs, class_map = load_split_csvs(split_csvs, taxon_id_column=args.taxon_id_column)
    save_class_map(class_map, output_dir)

    shards_per_split = {
        name: math.ceil(len(df) / args.images_per_shard)
        for name, df in dfs.items()
    }
    log(f"Shards per split (--images-per-shard={args.images_per_shard}): {shards_per_split}")

    structural   = {"photo_id", "relative_local_path", "class_id", "task_id", "split", "row_idx"}
    all_cols     = set().union(*(df.columns for df in dfs.values()))
    meta_columns = [c for c in all_cols if c not in structural]

    # ── Build per-task lookup ──────────────────────────────────────────────────
    log(f"Building image lookup for tasks {task_ids} ...")
    lookup = build_lookup(dfs, task_ids, meta_columns, args.images_per_shard)
    del dfs  # free RAM
    print()

    # ── Pre-create shard dirs ─────────────────────────────────────────────────
    log("Pre-creating shard dirs on NVMe ...")
    nvme_mnt.mkdir(exist_ok=True)
    for split, n in shards_per_split.items():
        for s in range(n):
            (nvme_shards / split / f"shard_{s:06d}").mkdir(parents=True, exist_ok=True)
    log(f"  Done ({sum(shards_per_split.values())} dirs across {len(shards_per_split)} splits)")
    print()

    # ── Scatter phase ─────────────────────────────────────────────────────────
    total_written = total_skipped = 0
    for i, sqfs_path_str in enumerate(args.sqfs_paths):
        sqfs_path = Path(sqfs_path_str)
        sqfs_idx  = args.sqfs_start_idx + i
        task_lkp  = lookup.get(sqfs_idx, {})
        log(f"--- Scatter sqfs {sqfs_idx} ({sqfs_path.name})  "
            f"lookup: {len(task_lkp):,} entries ---")
        w, s = scatter_sqfs(sqfs_path, sqfs_idx, nvme_dir, nvme_mnt,
                             nvme_shards, task_lkp,
                             no_nvme_copy=args.no_nvme_copy, limit=args.limit)
        total_written += w
        total_skipped += s
        print()

    if total_skipped:
        log(f"WARNING: {total_skipped:,} images skipped (in sqfs but not in CSVs)")

    # ── Pack phase ────────────────────────────────────────────────────────────
    log(f"--- Pack → Lustre (mode='{args.tar_mode}') ---")
    pack_to_lustre(
        [n for n, _ in split_csvs], shards_per_split,
        nvme_shards, output_dir, args.tar_mode, args.pack_workers, args.seed,
    )

    elapsed = (time.perf_counter() - _t_start) / 3600
    log(f"\nBatch done: {total_written:,} images written  "
        f"{total_skipped:,} skipped  {elapsed:.2f}h elapsed")


if __name__ == "__main__":
    main()
