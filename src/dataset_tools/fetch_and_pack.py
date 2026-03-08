#!/usr/bin/env python
# coding: utf-8

"""
Sharded fetch-and-pack pipeline for large datasets.

Downloads images in chunks, packs them into webdataset shards, then deletes raw
images before fetching the next chunk. At any point only one chunk of raw images
(~chunk_size files) exists on disk alongside the growing set of tars.

Designed to replace the separate fetch + webdataset steps for datasets where
the combined raw image count would exceed cluster file quota.
"""

import glob
import json
import os
import shutil
from functools import partial
from multiprocessing import Pool
from typing import Optional

import pandas as pd
import requests
import webdataset as wds

from src.dataset_tools.create_webdataset import (
    _create_samples,
    _get_category_map,
)
from src.dataset_tools.utils import set_random_seeds


def _url_retrieve(url: str, file_path: str, timeout: int):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:124.0) "
            "Gecko/20100101 Firefox/124.0"
        )
    }
    r = requests.get(url, timeout=(timeout, timeout * 10), headers=headers)
    with open(file_path, "wb") as f:
        f.write(r.content)


def _fetch_image_for_chunk(
    row,
    temp_dir: str,
    url_column: str,
    image_path_column: str,
    timeout: int,
):
    """Fetch a single image into temp_dir, preserving the relative image_path structure."""
    url = row[url_column]
    rel_path = row[image_path_column]
    full_path = os.path.join(temp_dir, rel_path)

    if os.path.isfile(full_path):
        return  # already fetched (e.g. resume within same chunk attempt)

    img_dir = os.path.dirname(full_path)
    if not os.path.isdir(img_dir):
        try:
            os.makedirs(img_dir, exist_ok=True)
        except OSError:
            print(f"Cannot create directory {img_dir}", flush=True)
            return

    try:
        _url_retrieve(url, full_path, timeout)
    except Exception as e:
        print(f"Error fetching {url}: {e}", flush=True)


def _load_progress(progress_file: str) -> tuple[list, int]:
    """
    Load progress file.

    Returns:
        completed: list of completed chunk indices
        global_shard_idx: shard index to use for the next chunk
    """
    if os.path.isfile(progress_file):
        with open(progress_file, "r") as f:
            data = json.load(f)
        return data.get("completed", []), data.get("global_shard_idx", 0)
    return [], 0


def _save_progress(progress_file: str, completed: list, global_shard_idx: int):
    with open(progress_file, "w") as f:
        json.dump({"completed": completed, "global_shard_idx": global_shard_idx}, f)


def fetch_and_pack(
    annotations_csv: str,
    temp_dir: str,
    webdataset_dir: str,
    split: str,
    label_column: str,
    image_path_column: str,
    url_column: str,
    max_shard_size: int,
    resize_min_size: Optional[int],
    category_map_json: Optional[str],
    save_category_map_json: Optional[str],
    columns_to_json: Optional[str],
    chunk_size: int,
    num_workers: int,
    request_timeout: int,
    random_seed: int,
    shuffle_images: bool,
):
    """
    Fetch images from URLs in a split CSV and pack them into webdataset shards
    one chunk at a time, deleting raw images after each chunk is packed.

    Args:
        annotations_csv: Path to CSV file with columns for url, image_path, label.
        temp_dir: Temporary directory to store raw images for the current chunk.
            Cleaned up after each chunk. A split-specific subdirectory is created
            inside this path.
        webdataset_dir: Directory where .tar shards will be written.
        split: Name of the split (e.g. "train", "val", "test"). Used as shard
            filename prefix: {split}-%06d.tar
        label_column: CSV column containing the category label.
        image_path_column: CSV column containing the relative image file path.
        url_column: CSV column containing the image download URL.
        max_shard_size: Maximum size of each shard in bytes.
        resize_min_size: If set, resize shortest image side to this value.
        category_map_json: Path to existing category map JSON. If None and
            save_category_map_json is set, the map is inferred from the CSV.
        save_category_map_json: Path to save the inferred category map.
        columns_to_json: Comma-separated list of CSV columns to embed as JSON
            metadata in each sample.
        chunk_size: Number of images to fetch and pack per chunk.
        num_workers: Number of parallel download workers.
        request_timeout: Per-request timeout in seconds.
        random_seed: Random seed for reproducibility (used when shuffling).
        shuffle_images: Whether to shuffle the dataset before chunking.
    """
    set_random_seeds(random_seed)

    df = pd.read_csv(annotations_csv)
    if shuffle_images:
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    categories_map = _get_category_map(
        df, label_column, category_map_json, save_category_map_json
    )

    os.makedirs(webdataset_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    progress_file = os.path.join(webdataset_dir, f"{split}_progress.json")
    completed, global_shard_idx = _load_progress(progress_file)

    # Shard filename pattern: e.g. .../train-%06d.tar
    pattern = os.path.join(webdataset_dir, f"{split}-%06d.tar")

    total_rows = len(df)
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    print(
        f"[fetch_and_pack] split={split} total_rows={total_rows} "
        f"chunk_size={chunk_size} num_chunks={num_chunks} "
        f"resume_from_chunk={len(completed)} global_shard_idx={global_shard_idx}",
        flush=True,
    )

    fetch_fn = partial(
        _fetch_image_for_chunk,
        temp_dir=temp_dir,
        url_column=url_column,
        image_path_column=image_path_column,
        timeout=request_timeout,
    )

    for chunk_idx in range(num_chunks):
        if chunk_idx in completed:
            print(
                f"[fetch_and_pack] chunk {chunk_idx}/{num_chunks - 1} already done, skipping.",
                flush=True,
            )
            continue

        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_rows)
        chunk_df = df.iloc[start:end].copy()

        print(
            f"[fetch_and_pack] chunk {chunk_idx}/{num_chunks - 1}: "
            f"rows {start}–{end - 1}, shard_start={global_shard_idx}",
            flush=True,
        )

        # --- Step 1: fetch images for this chunk ---
        rows = [row for _, row in chunk_df.iterrows()]
        with Pool(processes=num_workers) as pool:
            pool.map(fetch_fn, rows)

        # --- Step 2: pack fetched images into webdataset shards ---
        samples_written = 0
        next_shard_idx = global_shard_idx  # fallback if nothing written
        with wds.ShardWriter(
            pattern, maxsize=max_shard_size, start_shard=global_shard_idx
        ) as sink:
            for sample in _create_samples(
                dataset_path=temp_dir,
                categories_map=categories_map,
                dataset_df=chunk_df,
                md_results=None,
                label_column=label_column,
                image_path_column=image_path_column,
                columns_to_json=columns_to_json,
                resize_min_size=resize_min_size,
            ):
                sink.write(sample)
                samples_written += 1
            # Capture shard counter before __exit__ destroys it.
            # sink.shard is already the *next* shard index (incremented on open).
            if samples_written > 0:
                next_shard_idx = sink.shard

        global_shard_idx = next_shard_idx

        print(
            f"[fetch_and_pack] chunk {chunk_idx}: wrote {samples_written} samples, "
            f"next shard idx = {global_shard_idx}",
            flush=True,
        )

        # --- Step 3: delete raw images for this chunk ---
        for _, row in chunk_df.iterrows():
            fpath = os.path.join(temp_dir, row[image_path_column])
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except OSError as e:
                    print(f"Warning: could not delete {fpath}: {e}", flush=True)

        # Clean up any empty subdirectories left in temp_dir
        for dirpath, dirnames, filenames in os.walk(temp_dir, topdown=False):
            if dirpath == temp_dir:
                continue
            try:
                os.rmdir(dirpath)  # only removes if empty
            except OSError:
                pass  # directory not empty, leave it

        # --- Step 4: save progress ---
        completed.append(chunk_idx)
        _save_progress(progress_file, completed, global_shard_idx)

    # All chunks done — remove progress file
    if os.path.isfile(progress_file):
        os.remove(progress_file)

    print(
        f"[fetch_and_pack] complete: split={split} shards in {webdataset_dir}",
        flush=True,
    )
