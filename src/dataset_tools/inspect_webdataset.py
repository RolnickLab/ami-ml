#!/usr/bin/env python
# coding: utf-8

"""
Inspect a webdataset: count images, classes, and show class distribution.

Usage:
    python src/dataset_tools/inspect_webdataset.py \
        --webdataset-dir data/vermont_butterflies/webdataset \
        --sort-by count

    python src/dataset_tools/inspect_webdataset.py \
        --webdataset-dir data/vermont_butterflies/webdataset \
        --category-map-json data/vermont_butterflies/webdataset/vermont_category_map_verbatim.json \
        --split train \
        --top-n 20 \
        --output-json /tmp/stats.json
"""

import argparse
import collections
import glob
import json
import os
import statistics
import tarfile


def find_category_map(webdataset_dir, explicit_path=None):
    if explicit_path:
        return explicit_path
    candidates = glob.glob(os.path.join(webdataset_dir, "*_category_map.json"))
    if len(candidates) == 1:
        print(f"Auto-discovered category map: {candidates[0]}", flush=True)
        return candidates[0]
    elif len(candidates) == 0:
        print("Warning: no *_category_map.json found; showing raw class IDs.", flush=True)
        return None
    else:
        names = [os.path.basename(c) for c in candidates]
        print(
            f"Warning: multiple category maps found ({names}); pass --category-map-json explicitly.",
            flush=True,
        )
        return None


def inspect_webdataset(webdataset_dir, category_map_json=None, split=None):
    map_path = find_category_map(webdataset_dir, category_map_json)
    id_to_name = {}
    if map_path:
        with open(map_path) as f:
            name_to_id = json.load(f)
        id_to_name = {v: k for k, v in name_to_id.items()}

    shards = sorted(glob.glob(os.path.join(webdataset_dir, "**/*.tar"), recursive=True))
    if split:
        shards = [s for s in shards if split in os.path.basename(s)]

    class_counts = collections.Counter()
    for i, shard in enumerate(shards):
        print(f"  Reading shard {i + 1}/{len(shards)}: {os.path.basename(shard)}", flush=True)
        try:
            with tarfile.open(shard) as tf:
                for member in tf.getmembers():
                    if member.name.endswith(".cls"):
                        cls_bytes = tf.extractfile(member).read().strip()
                        cls_id = int(cls_bytes)
                        class_counts[cls_id] += 1
        except Exception as e:
            print(f"  Warning: error reading {shard}: {e}", flush=True)

    return {
        "total_images": sum(class_counts.values()),
        "num_classes": len(class_counts),
        "num_shards": len(shards),
        "class_counts": {id_to_name.get(k, str(k)): v for k, v in class_counts.items()},
    }


def print_stats(stats, webdataset_dir, split=None, sort_by="count", top_n=None):
    total = stats["total_images"]
    counts = stats["class_counts"]

    print(f"\nWebdataset: {webdataset_dir}" + (f"  [split: {split}]" if split else ""))
    print(f"Shards:      {stats['num_shards']:,}")
    print(f"Images:      {total:,}")
    print(f"Classes:     {stats['num_classes']:,}")

    if not counts:
        return

    if sort_by == "name":
        sorted_counts = sorted(counts.items(), key=lambda x: x[0])
    else:
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    if top_n:
        sorted_counts = sorted_counts[:top_n]

    print(f"\nClass distribution (sorted by {sort_by}):")
    print(f"  {'Rank':>4}  {'Class':<35} {'Count':>8}   {'%':>5}")
    print(f"  {'-'*4}  {'-'*35} {'-'*8}   {'-'*5}")
    for rank, (name, count) in enumerate(sorted_counts, 1):
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"  {rank:>4}  {name:<35} {count:>8,}   {pct:>5.1f}")

    values = list(counts.values())
    print(
        f"\nMin: {min(values):,}   Max: {max(values):,}   "
        f"Mean: {sum(values)/len(values):.1f}   "
        f"Median: {statistics.median(values):.1f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a webdataset: count images and show class distribution."
    )
    parser.add_argument(
        "--webdataset-dir",
        required=True,
        help="Directory to scan for .tar shards (searched recursively)",
    )
    parser.add_argument(
        "--category-map-json",
        default=None,
        help="Optional: JSON mapping class_name -> class_id. Auto-discovered if omitted.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional: filter to a specific split by name (e.g. train, val, test)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional: save full stats to a JSON file",
    )
    parser.add_argument(
        "--sort-by",
        choices=["name", "count"],
        default="count",
        help="Sort class list by name or count (default: count descending)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only print the top N classes (default: all)",
    )
    args = parser.parse_args()

    stats = inspect_webdataset(
        webdataset_dir=args.webdataset_dir,
        category_map_json=args.category_map_json,
        split=args.split,
    )

    print_stats(
        stats,
        webdataset_dir=args.webdataset_dir,
        split=args.split,
        sort_by=args.sort_by,
        top_n=args.top_n,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
