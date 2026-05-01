#!/usr/bin/env python3
"""
Split a BQ-exported CSV into train/val/test CSV files.

Uses stratified splitting (sklearn) on a category column so every species
is proportionally represented in all three splits.  When --split-by-occurrence
is set, images that share the same gbif_id (same field observation)
are kept together in one split to avoid data leakage.

Required CSV columns:
  photo_id               — integer (iNat photo ID, image-level)
  relative_local_path    — path of the image within its sqfs
  gbif_id                — GBIF occurrence ID (used by --split-by-occurrence)

Optional (passed through unchanged):
  species_name, inat_taxon_id, family, gbif_id, and any other columns

Usage:
    python split.py \\
        --csv bq_export.csv \\
        --output-dir /project/.../data/splits/ \\
        --category-key species_name \\
        --val-frac 0.1 \\
        --test-frac 0.1 \\
        --split-by-occurrence \\
        --max-instances 1000 \\
        --min-instances 5 \\
        --seed 42
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {"photo_id", "relative_local_path", "gbif_id"}


def _set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _create_test_split(
    cat_counts: pd.Series,
    category_key: str,
    metadata: pd.DataFrame,
    split_by_occurrence: bool,
    split_metadata: pd.DataFrame,
    test_frac: float,
) -> pd.DataFrame:
    min_instances = math.ceil(1 / test_frac)
    test_categories = list(cat_counts[cat_counts >= min_instances].keys())
    selected = split_metadata[split_metadata[category_key].isin(test_categories)]
    _, selected, _, _ = train_test_split(
        selected, selected[[category_key]], stratify=selected[[category_key]],
        test_size=test_frac,
    )
    if split_by_occurrence:
        return metadata[metadata["gbif_id"].isin(
            selected["gbif_id"].unique()
        )].copy()
    return selected.copy()


def _create_val_split(
    cat_counts: pd.Series,
    category_key: str,
    metadata: pd.DataFrame,
    split_by_occurrence: bool,
    split_metadata: pd.DataFrame,
    test_frac: float,
    test_set: pd.DataFrame,
    val_frac: float,
) -> pd.DataFrame:
    min_instances = math.ceil(1 / val_frac)
    val_categories = list(cat_counts[cat_counts >= min_instances].keys())
    selected = split_metadata[
        ~split_metadata["relative_local_path"].isin(test_set["relative_local_path"].unique())
    ]
    selected = selected[selected[category_key].isin(val_categories)]
    adjusted_val_frac = val_frac / (1 - test_frac)
    _, selected, _, _ = train_test_split(
        selected, selected[[category_key]], stratify=selected[[category_key]],
        test_size=adjusted_val_frac,
    )
    if split_by_occurrence:
        return metadata[metadata["gbif_id"].isin(
            selected["gbif_id"].unique()
        )].copy()
    return selected.copy()


def _create_train_split(
    metadata: pd.DataFrame,
    test_set: pd.DataFrame,
    val_set: pd.DataFrame,
) -> pd.DataFrame:
    exclude = set(test_set["relative_local_path"].unique()) | set(val_set["relative_local_path"].unique())
    return metadata[~metadata["relative_local_path"].isin(exclude)].copy()


def _subsample(dataset: pd.DataFrame, max_instances: int, category_key: str) -> pd.DataFrame:
    counts = dataset[category_key].value_counts()
    over_limit = list(counts[counts > max_instances].keys())
    under_limit = dataset[~dataset[category_key].isin(over_limit)].copy()
    capped = [
        dataset[dataset[category_key] == cat].sample(max_instances)
        for cat in over_limit
    ]
    return pd.concat([under_limit] + capped, ignore_index=True)


def split_dataset(
    dataset_csv: str,
    output_dir: str,
    test_frac: float,
    val_frac: float,
    split_by_occurrence: bool,
    category_key: str,
    max_instances: int,
    min_instances: int,
    seed: int,
) -> None:
    _set_random_seeds(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading {dataset_csv} ...")
    metadata = pd.read_csv(dataset_csv, dtype={"photo_id": "Int64"})
    print(f"  {len(metadata):,} rows  columns: {list(metadata.columns)}")

    missing = REQUIRED_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    if category_key not in metadata.columns:
        raise ValueError(f"Category key '{category_key}' not found in CSV columns: {list(metadata.columns)}")

    # When splitting by occurrence, deduplicate to one row per occurrence first,
    # then pull all images for the selected occurrences back in.
    if split_by_occurrence:
        split_metadata = metadata.drop_duplicates(subset=["gbif_id"], keep="first").copy()
        print(f"  Split by occurrence: {len(split_metadata):,} unique occurrences")
    else:
        split_metadata = metadata.copy()

    cat_counts = split_metadata[category_key].value_counts()
    print(f"  {len(cat_counts):,} unique categories in '{category_key}'")

    print(f"Creating test split (frac={test_frac}) ...")
    test_set = _create_test_split(
        cat_counts, category_key, metadata, split_by_occurrence, split_metadata, test_frac
    )

    print(f"Creating val split (frac={val_frac}) ...")
    val_set = _create_val_split(
        cat_counts, category_key, metadata, split_by_occurrence, split_metadata,
        test_frac, test_set, val_frac,
    )

    print("Creating train split ...")
    train_set = _create_train_split(metadata, test_set, val_set)

    if max_instances > 0:
        print(f"Capping at max_instances={max_instances} per category ...")
        train_set = _subsample(train_set, max_instances, category_key)
        val_set   = _subsample(val_set,   int(max_instances * val_frac),  category_key)
        test_set  = _subsample(test_set,  int(max_instances * test_frac), category_key)

    if min_instances > 0:
        print(f"Filtering to min_instances={min_instances} per category ...")
        cat_counts_train = train_set[category_key].value_counts()
        keep = list(cat_counts_train[cat_counts_train >= min_instances].keys())
        train_set = train_set[train_set[category_key].isin(keep)].copy()
        val_set   = val_set[val_set[category_key].isin(keep)].copy()
        test_set  = test_set[test_set[category_key].isin(keep)].copy()
        print(f"  Kept {len(keep):,} categories with >= {min_instances} train images")

    splits = {"train": train_set, "val": val_set, "test": test_set}
    print("\n=== Split summary ===")
    for name, df in splits.items():
        n_cats = df[category_key].nunique()
        out = output_path / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"  {name}: {len(df):,} images  {n_cats:,} categories → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv",                  required=True,
                        help="BQ-exported CSV file")
    parser.add_argument("--output-dir",           required=True,
                        help="Directory to write train.csv / val.csv / test.csv")
    parser.add_argument("--category-key",         default="species_name",
                        help="Column used for stratified splitting (default: species_name)")
    parser.add_argument("--val-frac",             type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")
    parser.add_argument("--test-frac",            type=float, default=0.1,
                        help="Fraction of data for test (default: 0.1)")
    parser.add_argument("--split-by-occurrence",  action="store_true",
                        help="Keep images from the same gbif_id (GBIF occurrence) in one split")
    parser.add_argument("--max-instances",        type=int, default=1000,
                        help="Max images per category per split (0 = no cap, default: 1000)")
    parser.add_argument("--min-instances",        type=int, default=0,
                        help="Min train images per category; drop categories below this (default: 0)")
    parser.add_argument("--seed",                 type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    split_dataset(
        dataset_csv=args.csv,
        output_dir=args.output_dir,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        split_by_occurrence=args.split_by_occurrence,
        category_key=args.category_key,
        max_instances=args.max_instances,
        min_instances=args.min_instances,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
