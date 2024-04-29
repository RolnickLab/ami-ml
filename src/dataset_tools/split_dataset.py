#!/usr/bin/env python
# coding: utf-8

""" Split dataset into train/val/test partitions
"""
import math

import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset_tools.utils import set_random_seeds


def _create_test_split(
    cat_counts, category_key, metadata, split_by_occurrence, split_metadata, test_frac
):
    # Test split
    # A category will have test instances only if it has more than 1/test_frac instances
    min_instances = math.ceil(1 / test_frac)
    test_categories = list(cat_counts[cat_counts >= min_instances].keys())
    selected_set = split_metadata[split_metadata[category_key].isin(test_categories)]
    x = selected_set
    y = selected_set[[category_key]]
    _, selected_set, _, _ = train_test_split(x, y, stratify=y, test_size=test_frac)
    if split_by_occurrence:
        test_set = metadata[metadata.coreid.isin(selected_set.coreid.unique())].copy()
    else:
        test_set = selected_set.copy()
    return test_set


def _create_validation_set(
    cat_counts,
    category_key,
    metadata,
    split_by_occurrence,
    split_metadata,
    test_frac,
    test_set,
    val_frac,
):
    # validation split
    # A category will have val instances only if it has more than 1/val_frac instances
    min_instances = math.ceil(1 / val_frac)
    val_categories = list(cat_counts[cat_counts >= min_instances].keys())
    selected_set = split_metadata[
        ~split_metadata.image_path.isin(test_set.image_path.unique())
    ]
    selected_set = selected_set[selected_set[category_key].isin(val_categories)]
    x = selected_set
    y = selected_set[[category_key]]
    ajust_val_frac = val_frac / (1 - test_frac)
    _, selected_set, _, _ = train_test_split(x, y, stratify=y, test_size=ajust_val_frac)
    if split_by_occurrence:
        val_set = metadata[metadata.coreid.isin(selected_set.coreid.unique())].copy()
    else:
        val_set = selected_set.copy()
    return val_set


def _create_training_set(metadata, test_set, val_set):
    # training split
    train_set = metadata[
        (~metadata.image_path.isin(test_set.image_path.unique()))
        & (~metadata.image_path.isin(val_set.image_path.unique()))
    ]
    return train_set


def _subsample(dataset, max_instances: int, category_key: str):
    counts = dataset[category_key].value_counts()
    many_categs = list(counts[counts > max_instances].keys())

    subsampling_metadata = dataset[~dataset[category_key].isin(many_categs)].copy()
    for categ in many_categs:
        categ_df = dataset[dataset[category_key] == categ].sample(max_instances).copy()
        subsampling_metadata = pd.concat(
            [subsampling_metadata, categ_df], ignore_index=True
        )

    return subsampling_metadata


def _subsample_sets(
    category_key, max_instances, test_frac, test_set, train_set, val_frac, val_set
):
    train_set = _subsample(train_set, max_instances, category_key)
    val_set = _subsample(val_set, int(max_instances * val_frac), category_key)
    test_set = _subsample(test_set, int(max_instances * test_frac), category_key)
    return test_set, train_set, val_set


def split_dataset(
    dataset_csv: str,
    split_prefix: str,
    test_frac: float,
    val_frac: float,
    split_by_occurrence: bool,
    category_key: str,
    max_instances: int,
    min_instances: int,
    random_seed: int,
):
    set_random_seeds(random_seed)
    metadata = pd.read_csv(dataset_csv)

    if split_by_occurrence:
        split_metadata = metadata.drop_duplicates(
            subset=["coreid"], keep="first"
        ).copy()
    else:
        split_metadata = metadata.copy()

    cat_counts = split_metadata[category_key].value_counts()

    test_set = _create_test_split(
        cat_counts,
        category_key,
        metadata,
        split_by_occurrence,
        split_metadata,
        test_frac,
    )

    val_set = _create_validation_set(
        cat_counts,
        category_key,
        metadata,
        split_by_occurrence,
        split_metadata,
        test_frac,
        test_set,
        val_frac,
    )

    train_set = _create_training_set(metadata, test_set, val_set)

    if max_instances > 0:
        test_set, train_set, val_set = _subsample_sets(
            category_key,
            max_instances,
            test_frac,
            test_set,
            train_set,
            val_frac,
            val_set,
        )

    if min_instances > 0:
        cat_counts = train_set[category_key].value_counts()
        train_categories = list(cat_counts[cat_counts >= min_instances].keys())

        train_set = train_set[train_set[category_key].isin(train_categories)].copy()
        val_set = val_set[val_set[category_key].isin(train_categories)].copy()
        test_set = test_set[test_set[category_key].isin(train_categories)].copy()

    data = {"train": train_set, "val": val_set, "test": test_set}
    for set_name in data:
        data[set_name].to_csv(split_prefix + set_name + ".csv", index=False)
