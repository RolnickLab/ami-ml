#!/usr/bin/env python
# coding: utf-8

""" Split dataset into train/val/test partitions
"""
import math

import click
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import set_random_seeds


@click.command(context_settings={"show_default": True})
@click.option(
    "--dataset_csv",
    type=str,
    default=True,
    help="CSV file with dataset metadata",
)
@click.option(
    "--split_prefix",
    type=str,
    default=True,
    help="Prefix used for saving splits.",
)
@click.option(
    "--test_frac",
    type=float,
    default=0.2,
    help="Fraction of data used for the test set",
)
@click.option(
    "--val_frac",
    type=float,
    default=0.1,
    help="Fraction of data used for the validation set",
)
@click.option(
    "--split_by_occurrence",
    type=bool,
    default=True,
    help=(
        "Whether images belonging to the same occurrence should be kept in the same "
        "partition"
    ),
)
@click.option(
    "--category_key",
    type=str,
    default="speciesKey",
    help="Key used as category id for strified spliting",
)
@click.option(
    "--random_seed",
    type=int,
    default=42,
    help="Random seed for reproductible experiments",
)
def main(
    dataset_csv: str,
    split_prefix: str,
    test_frac: float,
    val_frac: float,
    split_by_occurrence: bool,
    category_key: str,
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

    # training split
    train_set = metadata[
        (~metadata.image_path.isin(test_set.image_path.unique()))
        & (~metadata.image_path.isin(val_set.image_path.unique()))
    ]

    data = {"train": train_set, "val": val_set, "test": test_set}
    for set_name in data:
        data[set_name].to_csv(split_prefix + set_name + ".csv", index=False)


if __name__ == "__main__":
    main()
