#!/usr/bin/env python
# coding: utf-8

""" Split dataset into train/val/test partitions
"""

import os
import random
from pathlib import Path

import dotenv
import pandas as pd

dotenv.load_dotenv()


def _split_list_of_files(
    all_files: list[str], train_size: float, test_size: float
) -> tuple[list[str], list[str], list[str]]:
    """
    Splits the files list into train, validation, and test sets.

    Args:
        all_files (list[str]): List of files to split.
        train_size (float): Proportion of the dataset to include in the train split.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Train, validation, and test splits.
    """
    # Shuffle the files
    random.shuffle(all_files)

    # Calculate the number of samples for each split
    num_train = int(len(all_files) * train_size)
    num_test = int(len(all_files) * test_size)
    num_val = len(all_files) - num_train - num_test

    # Split the files into train, val, and test sets
    train_files = all_files[:num_train]
    val_files = all_files[num_train : num_train + num_val]
    test_files = all_files[num_train + num_val :]

    return train_files, val_files, test_files


def _append_data_to_dataframe(
    dataframe: pd.DataFrame, files: list[str], taxon_key: str
) -> pd.DataFrame:
    """
    Appends data to a dataframe.
    Args:
        dataframe (pd.DataFrame): Dataframe to append data to.
        files (list[str]): List of files to append.
    Returns:
        pd.DataFrame: Updated dataframe.
    """
    for file in files:
        new_row = pd.DataFrame(
            {
                "taxonkey": [taxon_key],
                "filename": [file],
            }
        )
        dataframe = pd.concat([dataframe, new_row], ignore_index=True)

    return dataframe


def create_dataset_splits(
    data_dir: str,
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_seed: int = 42,
    min_samples_per_class: int = 5,
) -> None:
    """
    Splits the dataset into train, validation, and test sets.

    Args:
        data (str): Path to the dataset directory.
        train_size (float): Proportion of the dataset to include in the train split.
        val_size (float): Proportion of the dataset to include in the validation split.
        test_size (float): Proportion of the dataset to include in the test split.
        random_seed (int): Random seed for reproducibility.
        min_samples_per_class (int): Minimum number of samples per class to include for training. Otherwise, the class will only appear in the test set.

    Returns:
        tuple: Train, validation, and test splits.
    """
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Splits should sum to 1
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"

    # Create dataframe splits
    train_set = pd.DataFrame(columns=["taxonkey", "filename"])
    val_set = pd.DataFrame(columns=["taxonkey", "filename"])
    test_set = pd.DataFrame(columns=["taxonkey", "filename"])

    # Iterate over each taxon key in the dataset directory
    for taxon_key in os.listdir(data_dir):
        # List all files in the taxon directory
        taxon_dir = Path(data_dir) / taxon_key
        all_files = os.listdir(taxon_dir)

        # # If the number of samples is less than the minimum, add all to test set
        if len(all_files) < min_samples_per_class:
            test_set = _append_data_to_dataframe(test_set, all_files, taxon_key)

        # Do the split
        else:
            train_files, val_files, test_files = _split_list_of_files(
                all_files,
                train_size=train_size,
                test_size=test_size,
            )

            # Append to respective dataframes
            train_set = _append_data_to_dataframe(train_set, train_files, taxon_key)
            val_set = _append_data_to_dataframe(val_set, val_files, taxon_key)
            test_set = _append_data_to_dataframe(test_set, test_files, taxon_key)

    # Save the splits to disk
    train_set.to_csv(Path(data_dir) / "train.csv", index=False)
    val_set.to_csv(Path(data_dir) / "val.csv", index=False)
    test_set.to_csv(Path(data_dir) / "test.csv", index=False)


if __name__ == "__main__":
    DATA_DIR = os.getenv(
        "FINE_TUNING_UK_DENMARK_AMI_TRAPS_DATASET", "./fine_tuning_data/ami_traps"
    )
    create_dataset_splits(DATA_DIR)
