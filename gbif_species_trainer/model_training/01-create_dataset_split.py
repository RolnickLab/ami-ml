#!/usr/bin/env python
# coding: utf-8

"""
Author        : Aditya Jain
Date Started  : July 20, 2022
About         : Division of dataset into train, validation and test sets
"""
import os
import glob
import random
import pandas as pd
import argparse


def prepare_split(global_pd: pd.DataFrame, image_list: list[str], fields: list[str]):
    """
    prepares a global csv list for every type of data split

    Args:
        global_pd: a global list into which new entries will be appended
        image_list : list of new images to be appended to global list
        fields   : contains the column names
    """
    data = []

    for path in image_list:
        path_split = path.split("/")
        filename = path_split[-1]
        species = path_split[-2]
        genus = path_split[-3]
        family = path_split[-4]

        data.append([filename, family, genus, species])

    data = pd.DataFrame(data, columns=fields, dtype=object)
    global_pd = pd.concat([global_pd, data], ignore_index=True)

    return global_pd


def create_dataset_split(args):
    """main function for creating the dataset split"""

    root_dir = args.root_dir  # root directory of data
    write_dir = args.write_dir  # split files to be written
    train_split = args.train_ratio  # train set ratio
    val_split = args.val_ratio  # validation set ration
    test_split = args.test_ratio  # test set ratio
    species_list = pd.read_csv(args.species_checklist)
    assert (
        train_split + val_split + test_split == 1
    ), "Train, val and test ratios should exactly sum to 1"

    fields = ["filename", "family", "genus", "species"]
    train_data = pd.DataFrame(columns=fields, dtype=object)
    val_data = pd.DataFrame(columns=fields, dtype=object)
    test_data = pd.DataFrame(columns=fields, dtype=object)

    for _, row in species_list.iterrows():
        family = row["family_name"]
        genus = row["genus_name"]
        species = row["gbif_species_name"]

        if os.path.isdir(root_dir + "/" + family + "/" + genus + "/" + species):
            image_data = glob.glob(
                root_dir
                + family
                + "/"
                + genus
                + "/"
                + species
                + "/*.jpg"
            )
            random.shuffle(image_data)

            # calculate number of datapoints based on split ratio provided
            total = len(image_data)
            train_amt = round(total * train_split)
            val_amt = round(total * val_split)

            # assign data to the three splits
            train_list = image_data[:train_amt]
            val_list = image_data[train_amt : train_amt + val_amt]
            test_list = image_data[train_amt + val_amt :]

            train_data = prepare_split(train_data, train_list, fields)
            val_data = prepare_split(val_data, val_list, fields)
            test_data = prepare_split(test_data, test_list, fields)

    # saving the lists to disk
    train_data.to_csv(write_dir + args.filename + "_train-split.csv", index=False)
    val_data.to_csv(write_dir + args.filename + "_val-split.csv", index=False)
    test_data.to_csv(write_dir + args.filename + "_test-split.csv", index=False)

    # printing stats
    print(f"Training data size is {len(train_data)}.")
    print(f"Validation data size is {len(val_data)}.")
    print(f"Testing data size is {len(test_data)}.")
    print(f"Total images for the checklist is {len(train_data) + len(val_data) + len(test_data)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        help="path to the root directory containing the data",
        required=True,
    )
    parser.add_argument(
        "--write_dir",
        help="path to the directory for saving the split files",
        required=True,
    )
    parser.add_argument(
        "--species_checklist",
        help="species checklist containing the taxon data",
        required=True,
    )
    parser.add_argument(
        "--train_ratio",
        help="proportion of data for training",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--val_ratio",
        help="proportion of data for validation",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--test_ratio", help="proportion of data for testing", required=True, type=float
    )
    parser.add_argument(
        "--filename", help="initial name for the split files", required=True
    )
    args = parser.parse_args()

    create_dataset_split(args)
