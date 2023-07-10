#!/usr/bin/env python
# coding: utf-8

"""
Author              : Aditya Jain
Date last modified  : May 8th, 2023
About               : Calculates information and statistics regarding the taxonomy and data
"""
import pandas as pd
import json
import argparse


def category_map(args):
    """converts string labels (species, genus, family) to numeric labels and also the reverse"""

    write_dir = args.write_dir
    data = pd.read_csv(args.species_checklist)

    species_list = list(set(data["gbif_species_name"]))
    genus_list = list(set(data["genus_name"]))
    family_list = list(set(data["family_name"]))

    try:
        species_list.remove("NotAvail")
        genus_list.remove("NotAvail")
        family_list.remove("NotAvail")
    except:
        pass

    print(
        f"Total families: {len(family_list)}, genuses: {len(genus_list)}, species: {len(species_list)}"
    )

    str2num = {}
    str2num["family"] = family_list
    str2num["genus"] = genus_list
    str2num["species"] = species_list
    str2num[
        "Note"
    ] = "The integer index in their respective list will be the numeric class label"

    with open(write_dir + args.numeric_labels_filename + ".json", "w") as outfile:
        json.dump(str2num, outfile, indent=4)

    # building the reverse category map
    categories_map = {}
    species_list = str2num["species"]

    for i in range(len(species_list)):
        categories_map[species_list[i]] = i

    with open(write_dir + args.category_map_filename + ".json", "w") as outfile:
        json.dump(categories_map, outfile)


def taxon_hierarchy(args):
    """saves the taxon hierarchy for each species"""

    write_dir = args.write_dir
    data = pd.read_csv(args.species_checklist)

    taxon_hierarchy = {}
    taxon_hierarchy["Note"] = "The 0th index is genus and 1st index is family"

    for indx in data.index:
        if (
            data["gbif_species_name"][indx] not in taxon_hierarchy.keys()
            and data["gbif_species_name"][indx] != "NotAvail"
        ):
            taxon_hierarchy[data["gbif_species_name"][indx]] = [
                data["genus_name"][indx],
                data["family_name"][indx],
            ]

    with open(write_dir + args.taxon_hierarchy_filename + ".json", "w") as outfile:
        json.dump(taxon_hierarchy, outfile, indent=4)


def count_training_points(args):
    """counts the number of training points for each taxa"""

    train_data = pd.read_csv(args.train_split_file)
    species_list = pd.read_csv(args.species_checklist)
    image_count = {}
    image_count["family"] = {}
    image_count["genus"] = {}
    image_count["species"] = {}

    for _, row in species_list.iterrows():
        family = row["family_name"]
        genus = row["genus_name"]
        species = row["gbif_species_name"]

        # calculate the counts
        image_count["family"][family] = list(train_data["family"]).count(family)
        image_count["genus"][genus] = list(train_data["genus"]).count(genus)
        image_count["species"][species] = list(train_data["species"]).count(species)

    with open(args.write_dir + args.training_points_filename + ".json", "w") as outfile:
        json.dump(image_count, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--species_checklist", help="path to the species list", required=True
    )
    parser.add_argument(
        "--write_dir",
        help="path to the directory for saving the information",
        required=True,
    )
    parser.add_argument(
        "--numeric_labels_filename",
        help="filename for numeric labels file",
        required=True,
    )
    parser.add_argument(
        "--category_map_filename",
        help="filename for the category map from integers to species names",
        required=True,
    )
    parser.add_argument(
        "--taxon_hierarchy_filename",
        help="filename for taxon hierarchy file",
        required=True,
    )
    parser.add_argument(
        "--training_points_filename",
        help="filename for storing the count of training points",
        required=True,
    )
    parser.add_argument(
        "--train_split_file", help="path to the training split file", required=True
    )
    args = parser.parse_args()

    category_map(args)
    taxon_hierarchy(args)
    count_training_points(args)
