#!/usr/bin/env python
# coding: utf-8

"""
Author       : Aditya Jain
Date Started : May 9, 2022
About        : This script fetches unique taxon keys for species list from GBIF database
"""

from pygbif import occurrences as occ
from pygbif import species as species_api
import pandas as pd
import os
import tqdm
import urllib
import json
import time
import argparse


def get_gbif_key_backbone(name: str, place: str):
    """given a species name, this function returns the unique gbif key and other
    attributes using backbone API
    """

    # default values
    acc_taxon_key = [-1]
    order = ["NotAvail"]
    family = ["NotAvail"]
    genus = ["NotAvail"]
    search_species = [name]
    gbif_species = [
        "NotAvail"
    ]  # the name returned on search, can be different from the search
    status = ["NotAvail"]
    rank = ["NotAvail"]
    place = [place]

    data = species_api.name_backbone(name=name, strict=True, rank="species")

    # add entries to the fields
    confidence = [data["confidence"]]
    match_type = [data["matchType"]]
    if "order" in data.keys():
        order = [data["order"]]
    if "family" in data.keys():
        family = [data["family"]]
    if "genus" in data.keys():
        genus = [data["genus"]]
    if "status" in data.keys():
        status = [data["status"]]
    if "rank" in data.keys():
        rank = [data["rank"]]

    if data["matchType"] != "NONE" and data["matchType"] != "HIGHERRANK":
        gbif_species = [data["species"]]
        if "acceptedUsageKey" in data.keys():
            acc_taxon_key = [data["acceptedUsageKey"]]
        else:
            acc_taxon_key = [data["usageKey"]]

    df = pd.DataFrame(
        list(
            zip(
                acc_taxon_key,
                order,
                family,
                genus,
                search_species,
                gbif_species,
                confidence,
                status,
                match_type,
                rank,
                place,
            )
        ),
        columns=[
            "accepted_taxon_key",
            "order_name",
            "family_name",
            "genus_name",
            "search_species_name",
            "gbif_species_name",
            "confidence",
            "status",
            "match_type",
            "rank",
            "source",
        ],
    )
    return df


def save_taxon_keys(args):
    """main function for saving the taxon keys and related data for each species"""

    # fetch species names from the list
    data = pd.read_csv(args.species_filepath, index_col=False)
    species_list = []
    for indx in data.index:
        species_list.append(data[args.column_name][indx])

    # define all columns
    data_final = pd.DataFrame(
        columns=[
            "accepted_taxon_key",
            "order_name",
            "family_name",
            "genus_name",
            "search_species_name",
            "gbif_species_name",
            "confidence",
            "status",
            "match_type",
            "rank",
            "source",
        ],
        dtype=object,
    )

    # fetch taxonomy data from GBIF
    for name in species_list:
        data = get_gbif_key_backbone(name, args.place)
        data_final = pd.concat([data_final, data], ignore_index=True)

    # save the file
    data_final.to_csv(args.output_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--species_filepath", help="path of the species list", required=True
    )
    parser.add_argument(
        "--column_name", help="column name of the species entries", required=True
    )
    parser.add_argument(
        "--output_filepath",
        help="path to the output file with csv extension",
        required=True,
    )
    parser.add_argument(
        "--place", help="source name from which the list is obtained", required=True
    )
    args = parser.parse_args()

    save_taxon_keys(args)
