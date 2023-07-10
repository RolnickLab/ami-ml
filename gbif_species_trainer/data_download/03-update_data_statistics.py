#!/usr/bin/env python
# coding: utf-8

"""
Author	     : Aditya Jain
Date Started : May 1st, 2023
About	     : Updates the data statistics file after the image download is complete
"""

import pandas as pd
import numpy as np
import os
import argparse
import glob

def update_data_statistics(args):
    """main function for updating the data statistics file in the root moths folder"""

    species_list = pd.read_csv(args.species_checklist)
    datacount_file = pd.read_csv(args.data_directory + "data_statistics.csv")

    columns = [
        "accepted_taxon_key",
        "family_name",
        "genus_name",
        "search_species_name",
        "gbif_species_name",
        "image_count",
    ]

    for _, row in species_list.iterrows():
        family = row["family_name"]
        genus = row["genus_name"]
        search_species = row["search_species_name"]
        gbif_species = row["gbif_species_name"]
        taxon_key = row["accepted_taxon_key"]
      
        # taxa not found in gbif backbone
        if taxon_key == -1:
            # append data if not already there
            if search_species not in datacount_file["search_species_name"].tolist():
                datacount_file = pd.concat(
                    [
                        datacount_file,
                        pd.DataFrame(
                            [
                                [
                                    -1,
                                    "NotAvail",
                                    "NotAvail",
                                    search_species,
                                    "NotAvail",
                                    -1,
                                ]
                            ],
                            columns=columns,
                        ),
                    ],
                    ignore_index=True,
                )
        # taxa available in gbif backbone
        elif taxon_key not in datacount_file["accepted_taxon_key"].tolist():
            image_directory = args.data_directory + family + "/" + genus + "/" + gbif_species
            if os.path.isdir(image_directory):
                species_data = glob.glob(image_directory + "/*.jpg")
                datacount_file = pd.concat(
                    [
                        datacount_file,
                        pd.DataFrame(
                            [
                                [
                                    taxon_key,
                                    family,
                                    genus,
                                    search_species,
                                    gbif_species,
                                    len(species_data),
                                ]
                            ],
                            columns=columns,
                        ),
                    ],
                    ignore_index=True,
                )
                if len(species_data) == 0:
                    print(f"{gbif_species} has no image in the database!")
            else:
                print(f"{gbif_species} has no data folder.")
        else:
            # image data exists in the database
            pass

    # save the final file
    datacount_file.to_csv(args.data_directory + "data_statistics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_directory", 
        help="root folder where image data is saved", 
        required=True
    )

    parser.add_argument(
        "--species_checklist",
        help="path of csv file containing list of species names along with accepted taxon keys",
        required=True,
    )
    args = parser.parse_args()
    update_data_statistics(args)
