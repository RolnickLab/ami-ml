#!/usr/bin/env python
# coding: utf-8

"""
Author	      : Aditya Jain
Last modified : May 1st, 2023
About	      : Fetches data from GBIF using Darwin Core Archive (DwC-A)
"""

from dwca.read import DwCAReader
import pandas as pd
import os
import shutil
import urllib.request 
import json
import argparse
from multiprocessing import Pool


def fetch_meta_data(data: pd.DataFrame):
    """returns the relevant metadata for a GBIF observation"""

    fields = [
        "decimalLatitude",
        "decimalLongitude",
        "order",
        "family",
        "genus",
        "species",
        "acceptedScientificName",
        "year",
        "month",
        "day",
        "datasetName",
        "taxonID",
        "acceptedTaxonKey",
        "lifeStage",
        "basisOfRecord",
    ]

    meta_data = {}

    for field in fields:
        if pd.isna(data[field]):
            meta_data[field] = "NA"
        else:
            meta_data[field] = data[field]

    return meta_data


def fetch_image_data(taxon_key: int):
    global existing_data

    # species not found on gbif
    if taxon_key == -1:
        return

    # species avaiable on gbif but data already exists
    elif (
        taxon_key in existing_data["accepted_taxon_key"].tolist()
        and existing_data.loc[
            existing_data["accepted_taxon_key"] == taxon_key, "image_count"
        ].item()
        != 0
    ):
        taxon_data = moth_data[moth_data["accepted_taxon_key"] == taxon_key].iloc[0]  # taking the first entry, just in case there are duplicate entries
        species = taxon_data["gbif_species"]

        print(f"Already downloaded for {species}.", flush=True)
        return

    # species available on gbif but data does not exist
    else:
        # get taxa information specific to the species
        taxon_data = moth_data[moth_data["accepted_taxon_key"] == taxon_key].iloc[0]  # taking the first entry, just in case there are duplicate entries
        family = taxon_data["family"]
        genus = taxon_data["genus"]
        species = taxon_data["gbif_species"]
        write_location = write_dir + family + "/" + genus + "/" + species

        # delete folder and its content, if exists already
        try:
            shutil.rmtree(write_location)
        except:
            pass

        # creating hierarchical folder structure for image storage
        try:
            os.makedirs(write_location)
        except:
            pass

        occurrence_data = occ_df.loc[occ_df["acceptedTaxonKey"] == taxon_key]
        total_occ = len(occurrence_data)
        print(
            f"Downloading for {species} which has a total of {total_occ} image occurrences.",
            flush=True,
        )
        occurrence_data = occurrence_data.sample(frac=1)
        image_count = 0
        meta_data = {}

        if total_occ != 0:
            for idx, row in occurrence_data.iterrows():
                obs_id = row["id"]

                # check occurrence entry in media dataframe
                try:
                    media_entry = media_df.loc[media_df["coreid"] == obs_id]
                    if len(media_entry) > 1:  # multiple images for an observation
                        media_entry = media_entry.iloc[0, :]
                        image_url = media_entry["identifier"]
                    else:
                        image_url = media_entry["identifier"].item()
                except Exception as e:
                    print(e, flush=True)
                    continue

                # download image
                try:
                    urllib.request.urlretrieve(
                        image_url, write_location + "/" + str(obs_id) + ".jpg"
                    )
                    image_count += 1
                    m_data = fetch_meta_data(row)
                    meta_data[str(obs_id) + ".jpg"] = m_data
                except Exception as e:
                    print(f"Error fetching url {image_url}", flush=True)
                    print(e, flush=True)
                    continue

                if image_count >= max_data_sp:
                    break

            with open(write_location + "/" + "meta_data.json", "w") as outfile:
                json.dump(meta_data, outfile)
        print(
            f"Downloading complete for {species} with {image_count} images.",
            flush=True,
        )

    return


def download_data(args):
    """main function for downloading image and meta data"""
    global media_df, occ_df, write_dir, max_data_sp, moth_data, existing_data, columns

    write_dir = args.write_directory
    species_list = args.species_checklist
    max_data_sp = args.max_images_per_species
    dwca_file = args.dwca_file

    print(f"Downloading for checklist {species_list}")

    with DwCAReader(dwca_file) as dwca:
        media_df = dwca.pd_read("multimedia.txt", parse_dates=True, on_bad_lines="skip")

        occ_df = dwca.pd_read("occurrence.txt", parse_dates=True, on_bad_lines="skip")

    # read species list
    moth_data = pd.read_csv(species_list)

    taxon_keys = list(moth_data["accepted_taxon_key"])
    taxon_keys = [int(taxon) for taxon in taxon_keys]

    # column names and data types for the count file
    columns = [
        "accepted_taxon_key",
        "family",
        "genus",
        "search_species",
        "gbif_species",
        "image_count",
    ]
    data_type = {
        "accepted_taxon_key": int,
        "family": str,
        "genus": str,
        "search_species": str,
        "gbif_species": str,
        "image_count": int,
    }

    # read an existing file or create a new one
    try:
        existing_data = pd.read_csv(
            args.write_directory + "data_statistics.csv", dtype=data_type
        )
    except:
        existing_data = pd.DataFrame(columns=columns, dtype=object)

    # fetch data using multi-processing
    with Pool(processes=args.num_workers) as pool:
        pool.map(fetch_image_data, taxon_keys)

    print("Finished downloading for the given list!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write_directory", help="path of the folder to save the data", required=True
    )
    parser.add_argument(
        "--dwca_file", help="path of the darwin core archive zip file", required=True
    )
    parser.add_argument(
        "--species_checklist",
        help="path of csv file containing list of species names along with unique GBIF taxon keys",
        required=True,
    )
    parser.add_argument(
        "--max_images_per_species",
        help="maximum number of images to download for any speices",
        default=1000,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        help="number of CPUs available",
        type=int,
    )

    args = parser.parse_args() 
    download_data(args)
