#!/usr/bin/env python
# coding: utf-8

"""
Author        : Aditya Jain
Date started  : April 9, 2024
About         : Exports the AMI-Traps dataset to webdataset format and individual crops
"""

import json
import os

import pandas as pd
import webdataset as wds

# 3rd party packages
from dotenv import load_dotenv
from PIL import Image

# Load secrets and config from optional .env file
load_dotenv()


def get_synonym(sp_checklist: pd.DataFrame, species: str):
    """Return synonym name of a species, if it exists on GBIF"""

    # Search for species in the checklist
    try:
        synonym = sp_checklist.loc[sp_checklist["search_species"] == species][
            "gbif_species"
        ].values[0]
    except IndexError:
        synonym = None
        # print(f"{species} is not found in the search species column.", flush=True)

    if synonym == species:
        return None
    else:
        return synonym


def get_gbif_keys(sp_key_map: pd.DataFrame, sp_checklist: pd.DataFrame, species: str):
    """Get the species key and accepted taxon key for a given species"""

    speciesKey, acceptedTaxonKey = None, None

    # Get species key from GBIF species key map
    try:
        speciesKey = int(
            sp_key_map.loc[sp_key_map["species"] == species]["speciesKey"].values[0]
        )
    except IndexError:
        pass

    # Get accepted taxon key from checklist
    try:
        acceptedTaxonKey = int(
            sp_checklist.loc[sp_checklist["search_species"] == species][
                "accepted_taxon_key"
            ].values[0]
        )
    except IndexError:
        try:
            acceptedTaxonKey = int(
                sp_checklist.loc[sp_checklist["gbif_species"] == species][
                    "accepted_taxon_key"
                ].values[0]
            )
        except IndexError:
            pass

    return speciesKey, acceptedTaxonKey


def export_to_webdataset_and_crops(
    data_dir: str, export_dir: str, sp_checklist: pd.DataFrame, sp_key_map: pd.DataFrame
):
    """Main function for exporting AMI-Traps to webdataset and individual crops"""

    # Get the list of raw camera trap images and other metadata
    image_dir = os.path.join(data_dir, "images")
    metadata_dir = os.path.join(data_dir, "metadata")
    labels_dir = os.path.join(data_dir, "labels")
    image_list = os.listdir(image_dir)

    # Get ground truth class names
    gt_class_list = json.load(open(os.path.join(data_dir, "notes.json")))["categories"]

    # Webdataset specific variables
    max_shard_size = 50 * 1024 * 1024
    wbd_pattern_binary = os.path.join(
        export_dir, "webdataset", "binary_classification", "binary-%06d.tar"
    )
    wbd_pattern_fgrained = os.path.join(
        export_dir, "webdataset", "fine-grained_classification", "fgrained-%06d.tar"
    )
    sink_binary = wds.ShardWriter(wbd_pattern_binary, maxsize=max_shard_size)
    sink_fgrained = wds.ShardWriter(wbd_pattern_fgrained, maxsize=max_shard_size)

    # Variables for writing annotations on the disk
    label_binary, label_fgrained = {}, {}
    binary_crop_count = 1
    fgrained_crop_count = 1

    # Iterate over each image separately
    for image in image_list:
        img_basename = os.path.splitext(image)[0]

        # Fetch the region name for the image
        metadata_file = os.path.join(metadata_dir, img_basename + ".json")
        metadata = json.load(open(metadata_file))
        region = metadata["region"]

        # Read the raw image
        try:
            raw_image = Image.open(os.path.join(image_dir, image))
            img_width, img_height = raw_image.size
        except FileNotFoundError:
            raise Exception(f"Image file not found: {os.path.join(image_dir, image)}")
        except Exception as e:
            raise Exception(f"Error opening image {image}: {str(e)}")

        # Get the ground truth bounding box and label
        labels = open(os.path.join(labels_dir, img_basename + ".txt"), "r")

        # Iterate over each annotation/crop separately
        for line in labels:
            label_id, x, y, w, h = (
                int(line.split()[0]),
                float(line.split()[1]),
                float(line.split()[2]),
                float(line.split()[3]),
                float(line.split()[4]),
            )

            # Get the ground truth class name
            for class_entry in gt_class_list:
                if class_entry["id"] == label_id:
                    label_name = class_entry["name"]
                    label_rank = class_entry["rank"]
                    break

            # Ignore unlabeled crops
            if label_name != "Unidentifiable" and label_name != "Unclassified":
                # Get the insect crop
                x_start = int((x - w / 2) * img_width)
                y_start = int((y - h / 2) * img_height)
                w_px, h_px = int(w * img_width), int(h * img_height)
                insect_crop = raw_image.crop(
                    (x_start, y_start, x_start + w_px, y_start + h_px)
                )

                # Save the insect crop as image
                insect_crop.save(
                    os.path.join(
                        export_dir, "insect_crops", str(binary_crop_count) + ".jpg"
                    )
                )

                # Export to webdataset for binary classification
                if label_name != "Non-Moth":
                    binary_class = "Moth"
                else:
                    binary_class = "Non-Moth"
                sample_binary_annotation = {"label": binary_class, "region": region}
                sample_binary = {
                    "__key__": img_basename + "_" + str(binary_crop_count),
                    "jpg": insect_crop,
                    "json": sample_binary_annotation,
                }
                sink_binary.write(sample_binary)

                # Save the binary annotation for the individual crop
                label_binary[str(binary_crop_count) + ".jpg"] = sample_binary_annotation

                # Export to webdataset for finegrained classification, if moth crop
                if label_rank != "NA":
                    # If exists, get the synonym name and gbif keys for the species
                    synonym = None
                    speciesKey = None
                    acceptedTaxonKey = None
                    if label_rank == "SPECIES":
                        synonym = get_synonym(sp_checklist, label_name)
                        speciesKey, acceptedTaxonKey = get_gbif_keys(
                            sp_key_map, sp_checklist, label_name
                        )

                    # Export to webdataset for fine-grained classification
                    sample_fgrained_annotation = {
                        "taxon_rank": label_rank,
                        "label": label_name,
                        "synonym": synonym,
                        "speciesKey": speciesKey,
                        "acceptedTaxonKey": acceptedTaxonKey,
                        "region": region,
                    }
                    sample_fgrained = {
                        "__key__": img_basename + "_" + str(fgrained_crop_count),
                        "jpg": insect_crop,
                        "json": sample_fgrained_annotation,
                    }
                    sink_fgrained.write(sample_fgrained)

                    # Save the finegrained annotation for the individual crop
                    label_fgrained[
                        str(binary_crop_count) + ".jpg"
                    ] = sample_fgrained_annotation
                    fgrained_crop_count += 1

                binary_crop_count += 1

    sink_binary.close()
    sink_fgrained.close()
    with open(os.path.join(export_dir, "insect_crops", "binary_labels.json"), "w") as f:
        json.dump(label_binary, f, indent=3)
    with open(
        os.path.join(export_dir, "insect_crops", "fgrained_labels.json"), "w"
    ) as f:
        json.dump(label_fgrained, f, indent=3)
    print(
        "The export is complete!",
        flush=True,
    )


if __name__ == "__main__":
    ECCV2024_DATA = os.getenv("ECCV2024_DATA_PATH")
    SPECIES_LISTS_DIR_PATH = os.getenv("SPECIES_LISTS_DIR_PATH")
    MASTER_SPECIES_LIST = os.getenv("MASTER_SPECIES_LIST")

    data_dir = f"{ECCV2024_DATA}/ami_traps_dataset"
    export_dir = f"{ECCV2024_DATA}/camera_ready_amitraps"

    species_checklist = pd.read_csv(f"{SPECIES_LISTS_DIR_PATH}/{MASTER_SPECIES_LIST}")
    species_key_map = pd.read_csv(f"{ECCV2024_DATA}/speciesKey_map.csv")

    export_to_webdataset_and_crops(
        data_dir, export_dir, species_checklist, species_key_map
    )
