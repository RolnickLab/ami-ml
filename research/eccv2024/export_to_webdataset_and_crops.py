#!/usr/bin/env python
# coding: utf-8

"""
Author        : Aditya Jain
Date started  : April 9, 2024
About         : Exports the AMI-Traps dataset to webdataset format and individual crops
"""

import json
import os
import pathlib
from pathlib import Path

import pandas as pd

# 3rd party packages
import PIL
import PIL.Image
import webdataset as wds
from PIL import Image


def _get_synonym(sp_checklist: pd.DataFrame, species: str):
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


def _get_gbif_keys(sp_key_map: pd.DataFrame, sp_checklist: pd.DataFrame, species: str):
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


def _create_webdataset_sink(export_dir: str, max_shard_size: int):
    """Create webdataset sink for binary and fine-grained classification"""
    wbd_pattern_binary = str(
        export_dir / "webdataset" / "binary_classification" / "binary-%06d.tar"
    )
    wbd_pattern_fgrained = str(
        export_dir / "webdataset" / "fine-grained_classification" / "fgrained-%06d.tar"
    )
    sink_binary = wds.ShardWriter(wbd_pattern_binary, maxsize=max_shard_size)
    sink_fgrained = wds.ShardWriter(wbd_pattern_fgrained, maxsize=max_shard_size)
    return sink_binary, sink_fgrained


def _get_region_name(metadata_dir: pathlib.PosixPath, img_basename: str):
    """Get region name from the metadata file"""
    metadata_file = metadata_dir / (img_basename + ".json")
    with open(metadata_file) as f:
        metadata = json.load(f)
        region = metadata["region"]
    return region


def _read_image(image_dir: pathlib.PosixPath, image: str):
    """Read the raw image"""
    try:
        raw_image = Image.open(image_dir / image)
        img_width, img_height = raw_image.size
    except FileNotFoundError:
        raise Exception(f"Image file not found: {image_dir / image}")
    except Exception as e:
        raise Exception(f"Error opening image {image}: {str(e)}")
    return raw_image, img_width, img_height


def _get_gt_classes(data_dir: pathlib.PosixPath):
    """Get the ground truth class names"""
    with open(data_dir / "notes.json") as f:
        gt_class_list = json.load(f)["categories"]
    return gt_class_list


def _get_gt_labels(labels_dir: pathlib.PosixPath, img_basename: str):
    """Get the ground truth bounding boxes and label"""
    with open((labels_dir / (img_basename + ".txt"))) as f:
        labels = f.readlines()
    return labels


def _get_bbox_annotation(line: str):
    """Get the ground truth class name and bounding box annotation"""
    label_id, x, y, w, h = (
        int(line.split()[0]),
        float(line.split()[1]),
        float(line.split()[2]),
        float(line.split()[3]),
        float(line.split()[4]),
    )
    return label_id, x, y, w, h


def _get_gt_class_name(gt_class_list: list, label_id: int):
    """Get the ground truth class name and rank from numerical id"""
    for class_entry in gt_class_list:
        if class_entry["id"] == label_id:
            label_name = class_entry["name"]
            label_rank = class_entry["rank"]
            break
    return label_name, label_rank


def _get_insect_crop(
    raw_image: PIL.Image.Image,
    x: float,
    y: float,
    w: float,
    h: float,
    img_width: int,
    img_height: int,
):
    """Get the insect crop from the raw image"""
    x_start = int((x - w / 2) * img_width)
    y_start = int((y - h / 2) * img_height)
    w_px, h_px = int(w * img_width), int(h * img_height)
    insect_crop = raw_image.crop((x_start, y_start, x_start + w_px, y_start + h_px))
    return insect_crop


def _get_binary_wbd_sample(
    label_name: str,
    region: str,
    img_basename: str,
    binary_crop_count: int,
    insect_crop: PIL.Image.Image,
):
    """Get the webdataset sample for binary classification"""

    if label_name != "Non-Moth":
        binary_class = "Moth"
    else:
        binary_class = "Non-Moth"
    sample_binary_annotation = {"label": binary_class, "region": region}
    sample_binary_wbd = {
        "__key__": img_basename + "_" + str(binary_crop_count),
        "jpg": insect_crop,
        "json": sample_binary_annotation,
    }
    return sample_binary_wbd, sample_binary_annotation


def _get_fgrained_wbd_sample(
    label_name: str,
    region: str,
    img_basename: str,
    fgrained_crop_count: int,
    insect_crop: PIL.Image.Image,
    label_rank: str,
    sp_checklist: pd.DataFrame,
    sp_key_map: pd.DataFrame,
):
    """Get the webdataset sample for fine-grained classification"""

    # If exists, get the synonym name and gbif keys for the species
    synonym = None
    speciesKey = None
    acceptedTaxonKey = None
    if label_rank == "SPECIES":
        synonym = _get_synonym(sp_checklist, label_name)
        speciesKey, acceptedTaxonKey = _get_gbif_keys(
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
    sample_fgrained_wbd = {
        "__key__": img_basename + "_" + str(fgrained_crop_count),
        "jpg": insect_crop,
        "json": sample_fgrained_annotation,
    }
    return sample_fgrained_wbd, sample_fgrained_annotation


def export_to_webdataset_and_crops(
    data_dir: str, export_dir: str, sp_checklist: pd.DataFrame, sp_key_map: pd.DataFrame
):
    """Main function for exporting AMI-Traps to webdataset and individual crops"""

    # Get the list of raw camera trap images and other metadata
    data_dir = Path(data_dir)
    export_dir = Path(export_dir)
    image_dir = data_dir / "images"
    metadata_dir = data_dir / "metadata"
    labels_dir = data_dir / "labels"
    image_list = os.listdir(image_dir)

    # Get ground truth class names
    gt_class_list = _get_gt_classes(data_dir)

    # Webdataset specific variables
    max_shard_size = 50 * 1024 * 1024
    sink_binary, sink_fgrained = _create_webdataset_sink(export_dir, max_shard_size)

    # Variables for writing annotations on the disk
    label_binary, label_fgrained = {}, {}
    binary_crop_count = 1
    fgrained_crop_count = 1

    # Iterate over each image separately
    for image in image_list:
        img_basename = os.path.splitext(image)[0]

        # Fetch the region name for the image
        region = _get_region_name(metadata_dir, img_basename)

        # Read the raw image
        raw_image, img_width, img_height = _read_image(image_dir, image)

        # Get the ground truth bounding box and label
        labels = _get_gt_labels(labels_dir, img_basename)

        # Iterate over each annotation separately
        for line in labels:
            label_id, x, y, w, h = _get_bbox_annotation(line)

            # Get the ground truth class name and rank
            label_name, label_rank = _get_gt_class_name(gt_class_list, label_id)

            # Ignore unlabeled crops
            if label_name != "Unidentifiable" and label_name != "Unclassified":
                # Get the insect crop
                insect_crop = _get_insect_crop(
                    raw_image, x, y, w, h, img_width, img_height
                )

                # Save the insect crop as image
                insect_crop.save(
                    export_dir / "insect_crops" / (str(binary_crop_count) + ".jpg")
                )

                # Export to webdataset for binary classification
                sample_binary_wbd, sample_binary_annotation = _get_binary_wbd_sample(
                    label_name, region, img_basename, binary_crop_count, insect_crop
                )
                sink_binary.write(sample_binary_wbd)

                # Save the binary annotation for the individual crop
                label_binary[str(binary_crop_count) + ".jpg"] = sample_binary_annotation

                # Export to webdataset for finegrained classification, if moth crop
                if label_rank != "NA":
                    (
                        sample_fgrained_wbd,
                        sample_fgrained_annotation,
                    ) = _get_fgrained_wbd_sample(
                        label_name,
                        region,
                        img_basename,
                        fgrained_crop_count,
                        insect_crop,
                        label_rank,
                        sp_checklist,
                        sp_key_map,
                    )
                    sink_fgrained.write(sample_fgrained_wbd)

                    # Save the finegrained annotation for the individual crop
                    label_fgrained[
                        str(binary_crop_count) + ".jpg"
                    ] = sample_fgrained_annotation
                    fgrained_crop_count += 1

                binary_crop_count += 1

    sink_binary.close()
    sink_fgrained.close()
    with open(export_dir / "insect_crops" / "binary_labels.json", "w") as f:
        json.dump(label_binary, f, indent=3)
    with open(export_dir / "insect_crops" / "fgrained_labels.json", "w") as f:
        json.dump(label_fgrained, f, indent=3)
    print(
        "The export is complete!",
        flush=True,
    )


if __name__ == "__main__":
    ECCV2024_DATA = os.getenv("ECCV2024_DATA")
    SPECIES_LISTS_DIR = os.getenv("SPECIES_LISTS_DIR")
    MASTER_SPECIES_LIST = os.getenv("MASTER_SPECIES_LIST")

    data_dir = f"{ECCV2024_DATA}/ami_traps_dataset"
    export_dir = f"{ECCV2024_DATA}/camera_ready_amitraps"

    species_checklist = pd.read_csv(f"{SPECIES_LISTS_DIR}/{MASTER_SPECIES_LIST}")
    species_key_map = pd.read_csv(f"{ECCV2024_DATA}/speciesKey_map.csv")

    export_to_webdataset_and_crops(
        data_dir, export_dir, species_checklist, species_key_map
    )
