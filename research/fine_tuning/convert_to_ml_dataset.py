#!/usr/bin/env python
# coding: utf-8

""" Conversion of labelled insect camera trap data to a ML format data
"""

import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Tuple

import dotenv

dotenv.load_dotenv()

AMI_TRAPS_DATASET = os.getenv("AMI_TRAPS_DATASET", "./ami_traps_dataset")
FINE_TUNING_UK_DENMARK_AMI_TRAPS_DATASET = os.getenv(
    "FINE_TUNING_UK_DENMARK_AMI_TRAPS_DATASET", "./fine_tuning_data/ami_traps"
)
SPECIES_EXCLUSION_LIST = os.getenv(
    "SPECIES_EXCLUSION_LIST", "ami-traps_sp_missing_in_ami-gbif.pickle"
)


def _get_ground_truth_info(
    insect_labels: dict, img_name: str
) -> Tuple[str, str, int, str]:
    """Get ground truth label information

    Args:
        insect_labels (dict): Dictionary containing insect labels.
        img_name (str): Name of the image.

    Returns:
        tuple: Ground truth label, rank, accepted taxon key, and region.
    """

    gt_label = insect_labels[img_name]["label"]
    gt_rank = insect_labels[img_name]["taxon_rank"]
    gt_accepted_taxon_key = insect_labels[img_name]["acceptedTaxonKey"]
    gt_region = insect_labels[img_name]["region"]

    return gt_label, gt_rank, gt_accepted_taxon_key, gt_region


def convert_raw_data_to_structured_format(
    ami_traps_dataset_dir: str,
    fine_tuning_dataset_dir: str,
    species_exclusion_list_f: str,
    region: str,
) -> None:
    """Convert raw AMI Traps data to a structured ML format. It only fiters crops for a particular region, labelled at the species level, and those not in the species exclusion list.

    Args:
        ami_traps_dataset_dir (str): Path to the directory containing the original AMI traps dataset.
        fine_tuning_dataset_dir (str): Path to the directory where the AMI traps data will be structured and stored.
        species_exclusion_list_f (str): Path to the file containing the species exclusion list.
        region (str): Region to store data for.

    Returns:
        None: The function processes and saves the dataset in the specified directory.
    """

    # Get all moth insect crops label information
    with open(Path(ami_traps_dataset_dir) / "fgrained_labels.json") as f:
        insect_labels = json.load(f)

    # Get the species exclusion list
    with open(species_exclusion_list_f, "rb") as f:
        species_exclusion_list = pickle.load(f)

    for image_name in insect_labels.keys():
        # Get ground truth label information
        _, gt_rank, gt_accepted_taxon_key, gt_region = _get_ground_truth_info(
            insect_labels, image_name
        )

        if (
            gt_rank == "SPECIES"
            and gt_region == region
            and gt_accepted_taxon_key != -1
            and (gt_accepted_taxon_key not in species_exclusion_list)
        ):
            # Create directory if doesn't exist
            target_dir = Path(fine_tuning_dataset_dir) / str(gt_accepted_taxon_key)
            os.makedirs(target_dir, exist_ok=True)

            # Copy image to the directory
            image_path = Path(ami_traps_dataset_dir) / (Path(image_name).stem + ".png")
            try:
                shutil.copy(image_path, target_dir)
            except FileNotFoundError:
                print(f"File not found: {image_path}")
                continue


if __name__ == "__main__":
    convert_raw_data_to_structured_format(
        AMI_TRAPS_DATASET,
        FINE_TUNING_UK_DENMARK_AMI_TRAPS_DATASET,
        SPECIES_EXCLUSION_LIST,
        "WesternEurope",
    )
