"""
Author: Aditya Jain
Date last modified: March 6th, 2024
About: Long-tailed analysis
"""

import json
import os
import pickle

import pandas as pd
from dotenv import load_dotenv

# Load secrets and config from optional .env file
load_dotenv()


def long_tailed_accuracy(
    pred_dir: str, exclusion_sp: list[str], sp_key_map: pd.DataFrame, gbif_count: dict
):
    """Main function for calculating accuracy in many, medium and few buckets"""

    # Variables
    image_pred_list = os.listdir(pred_dir)
    species_acc = {}
    many, medium, few = [], [], []

    # Iterate over each image prediction
    for image_pred in image_pred_list:
        pred_data = json.load(open(os.path.join(pred_dir, image_pred)))

        # Iterate over each bounding box
        for bbox in pred_data:
            gt_label = bbox["ground_truth"][0]
            gt_rank = bbox["ground_truth"][1]
            prediction = bbox["moth_classification"]

            # Get only the moth crops and
            if (
                gt_rank == "SPECIES"
                and gt_label != "Non-Moth"
                and gt_label != "Unidentifiable"
                and gt_label != "Unclassified"
                and gt_label not in exclusion_sp
            ):
                sp_key = sp_key_map.loc[
                    sp_key_map["species"] == gt_label, "speciesKey"
                ].values[0]
                pred_label = prediction[0][0]

                if sp_key not in species_acc.keys():
                    if gt_label == pred_label:  # Correct
                        species_acc[sp_key] = [1, 1]
                    else:  # Incorrect
                        species_acc[sp_key] = [0, 1]
                else:
                    if gt_label == pred_label:  # Correct
                        species_acc[sp_key][0] += 1
                        species_acc[sp_key][1] += 1
                    else:  # Incorrect
                        species_acc[sp_key][1] += 1

    # Add accuracy in three training buckets
    for sp_key in species_acc.keys():
        try:
            count = gbif_count[str(sp_key)]
        except KeyError:
            print(f"Species {sp_key} bucket not found.")
        accuracy = round(species_acc[sp_key][0] / species_acc[sp_key][1] * 100, 2)

        if count < 20:
            few.append(accuracy)
        elif count <= 100:
            medium.append(accuracy)
        else:
            many.append(accuracy)

    many_avg = round(sum(many) / len(many), 2)
    medium_avg = round(sum(medium) / len(medium), 2)
    few_avg = round(sum(few) / len(few), 2)
    print(f"Many avg accuracy is {many_avg} with {len(many)} classes.")
    print(f"Medium avg accuracy is {medium_avg} with {len(medium)} classes.")
    print(f"Few avg accuracy is {few_avg} with {len(few)} classes.")


if __name__ == "__main__":
    GBIF_COUNT_FILE = os.getenv("GBIF_COUNT_FILE")
    GENERAL_PREDICTION_DIR = os.getenv("GENERAL_PREDICTION_DIR")
    EXCLUSION_SPECIES_LIST = os.getenv("EXCLUSION_SPECIES_LIST")
    SPECIES_KEY_MAP = os.getenv("SPECIES_KEY_MAP")

    model = "centralamerica_vit_b_baseline_run1"
    print(f"Long-tailed accuracy for {model}.\n")

    # Try opening the files
    try:
        with open(GBIF_COUNT_FILE) as f:
            gbif_count = json.load(f)
        with open(EXCLUSION_SPECIES_LIST, "rb") as f:
            exclusion_sp = pickle.load(f)
        sp_key_map = pd.read_csv(SPECIES_KEY_MAP)
    except Exception as e:
        raise Exception(f"Error loading files: {e}")

    model_prediction_dir = os.path.join(GENERAL_PREDICTION_DIR, model)
    long_tailed_accuracy(model_prediction_dir, exclusion_sp, sp_key_map, gbif_count)
