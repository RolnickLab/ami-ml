"""
Author: Aditya Jain
Date last modified: March 6th, 2024
About: Long-tailed analysis
"""

import os
import json
import pickle
import pandas as pd

def long_tailed_accuracy(pred_dir: str, exclusion_sp: list[str], sp_key_map: pd.DataFrame, gbif_count: dict):
    """Main function for calculating accuracy in many, medium and few buckets
    """

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
            if gt_rank=="SPECIES" and gt_label!="Non-Moth" and gt_label!="Unidentifiable" \
            and gt_label!="Unclassified" and gt_label not in exclusion_sp:
                sp_key = sp_key_map.loc[sp_key_map["species"] == gt_label, "speciesKey"].values[0]
                pred_label = prediction[0][0]

                if sp_key not in species_acc.keys():
                    if gt_label == pred_label: # Correct
                        species_acc[sp_key] = [1, 1]
                    else:                      # Incorrect
                        species_acc[sp_key] = [0, 1]
                else:
                    if gt_label == pred_label: # Correct
                        species_acc[sp_key][0] += 1
                        species_acc[sp_key][1] += 1
                    else:                      # Incorrect
                        species_acc[sp_key][1] += 1

    # Add accuracy in three training buckets
    for sp_key in species_acc.keys():
        try:
            count = gbif_count[str(sp_key)]
        except:
            print(f"Species {sp_key} bucket not found.")
        accuracy = round(species_acc[sp_key][0]/species_acc[sp_key][1]*100, 2)
        
        if count < 20: few.append(accuracy)
        elif count <= 100 : medium.append(accuracy)
        else: many.append(accuracy)

    print(f"Many avg accuracy is {round(sum(many)/len(many), 2)} with {len(many)} classes.")
    print(f"Medium avg accuracy is {round(sum(medium)/len(medium), 2)} with {len(medium)} classes.")
    print(f"Few avg accuracy is {round(sum(few)/len(few), 2)} with {len(few)} classes.")

if __name__ == "__main__":
    model = "centralamerica_vit_b_baseline_run1"

    print(f"Long-tailed accuracy for {model}.\n")
    gbif_count_file = "/home/mila/a/aditya.jain/scratch/cvpr2024_data/gbif_train_counts.json"
    general_prediction_dir = "/home/mila/a/aditya.jain/scratch/cvpr2024_data/ami_traps_dataset/model_predictions/all-architectures"
    gbif_count = json.load(open(gbif_count_file))
    model_prediction_dir = os.path.join(general_prediction_dir, model)
    exclusion_sp_file = "/home/mila/a/aditya.jain/scratch/cvpr2024_data/excluded_sp_from_AMI-GBIF.pickle"
    exclusion_sp = pickle.load(open(exclusion_sp_file, "rb"))
    sp_key_map_file = "/home/mila/a/aditya.jain/scratch/cvpr2024_data/speciesKey_map.csv"
    sp_key_map = pd.read_csv(sp_key_map_file)

    long_tailed_accuracy(model_prediction_dir, exclusion_sp, sp_key_map, gbif_count)