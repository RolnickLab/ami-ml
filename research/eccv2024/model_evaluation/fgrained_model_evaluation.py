"""
Author: Aditya Jain
Date started: April 19, 2024
About: Evaluation of AMI-GBIF trained moth fine-grained classifier on AMI-Traps data
"""

import os
import glob
import json
import pandas as pd
import pickle
from PIL import Image
import torch
from torchvision import transforms
import wandb
import typer

from model_inference import ModelInference


def check_prediction(gt_label, pred_label):
    """
    Check for top1 and top5 prediction
    
    Returns 0, 1 for incorrect and correct respectively
    """
    
    # Variable definitions
    top1, top5 = 0, 0

    # Accuracy calculation
    top5_pred = pred_label[:5]
    top5_labels = [item[0] for item in top5_pred]
    if gt_label in top5_labels[:1]: top1 = 1
    if gt_label in top5_labels[:5]: top5 = 1

    return top1, top5


def get_higher_taxon_pred(sp_pred: list[list[str, float]], taxonomy_map: pd.DataFrame, gbif_taxonomy_hierarchy: dict):
    """Rollup model species prediction at genus and family level"""
    
    # Variables definitions
    genus_pred, family_pred = {}, {}

    for prediction in sp_pred:
        # Get family and genus name for every species prediction
        sp_key, conf = prediction[0], round(float(prediction[1]), 3)
        genus = gbif_taxonomy_hierarchy[sp_key][0]
        family = gbif_taxonomy_hierarchy[sp_key][1]

        # Genus accuracy calculation
        if genus not in genus_pred.keys(): genus_pred[genus] = conf
        else: genus_pred[genus] += conf

        # Family accuracy calculation
        if family not in family_pred.keys(): family_pred[family] = conf
        else: family_pred[family] += conf

    # Sort the prediction in decreasing order of confidence
    genus_pred_sorted = [[taxa, round(conf, 3)] for taxa, conf in sorted(list(genus_pred.items()), key=lambda item: item[1], reverse=True)]
    family_pred_sorted = [[taxa, round(conf, 3)] for taxa, conf in sorted(list(family_pred.items()), key=lambda item: item[1], reverse=True)]

    return genus_pred_sorted, family_pred_sorted



def get_higher_taxon_gt(label: str, rank: str, taxonomy_map: pd.DataFrame):
    """Get higher taxon for a ground truth label"""
    
    if rank == "SPECIES":
        try: 
            query = taxonomy_map.loc[taxonomy_map["name"] == label]
            return query["GENUS"].values[0], query["FAMILY"].values[0]
        except: 
            print(f"{label} of rank {rank} not found in the taxonomy database.")

    else:
        try: 
            query = taxonomy_map.loc[taxonomy_map["name"] == label]
            return query["FAMILY"].values[0]
        except: 
            print(f"{label} of rank {rank} not found in the taxonomy database.")



def fgrained_model_evaluation(
    run_name: str,
    artifact: str,
    region: str,
    model_type: str,
    model_dir: str,
    category_map: str,
    insect_crops_dir: str,
    sp_exclusion_list_file: str,
    ami_traps_taxonomy_map_file: str,
    ami_gbif_taxonomy_map_file: str,
    gbif_taxonomy_hierarchy_file: str
):
    """Main function for fine-grained model evaluation"""

    # Get the environment variable and other files
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device {device} is available.")
    sp_exclusion_list = pickle.load(open(sp_exclusion_list_file, "rb"))
    ami_traps_taxonomy_map = pd.read_csv(ami_traps_taxonomy_map_file)
    ami_gbif_taxonomy_map = pd.read_csv(ami_gbif_taxonomy_map_file)
    gbif_taxonomy_hierarchy = json.load(open(gbif_taxonomy_hierarchy_file))

    # Download the model
    api = wandb.Api()
    artifact = api.artifact(artifact)
    artifact.download(root=model_dir)

    # Change downloaded model name to the run name
    files = glob.glob(os.path.join(model_dir, "*"))
    latest_file = max(files, key=os.path.getctime)
    new_model = os.path.join(model_dir, run_name + ".pth")
    os.rename(latest_file, new_model)

    # Build the fine-grained classification model
    categ_map_path = os.path.join(model_dir, category_map)
    fgrained_classifier = ModelInference(new_model, model_type, categ_map_path, device, topk=0)

    # Get all moth insect crops label information
    insect_labels = json.load(
        open(os.path.join(insect_crops_dir, "fgrained_labels.json"))
    )

    # Evaluation metrics variables
    sp_top1, sp_top5 = 0, 0
    gs_top1, gs_top5 = 0, 0
    fm_top1, fm_top5 = 0, 0
    sp_count, gs_count, fm_count = 0, 0, 0

    # Iterate over each moth crop
    for img_name in insect_labels.keys():
        # Read the image
        image = Image.open(os.path.join(insect_crops_dir, img_name))        

        # Get ground truth label information
        gt_label = insect_labels[img_name]["label"]
        gt_rank = insect_labels[img_name]["taxon_rank"]
        gt_acceptedTaxonKey = insect_labels[img_name]["acceptedTaxonKey"]
        gt_region = insect_labels[img_name]["region"]

        if gt_rank != "ORDER" and gt_region == region and gt_acceptedTaxonKey!=-1 and (gt_acceptedTaxonKey not in sp_exclusion_list):
            # Fine-grained model prediction
            transform_to_tensor = transforms.Compose([transforms.ToTensor()])
            image = transform_to_tensor(image)
            sp_pred = fgrained_classifier.predict(image)

            # Get rolled up predictions to genus and family level
            gs_pred, fm_pred = get_higher_taxon_pred(sp_pred, ami_gbif_taxonomy_map, gbif_taxonomy_hierarchy)

            # Calculate species accuracy
            if gt_rank == "SPECIES":
                # Get ground truth at the labeled rank and above
                gt_label_sp = gt_label
                gt_label_gs, gt_label_fm = get_higher_taxon_gt(gt_label_sp, gt_rank, ami_traps_taxonomy_map)

                # Species accuracy calculation
                top1, top5 = check_prediction(str(gt_acceptedTaxonKey), sp_pred)
                sp_top1 += top1; sp_top5 += top5; sp_count += 1

                # Genus accuracy calculation
                top1, top5 = check_prediction(gt_label_gs, gs_pred)
                gs_top1 += top1; gs_top5 += top5; gs_count += 1 

                # Family accuracy calculation
                top1, top5 = check_prediction(gt_label_fm, fm_pred)
                fm_top1 += top1; fm_top5 += top5; fm_count += 1 

            # Calculate genus accuracy
            elif gt_rank == "GENUS":
                # Get ground truth at the labeled rank and above
                gt_label_gs = gt_label
                gt_label_fm = get_higher_taxon_gt(gt_label_gs, gt_rank, ami_traps_taxonomy_map)
        
                # Genus accuracy calculation
                top1, top5 = check_prediction(gt_label_gs.split(" ")[0], gs_pred)
                gs_top1 += top1; gs_top5 += top5; gs_count += 1

                # Family accuracy calculation
                top1, top5 = check_prediction(gt_label_fm, fm_pred)
                fm_top1 += top1; fm_top5 += top5; fm_count += 1   

            # Calculate sub-tribe, tribe and family accuracy
            else:
                # Rollup sub-tribe and tribe to family level
                if gt_rank != "FAMILY":
                    gt_label_fm = get_higher_taxon_gt(gt_label, gt_rank, ami_traps_taxonomy_map)
                else:
                    gt_label_fm = gt_label

                # Family accuracy calculation
                top1, top5 = check_prediction(gt_label_fm, fm_pred)
                fm_top1 += top1; fm_top5 += top5; fm_count += 1 

    print(
        f"\nFine-grained classification evaluation for {run_name}:\
        \nSpecies: {round(sp_top1/sp_count*100,2)}%, {round(sp_top5/sp_count*100,2)}%\
        \nGenus: {round(gs_top1/gs_count*100,2)}%, {round(gs_top5/gs_count*100,2)}%\
        \nFamily: {round(fm_top1/fm_count*100,2)}%, {round(fm_top5/fm_count*100,2)}%\
        \n",
        flush=True,
    )


if __name__ == "__main__":
    typer.run(fgrained_model_evaluation)
