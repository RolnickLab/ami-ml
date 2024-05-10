"""
Author: Aditya Jain
Date started: April 19, 2024
About: Evaluation of AMI-GBIF trained moth fine-grained classifier on AMI-Traps data
"""

import os
import glob
import json
import math
import pandas as pd
import pickle
from PIL import Image
import torch
from torchvision import transforms
import wandb
import typer

from model_inference import ModelInference


def check_prediction(gt_label: str, pred_label: list[list[str, float]]):
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



def get_higher_taxon_pred(
    sp_pred: list[list[str, float]], 
    gbif_taxonomy_hierarchy: dict
):
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



def accuracy_versus_confidence(
    gt_label: str, 
    pred_label: list[list[str, float]], 
    accuracy_vs_conf_df: pd.DataFrame
):
    """Calculate accuracy alongside model prediction confidence"""

    top1_class, top1_acc = pred_label[0][0], math.floor(pred_label[0][1]*100)
    thresh = 0 # for cases with confidence < 10%

    for thresh in range(10, top1_acc, 10):  
        # Add the counts upto the threshold  
        accuracy_vs_conf_df.loc["conf-total"]["conf-"+str(thresh)] += 1

        # Add only correct counts
        if top1_class == gt_label: 
            accuracy_vs_conf_df.loc["correct"]["conf-"+str(thresh)] += 1

    # Add counts for upwards rejected thresholds
    for upwards_thresh in range(thresh+10, 100, 10):
        accuracy_vs_conf_df.loc["reject"]["conf-"+str(upwards_thresh)] += 1

    return accuracy_vs_conf_df


def update_taxa_accuracy(macro_acc: dict, top1: int, top5: int, gt_label: str, gt_rank: str):
    """Update accuracy data for every class at every taxonomic level"""

    # Every value in each taxa key is [top1_correct, top5_correct, total]

    # Make a key in the dict if the taxa seen first time
    if gt_label not in macro_acc[gt_rank].keys():
        macro_acc[gt_rank][gt_label] = [0, 0, 0]

    # Update the accuracy numbers
    macro_acc[gt_rank][gt_label][0] += top1
    macro_acc[gt_rank][gt_label][1] += top5
    macro_acc[gt_rank][gt_label][2] += 1

    return macro_acc



def calculate_macro_accuracy(macro_acc_taxa: dict):
    """Calculate macro accuracy at the given taxonomic level"""
    
    # Lists containing each class' top1 and top5 accuracy
    top1_acc, top5_acc = [], []

    # Calculate the individual taxa accuracy
    for taxa in macro_acc_taxa.keys():
        top1_acc.append(macro_acc_taxa[taxa][0]/macro_acc_taxa[taxa][2])
        top5_acc.append(macro_acc_taxa[taxa][1]/macro_acc_taxa[taxa][2])

    # Calculate the macro accuracy
    macro_top1_acc = round(sum(top1_acc)/len(top1_acc)*100, 2)
    macro_top5_acc = round(sum(top5_acc)/len(top5_acc)*100, 2)

    return macro_top1_acc, macro_top5_acc



def fgrained_model_evaluation(
    run_name: str = typer.Option(),
    artifact: str = typer.Option(),
    region: str = typer.Option(),
    model_type: str = typer.Option(),
    model_dir: str = typer.Option(),
    category_map: str = typer.Option(),
    insect_crops_dir: str = typer.Option(),
    sp_exclusion_list_file: str = typer.Option(),
    ami_traps_taxonomy_map_file: str = typer.Option(),
    gbif_taxonomy_hierarchy_file: str = typer.Option(),
    save_acccuracy_vs_conf: bool = False
):
    """Main function for fine-grained model evaluation"""

    # Get the environment variable and other files
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device {device} is available.")
    sp_exclusion_list = pickle.load(open(sp_exclusion_list_file, "rb"))
    ami_traps_taxonomy_map = pd.read_csv(ami_traps_taxonomy_map_file)
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

    # Micro-accuracy evaluation metrics 
    sp_top1, sp_top5 = 0, 0
    gs_top1, gs_top5 = 0, 0
    fm_top1, fm_top5 = 0, 0
    sp_count, gs_count, fm_count = 0, 0, 0

    # Macro-accuracy evaluation metrics
    macro_acc = {}
    macro_acc["SPECIES"], macro_acc["GENUS"], macro_acc["FAMILY"] = {}, {}, {}
    # Every value in each taxa key will be [top1_correct, top5_correct, total]

    # Accuracy v/s confidence variables 
    accuracy_w_conf_sp = pd.DataFrame(0, columns=["conf-"+str(thresh) for thresh in range(10, 100, 10)], index=["correct", "conf-total", "reject", "moths-total"])
    accuracy_w_conf_gs = pd.DataFrame(0, columns=["conf-"+str(thresh) for thresh in range(10, 100, 10)], index=["correct", "conf-total", "reject", "moths-total"])
    accuracy_w_conf_fm = pd.DataFrame(0, columns=["conf-"+str(thresh) for thresh in range(10, 100, 10)], index=["correct", "conf-total", "reject", "moths-total"])

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
            gs_pred, fm_pred = get_higher_taxon_pred(sp_pred, gbif_taxonomy_hierarchy)

            # Calculate species accuracy
            if gt_rank == "SPECIES":
                # Get ground truth at the labeled rank and above
                gt_label_sp = gt_label
                gt_label_gs, gt_label_fm = get_higher_taxon_gt(gt_label_sp, gt_rank, ami_traps_taxonomy_map)

                # Species accuracy calculation
                top1, top5 = check_prediction(str(gt_acceptedTaxonKey), sp_pred)
                sp_top1 += top1; sp_top5 += top5; sp_count += 1
                accuracy_w_conf_sp = accuracy_versus_confidence(str(gt_acceptedTaxonKey), sp_pred, accuracy_w_conf_sp)
                macro_acc = update_taxa_accuracy(macro_acc, top1, top5, gt_label_sp, "SPECIES")

                # Genus accuracy calculation
                top1, top5 = check_prediction(gt_label_gs, gs_pred)
                gs_top1 += top1; gs_top5 += top5; gs_count += 1 
                accuracy_w_conf_gs = accuracy_versus_confidence(gt_label_gs, gs_pred, accuracy_w_conf_gs)
                macro_acc = update_taxa_accuracy(macro_acc, top1, top5, gt_label_gs, "GENUS")

                # Family accuracy calculation
                top1, top5 = check_prediction(gt_label_fm, fm_pred)
                fm_top1 += top1; fm_top5 += top5; fm_count += 1 
                accuracy_w_conf_fm = accuracy_versus_confidence(gt_label_fm, fm_pred, accuracy_w_conf_fm)
                macro_acc = update_taxa_accuracy(macro_acc, top1, top5, gt_label_fm, "FAMILY")


            # Calculate genus accuracy
            elif gt_rank == "GENUS":
                # Get ground truth at the labeled rank and above
                gt_label_gs = gt_label
                gt_label_fm = get_higher_taxon_gt(gt_label_gs, gt_rank, ami_traps_taxonomy_map)
        
                # Genus accuracy calculation
                top1, top5 = check_prediction(gt_label_gs.split(" ")[0], gs_pred)
                gs_top1 += top1; gs_top5 += top5; gs_count += 1
                accuracy_w_conf_gs = accuracy_versus_confidence(gt_label_gs.split(" ")[0], gs_pred, accuracy_w_conf_gs)
                macro_acc = update_taxa_accuracy(macro_acc, top1, top5, gt_label_gs.split(" ")[0], "GENUS")

                # Family accuracy calculation
                top1, top5 = check_prediction(gt_label_fm, fm_pred)
                fm_top1 += top1; fm_top5 += top5; fm_count += 1 
                accuracy_w_conf_fm = accuracy_versus_confidence(gt_label_fm, fm_pred, accuracy_w_conf_fm) 
                macro_acc = update_taxa_accuracy(macro_acc, top1, top5, gt_label_fm, "FAMILY") 

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
                accuracy_w_conf_fm = accuracy_versus_confidence(gt_label_fm, fm_pred, accuracy_w_conf_fm)
                macro_acc = update_taxa_accuracy(macro_acc, top1, top5, gt_label_fm, "FAMILY")

    # Calculate the macro-accuracy at each taxonomic level
    macro_sp_top1, macro_sp_top5 = calculate_macro_accuracy(macro_acc["SPECIES"])
    macro_gs_top1, macro_gs_top5 = calculate_macro_accuracy(macro_acc["GENUS"])
    macro_fm_top1, macro_fm_top5 = calculate_macro_accuracy(macro_acc["FAMILY"])


    print(
        f"\nFine-grained classification micro-accuracy (Top1, Top5) for {run_name}:\
        \nSpecies: {round(sp_top1/sp_count*100,2)}%, {round(sp_top5/sp_count*100,2)}%\
        \nGenus: {round(gs_top1/gs_count*100,2)}%, {round(gs_top5/gs_count*100,2)}%\
        \nFamily: {round(fm_top1/fm_count*100,2)}%, {round(fm_top5/fm_count*100,2)}%\
        \n\
        \nFine-grained classification macro-accuracy (Top1, Top5) for {run_name}:\
        \nSpecies: {macro_sp_top1}%, {macro_sp_top5}%\
        \nGenus: {macro_gs_top1}%, {macro_gs_top5}%\
        \nFamily: {macro_fm_top1}%, {macro_fm_top5}%\
        \n",
        flush=True,
    )

    # Save accuracy v/s confidence data
    if save_acccuracy_vs_conf:
        for thresh in range(10, 100, 10):
            accuracy_w_conf_sp.loc["moths-total"]["conf-"+str(thresh)] = sp_count
            accuracy_w_conf_gs.loc["moths-total"]["conf-"+str(thresh)] = gs_count
            accuracy_w_conf_fm.loc["moths-total"]["conf-"+str(thresh)] = fm_count
        accuracy_w_conf_sp.to_csv(os.path.join("./plots", run_name + "-sp_acc_rej_vs_conf.csv"))
        accuracy_w_conf_gs.to_csv(os.path.join("./plots", run_name + "-gs_acc_rej_vs_conf.csv"))
        accuracy_w_conf_fm.to_csv(os.path.join("./plots", run_name + "-fm_acc_rej_vs_conf.csv"))



if __name__ == "__main__":
    # run_name = "ne-america_resnet50_baseline_run1"
    # artifact =  "moth-ai/ami-gbif-fine-grained/model:v13" 
    # region = "NorthEasternAmerica"
    # model_type = "resnet50"
    # model_dir = "/home/mila/a/aditya.jain/scratch/eccv2024_data/models/fine_grained"
    # category_map = "01_ami-gbif_fine-grained_ne-america_category_map.json"
    # insect_crops_dir = "/home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/insect_crops"
    # sp_exclusion_list_file = "/home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/metadata/ami-traps_sp_missing_in_ami-gbif.pickle"    
    # ami_traps_taxonomy_map_file = "/home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/metadata/ami-traps_taxonomy_map.csv"
    # gbif_taxonomy_hierarchy_file = "/home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/metadata/gbif_taxonomy_hierarchy.json"

    # fgrained_model_evaluation(run_name, artifact, region, model_type, model_dir, category_map, insect_crops_dir, sp_exclusion_list_file, ami_traps_taxonomy_map_file, gbif_taxonomy_hierarchy_file)

    typer.run(fgrained_model_evaluation)
