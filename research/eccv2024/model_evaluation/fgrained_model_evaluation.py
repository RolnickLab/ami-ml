"""
Author: Aditya Jain
Date started: April 19, 2024
About: Evaluation of AMI-GBIF trained moth fine-grained classifier on AMI-Traps data
"""

import os
import glob
import json
import argparse
import pickle
from PIL import Image
import torch
from torchvision import transforms
import wandb

from model_inference import ModelInference
# from taxon_grouping import TaxonGrouping

def check_prediction(gt_label, pred_label):
    """
    Check for top1 and top5 prediction
    
    Returns 0, 1 for incorrect and correct respectively
    """
    pass


def get_higher_taxon_pred():
    """Rollup species prediction at genus and family level"""
    pass


def get_higher_taxon_gt(label, rank):
    """Get higher taxon for a ground truth label"""
    pass


def fgrained_model_evaluation(
    run_name: str,
    artifact: str,
    region: str,
    model_type: str,
    model_dir: str,
    category_map: str,
    insect_crops_dir: str,
    sp_exclusion_list: list[int]
):
    """Main function for fine-grained model evaluation"""

    # Get the environment variables
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device {device} is available.")

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
    fgrained_classifier = ModelInference(new_model, model_type, categ_map_path, device)

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
        gt_speciesKey = insect_labels[img_name]["speciesKey"]
        gt_acceptedTaxonKey = insect_labels[img_name]["acceptedTaxonKey"]
        gt_region = insect_labels[img_name]["region"]

        if gt_region == region and gt_acceptedTaxonKey!=-1 and (gt_acceptedTaxonKey not in sp_exclusion_list):
            # Fine-grained model prediction
            transform_to_tensor = transforms.Compose([transforms.ToTensor()])
            image = transform_to_tensor(image)
            sp_pred = fgrained_classifier.predict(image)
            gs_pred, fm_pred = get_higher_taxon_pred(sp_pred)

            ## TEST ##
            # higher_taxon_pred = TaxonGrouping(taxa_dict, FLAGS.taxa_hierarchy, device)

            ##########

            # Calculate species accuracy
            if gt_rank == "SPECIES":
                # Get ground truth at the labeled rank and above
                gt_label_sp = gt_label
                gt_label_gs, gt_label_fm = get_higher_taxon_gt(gt_label_sp, gt_rank)

                # Species accuracy calculation
                top1, top5 = check_prediction(gt_label_sp, sp_pred)
                sp_top1 += top1; sp_top5 += top5; sp_count += 1

                # Genus accuracy calculation
                top1, top5 = check_prediction(gt_label_gs, gs_pred)
                gs_top1 += top1; gs_top5 += top5; gs_count += 1 

                # Family accuracy calculation
                top1, top5 = check_prediction(gt_label_fm, fm_pred)
                fm_top1 += top1; fm_top5 += top5; fm_count += 1 

            # Calculate genus accuracy
            if gt_rank == "GENUS":
                # Get ground truth at the labeled rank and above
                gt_label_gs = gt_label
                gt_label_fm = get_higher_taxon_gt(gt_label_gs, gt_rank)

                # Genus accuracy calculation
                top1, top5 = check_prediction(gt_label_gs, gs_pred)
                gs_top1 += top1; gs_top5 += top5; gs_count += 1

                # Family accuracy calculation
                top1, top5 = check_prediction(gt_label_fm, fm_pred)
                fm_top1 += top1; fm_top5 += top5; fm_count += 1   

            # Calculate sub-tribe, tribe and family accuracy
            else:
                # Rollup sub-tribe and tribe to family level
                if gt_rank != "FAMILY":
                    gt_label_fm = get_higher_taxon_gt(gt_label, gt_rank)

                # Family accuracy calculation
                top1, top5 = check_prediction(gt_label_fm, fm_pred)
                fm_top1 += top1; fm_top5 += top5; fm_count += 1 

    print(
        f"\nFine-grained classification evaluation for {run_name}:\
        \nSpecies: {round(sp_top1/sp_count*100,2)}%, {round(sp_top5/sp_count*100,2)}%\
        \nGenus: {round(gs_top1/gs_count*100,2)}%, {round(gs_top5/gs_count*100,2)}%\
        \nFamily: {round(fm_top1/fm_count*100,2)}%, {round(fm_top5/fm_count*100,2)}%\
        \
        \n",
        flush=True,
    )


if __name__ == "__main__":
    run_name = "ne-america_resnet50_baseline_run1"
    artifact =  "moth-ai/ami-gbif-fine-grained/model:v13" 
    region = "NorthEasternAmerica"
    model_type = "resnet50"
    model_dir = "/home/mila/a/aditya.jain/scratch/eccv2024_data/models/fine_grained"
    category_map = "01_ami-gbif_fine-grained_ne-america_category_map.json"
    insect_crops_dir = "/home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/insect_crops"
    sp_exclusion_list_file = "/home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/metadata/ami-traps_sp_missing_in_ami-gbif.pickle"
    sp_exclusion_list = exclusion_sp = pickle.load(open(sp_exclusion_list_file, "rb"))

    fgrained_model_evaluation(run_name, artifact, region, model_type, model_dir, category_map, insect_crops_dir, sp_exclusion_list)



    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--run_name",
    #     help="Run name of the model on Weights and Biases.",
    #     required=True,
    # )
    # parser.add_argument(
    #     "--wandb_model_artifact",
    #     help="Model artifact on Weights and Biases.",
    #     required=True,
    # )
    # parser.add_argument(
    #     "--model_type",
    #     help="Model type of the binary classifier.",
    #     required=True,
    # )
    # parser.add_argument(
    #     "--model_dir",
    #     help="Model directory where the binary models are downloaded.",
    #     required=True,
    # )
    # parser.add_argument(
    #     "--category_map",
    #     help="Category map for the binary classifier.",
    #     required=True,
    # )
    # parser.add_argument(
    #     "--insect_crops_dir",
    #     help="Directory containing the insect crops.",
    #     required=True,
    # )
    # parser.add_argument(
    #     "--skip_small_crops",
    #     help="Whether to skip crops below a certain size.",
    #     required=True,
    # )
    # parser.add_argument(
    #     "--min_crop_dim",
    #     help="Minimum crop length in pixels to consider for prediction.",
    #     type=int,
    # )

    # args = parser.parse_args()
    # binary_model_evaluation(
    #     args.run_name,
    #     args.wandb_model_artifact,
    #     args.model_type,
    #     args.model_dir,
    #     args.category_map,
    #     args.insect_crops_dir,
    #     args.skip_small_crops,
    #     args.min_crop_dim,
    # )
