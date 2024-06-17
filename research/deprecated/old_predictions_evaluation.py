# *** DEPRECATED; DO NOT USE AS IS; FOR REFERENCE ONLY ***

"""
Author: Aditya Jainn
Date last modified: November 28, 2023
About: Evaluate GBIF-trained models on AMI benchmark data
"""

import copy
import json
import math
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Load secrets and config from optional .env file
load_dotenv()


def save_insect_crop(bbox: list[str], image_pred_file: str, data_dir: str, label: str):
    """Save the insect crop image on the disk"""

    # Directories for loading and storing images
    image_dir = os.path.join(data_dir, "ami_traps_dataset", "images")
    save_dir = os.path.join(data_dir, "binary_misclassifications", label)

    # Parse the image name
    image_name = image_pred_file.split("_")[0] + ".jpg"

    # Read the raw image
    try:
        raw_image = Image.open(os.path.join(image_dir, image_name))
        img_width, img_height = raw_image.size
    except:
        raise Exception(f"Issue with image {image_name}")

    # Convert the raw image to tensor
    transform_totensor = transforms.Compose([transforms.ToTensor()])
    try:
        image = transform_totensor(raw_image)
    except OSError as e:
        print(f"Error {e} with image {image_name}")

    # Get the insect crop and save on the disk
    x, y = float(bbox[0]), float(bbox[1])
    w, h = float(bbox[2]), float(bbox[3])
    x_start = int((x - w / 2) * img_width)
    y_start = int((y - h / 2) * img_height)
    w_px, h_px = int(w * img_width), int(h * img_height)
    cropped_image = image[:, y_start : y_start + h_px, x_start : x_start + w_px]
    crop_name = str(len(os.listdir(save_dir)) + 1) + ".jpg"
    save_image(cropped_image, os.path.join(save_dir, crop_name))


def check_crop_size(
    bbox: list[str], image_pred_file: str, data_dir: str, min_crop_dim: int
):
    """Check if crop size is too small for model analysis"""

    # Get the metadata directory and file file
    metadata_dir = os.path.join(data_dir, "ami_traps_dataset", "metadata")
    img_basename = image_pred_file.split("_")[0]
    metadata_file = os.path.join(metadata_dir, img_basename + ".json")

    # Get raw image dimensions
    metadata = json.load(open(metadata_file))
    img_width, img_height = metadata["image_width"], metadata["image_height"]

    # Get the insect crop dimensions
    w, h = float(bbox[2]), float(bbox[3])
    w_px, h_px = int(w * img_width), int(h * img_height)

    # Both dimensions should be less than threshold
    if w_px < min_crop_dim and h_px < min_crop_dim:  # skip this crop
        return True
    else:
        return False


def get_higher_taxon(label: str, rank: str, taxonomy_db: pd.DataFrame):
    """Returns higher taxon level names for a given taxa name"""

    if rank == "SPECIES":
        try:
            query = taxonomy_db.loc[taxonomy_db["name"] == label]
            return query["GENUS"].values[0] + " sp.", query["FAMILY"].values[0]
        except:
            print(f"{label} of rank {rank} not found in the taxonomy database.")

    else:
        try:
            query = taxonomy_db.loc[taxonomy_db["name"] == label]
            return query["FAMILY"].values[0]
        except:
            print(f"{label} of rank {rank} not found in the taxonomy database.")


def count_crops_at_taxon_level(label: str, rank: str, exclusion_sp: list[str]):
    """Count the moth crops at the three taxon levels"""

    global skip_crops_excluded, gt_moths, gt_family, gt_genus, gt_species

    if rank == "ORDER":
        gt_moths += 1
    elif rank == "SPECIES" and label in exclusion_sp:
        skip_crops_excluded += 1
    elif rank == "SPECIES" and label not in exclusion_sp:
        gt_family += 1
        gt_genus += 1
        gt_species += 1
        gt_moths += 1
    elif rank == "GENUS":
        gt_family += 1
        gt_genus += 1
        gt_moths += 1
    else:
        gt_family += 1
        gt_moths += 1


def histogram_at_taxon_level(
    label: str, rank: str, region: str, taxonomy_db: str, exclusion_sp: list[str]
):
    """Calculations for moth crop counts for various classes at diff. taxon levels"""

    global hist_dict
    hierarchy = ["SPECIES", "GENUS", "FAMILY"]
    global neamerica_family, neamerica_genus, neamerica_species, camerica_family, camerica_genus, camerica_species, weurope_family, weurope_genus, weurope_species

    if rank == "ORDER":
        pass

    elif rank == "SPECIES":
        if label not in exclusion_sp:
            genus, family = get_higher_taxon(label, rank, taxonomy_db)
            species = label
            label_hierarchy = [species, genus, family]
            for i in range(len(label_hierarchy)):
                if label_hierarchy[i] not in hist_dict[hierarchy[i]].keys():
                    hist_dict[hierarchy[i]][label_hierarchy[i]] = 1
                else:
                    hist_dict[hierarchy[i]][label_hierarchy[i]] += 1

            # Count region-wise statistics
            if region == "NorthEasternAmerica":
                if species not in neamerica_species:
                    neamerica_species.append(species)
                if genus not in neamerica_genus:
                    neamerica_genus.append(genus)
                if family not in neamerica_family:
                    neamerica_family.append(family)
            if region == "WesternEurope":
                if species not in weurope_species:
                    weurope_species.append(species)
                if genus not in weurope_genus:
                    weurope_genus.append(genus)
                if family not in weurope_family:
                    weurope_family.append(family)
            if region == "CentralAmerica":
                if species not in camerica_species:
                    camerica_species.append(species)
                if genus not in camerica_genus:
                    camerica_genus.append(genus)
                if family not in camerica_family:
                    camerica_family.append(family)

    elif rank == "GENUS":
        family = get_higher_taxon(label, rank, taxonomy_db)
        genus = label
        label_hierarchy = [genus, family]
        for i in range(len(label_hierarchy)):
            if label_hierarchy[i] not in hist_dict[hierarchy[i + 1]].keys():
                hist_dict[hierarchy[i + 1]][label_hierarchy[i]] = 1
            else:
                hist_dict[hierarchy[i + 1]][label_hierarchy[i]] += 1

        # Count region-wise statistics
        if region == "NorthEasternAmerica":
            if genus not in neamerica_genus:
                neamerica_genus.append(genus)
            if family not in neamerica_family:
                neamerica_family.append(family)
        if region == "WesternEurope":
            if genus not in weurope_genus:
                weurope_genus.append(genus)
            if family not in weurope_family:
                weurope_family.append(family)
        if region == "CentralAmerica":
            if genus not in camerica_genus:
                camerica_genus.append(genus)
            if family not in camerica_family:
                camerica_family.append(family)

    else:
        if rank != "FAMILY":
            label = get_higher_taxon(label, rank, taxonomy_db)

        if label not in hist_dict[hierarchy[-1]].keys():
            hist_dict[hierarchy[-1]][label] = 1
        else:
            hist_dict[hierarchy[-1]][label] += 1

        # Count region-wise statistics
        family = label
        if region == "NorthEasternAmerica":
            if family not in neamerica_family:
                neamerica_family.append(family)
        if region == "WesternEurope":
            if family not in weurope_family:
                weurope_family.append(family)
        if region == "CentralAmerica":
            if family not in camerica_family:
                camerica_family.append(family)


def _helper_finegrained_accuracy(
    ground_truth: str,
    rank: str,
    prediction: list[list[str, str]],
    region: str,
    species_list: pd.DataFrame,
    taxonomy_db: pd.DataFrame,
    exclusion_sp: list[str],
):
    """Calculate fine-grained classification accuracy for diff taxon levels"""

    global sp_top1, sp_top5, sp_top10, gs_top1, gs_top5, gs_top10, fm_top1, fm_top5, fm_top10

    # Region-wise global variables for accuracy
    global gt_family_neamerica, gt_genus_neamerica, gt_species_neamerica, neamerica_sp_top1, neamerica_sp_top5, neamerica_gs_top1, neamerica_gs_top5, neamerica_fm_top1, neamerica_fm_top5, gt_family_weurope, gt_genus_weurope, gt_species_weurope, weurope_sp_top1, weurope_sp_top5, weurope_gs_top1, weurope_gs_top5, weurope_fm_top1, weurope_fm_top5, gt_family_camerica, gt_genus_camerica, gt_species_camerica, camerica_sp_top1, camerica_sp_top5, camerica_gs_top1, camerica_gs_top5, camerica_fm_top1, camerica_fm_top5

    if rank == "ORDER":
        pass

    elif rank == "SPECIES":
        if ground_truth not in exclusion_sp:
            sp_pred_list = []
            gs_pred_list = []
            fm_pred_list = []
            gt_genus, gt_family = get_higher_taxon(ground_truth, rank, taxonomy_db)
            for pred in prediction:
                genus = species_list.loc[species_list["gbif_species"] == pred[0]][
                    "genus"
                ].values[0]
                family = species_list.loc[species_list["gbif_species"] == pred[0]][
                    "family"
                ].values[0]
                sp_pred_list.append(pred[0])
                gs_pred_list.append(genus + " sp.")
                fm_pred_list.append(family)
            synonym_list = [
                species_list.loc[species_list["gbif_species"] == species][
                    "search_species"
                ].values[0]
                for species in sp_pred_list
            ]  # Get the synonym names

            # Add region-wise moth numbers
            if region == "NorthEasternAmerica":
                gt_species_neamerica += 1
                gt_genus_neamerica += 1
                gt_family_neamerica += 1
            if region == "WesternEurope":
                gt_species_weurope += 1
                gt_genus_weurope += 1
                gt_family_weurope += 1
            if region == "CentralAmerica":
                gt_species_camerica += 1
                gt_genus_camerica += 1
                gt_family_camerica += 1

            # Specis calculation
            if ground_truth in sp_pred_list[:1] or ground_truth in synonym_list[:1]:
                sp_top1 += 1
                if region == "NorthEasternAmerica":
                    neamerica_sp_top1 += 1
                if region == "WesternEurope":
                    weurope_sp_top1 += 1
                if region == "CentralAmerica":
                    camerica_sp_top1 += 1
            if ground_truth in sp_pred_list[:5] or ground_truth in synonym_list[:5]:
                sp_top5 += 1
                if region == "NorthEasternAmerica":
                    neamerica_sp_top5 += 1
                if region == "WesternEurope":
                    weurope_sp_top5 += 1
                if region == "CentralAmerica":
                    camerica_sp_top5 += 1
            if ground_truth in sp_pred_list[:10] or ground_truth in synonym_list[:10]:
                sp_top10 += 1

            # Genus calculation
            if gt_genus in gs_pred_list[:1]:
                gs_top1 += 1
                if region == "NorthEasternAmerica":
                    neamerica_gs_top1 += 1
                if region == "WesternEurope":
                    weurope_gs_top1 += 1
                if region == "CentralAmerica":
                    camerica_gs_top1 += 1
            if gt_genus in gs_pred_list[:5]:
                gs_top5 += 1
                if region == "NorthEasternAmerica":
                    neamerica_gs_top5 += 1
                if region == "WesternEurope":
                    weurope_gs_top5 += 1
                if region == "CentralAmerica":
                    camerica_gs_top5 += 1
            if gt_genus in gs_pred_list[:10]:
                gs_top10 += 1

            # Family calculation
            if gt_family in fm_pred_list[:1]:
                fm_top1 += 1
                if region == "NorthEasternAmerica":
                    neamerica_fm_top1 += 1
                if region == "WesternEurope":
                    weurope_fm_top1 += 1
                if region == "CentralAmerica":
                    camerica_fm_top1 += 1
            if gt_family in fm_pred_list[:5]:
                fm_top5 += 1
                if region == "NorthEasternAmerica":
                    neamerica_fm_top5 += 1
                if region == "WesternEurope":
                    weurope_fm_top5 += 1
                if region == "CentralAmerica":
                    camerica_fm_top5 += 1
            if gt_family in fm_pred_list[:10]:
                fm_top10 += 1

    elif rank == "GENUS":
        gs_pred_list = []
        fm_pred_list = []
        gt_family = get_higher_taxon(ground_truth, rank, taxonomy_db)
        for pred in prediction:
            genus = species_list.loc[species_list["gbif_species"] == pred[0]][
                "genus"
            ].values[0]
            family = species_list.loc[species_list["gbif_species"] == pred[0]][
                "family"
            ].values[0]
            gs_pred_list.append(genus + " sp.")
            fm_pred_list.append(family)

        # Add region-wise moth numbers
        if region == "NorthEasternAmerica":
            gt_genus_neamerica += 1
            gt_family_neamerica += 1
        if region == "WesternEurope":
            gt_genus_weurope += 1
            gt_family_weurope += 1
        if region == "CentralAmerica":
            gt_genus_camerica += 1
            gt_family_camerica += 1

        # Genus calculation
        if ground_truth in gs_pred_list[:1]:
            gs_top1 += 1
            if region == "NorthEasternAmerica":
                neamerica_gs_top1 += 1
            if region == "WesternEurope":
                weurope_gs_top1 += 1
            if region == "CentralAmerica":
                camerica_gs_top1 += 1
        if ground_truth in gs_pred_list[:5]:
            gs_top5 += 1
            if region == "NorthEasternAmerica":
                neamerica_gs_top5 += 1
            if region == "WesternEurope":
                weurope_gs_top5 += 1
            if region == "CentralAmerica":
                camerica_gs_top5 += 1
        if ground_truth in gs_pred_list[:10]:
            gs_top10 += 1

        # Family calculation
        if gt_family in fm_pred_list[:1]:
            fm_top1 += 1
            if region == "NorthEasternAmerica":
                neamerica_fm_top1 += 1
            if region == "WesternEurope":
                weurope_fm_top1 += 1
            if region == "CentralAmerica":
                camerica_fm_top1 += 1
        if gt_family in fm_pred_list[:5]:
            fm_top5 += 1
            if region == "NorthEasternAmerica":
                neamerica_fm_top5 += 1
            if region == "WesternEurope":
                weurope_fm_top5 += 1
            if region == "CentralAmerica":
                camerica_fm_top5 += 1
        if gt_family in fm_pred_list[:10]:
            fm_top10 += 1

    else:
        if rank != "FAMILY":
            ground_truth = get_higher_taxon(ground_truth, rank, taxonomy_db)
        fm_pred_list = []
        for pred in prediction:
            family = species_list.loc[species_list["gbif_species"] == pred[0]][
                "family"
            ].values[0]
            fm_pred_list.append(family)

        # Add region-wise moth numbers
        if region == "NorthEasternAmerica":
            gt_family_neamerica += 1
        if region == "WesternEurope":
            gt_family_weurope += 1
        if region == "CentralAmerica":
            gt_family_camerica += 1

        # Family calculation
        if ground_truth in fm_pred_list[:1]:
            fm_top1 += 1
            if region == "NorthEasternAmerica":
                neamerica_fm_top1 += 1
            if region == "WesternEurope":
                weurope_fm_top1 += 1
            if region == "CentralAmerica":
                camerica_fm_top1 += 1
        if ground_truth in fm_pred_list[:5]:
            fm_top5 += 1
            if region == "NorthEasternAmerica":
                neamerica_fm_top5 += 1
            if region == "WesternEurope":
                weurope_fm_top5 += 1
            if region == "CentralAmerica":
                camerica_fm_top5 += 1
        if ground_truth in fm_pred_list[:10]:
            fm_top10 += 1


def _helper_combine_pred(prediction: list[list[str, str]]):
    """Helper function to combine predictions and their confidences
    at the genus and family level
    """
    pred_dict = {}

    for pred in prediction:
        if pred[0] not in pred_dict.keys():
            pred_dict[pred[0]] = float(pred[1])
        else:
            pred_dict[pred[0]] += float(pred[1])

    return [
        [sp, conf]
        for sp, conf in sorted(
            list(pred_dict.items()), key=lambda item: item[1], reverse=True
        )
    ]


def _helper_acc_alongside_conf(
    ground_truth: str, prediction: list[list[str, str]], rank: str
):
    """Helper function for accuracy_alongside_confidence"""
    global accuracy_w_conf_sp, accuracy_w_conf_gs, accuracy_w_conf_fm

    top1_class, top1_acc = prediction[0][0], math.floor(float(prediction[0][1]) * 100)
    thresh = 0  # for cases with confidence < 10%

    for thresh in range(10, top1_acc, 10):
        # Add the counts upto the threshold
        if rank == "SPECIES":
            accuracy_w_conf_sp.loc["conf-total"]["conf-" + str(thresh)] += 1
        if rank == "GENUS":
            accuracy_w_conf_gs.loc["conf-total"]["conf-" + str(thresh)] += 1
        if rank == "FAMILY":
            accuracy_w_conf_fm.loc["conf-total"]["conf-" + str(thresh)] += 1

        # Add only correct counts
        if top1_class == ground_truth:
            if rank == "SPECIES":
                accuracy_w_conf_sp.loc["correct"]["conf-" + str(thresh)] += 1
            if rank == "GENUS":
                accuracy_w_conf_gs.loc["correct"]["conf-" + str(thresh)] += 1
            if rank == "FAMILY":
                accuracy_w_conf_fm.loc["correct"]["conf-" + str(thresh)] += 1

    # Add counts for upwards rejected thresholds
    for upwards_thresh in range(thresh + 10, 100, 10):
        if rank == "SPECIES":
            accuracy_w_conf_sp.loc["reject"]["conf-" + str(upwards_thresh)] += 1
        if rank == "GENUS":
            accuracy_w_conf_gs.loc["reject"]["conf-" + str(upwards_thresh)] += 1
        if rank == "FAMILY":
            accuracy_w_conf_fm.loc["reject"]["conf-" + str(upwards_thresh)] += 1


def accuracy_alongside_confidence(
    ground_truth: str, rank: str, prediction: list[list[str, str]], taxonomy_db: str
):
    """Calculate classification accuracy as a function of prediction confidence"""

    global accuracy_w_conf_sp, accuracy_w_conf_gs

    if rank == "ORDER":
        pass

    elif rank == "SPECIES":
        ## Calculation at species level
        _helper_acc_alongside_conf(ground_truth, prediction, rank)

        ## Calculation at genus and family level
        gt_genus, gt_family = get_higher_taxon(ground_truth, rank, taxonomy_db)
        # Fing genus and family labels for the species prediction
        pred_genus, pred_family = copy.deepcopy(prediction), copy.deepcopy(prediction)
        for i in range(len(prediction)):
            genus, family = get_higher_taxon(prediction[i][0], "SPECIES", taxonomy_db)
            pred_genus[i][0], pred_family[i][0] = genus, family
        pred_genus_combined = _helper_combine_pred(pred_genus)
        pred_family_combined = _helper_combine_pred(pred_family)
        _helper_acc_alongside_conf(gt_genus, pred_genus_combined, "GENUS")
        _helper_acc_alongside_conf(gt_family, pred_family_combined, "FAMILY")

    elif rank == "GENUS":
        gt_genus = ground_truth
        gt_family = get_higher_taxon(ground_truth, rank, taxonomy_db)
        # Fing genus and family labels for the species prediction
        pred_genus, pred_family = copy.deepcopy(prediction), copy.deepcopy(prediction)
        for i in range(len(prediction)):
            genus, family = get_higher_taxon(prediction[i][0], "SPECIES", taxonomy_db)
            pred_genus[i][0], pred_family[i][0] = genus, family
        pred_genus_combined = _helper_combine_pred(pred_genus)
        pred_family_combined = _helper_combine_pred(pred_family)
        _helper_acc_alongside_conf(gt_genus, pred_genus_combined, "GENUS")
        _helper_acc_alongside_conf(gt_family, pred_family_combined, "FAMILY")

    else:
        if rank != "FAMILY":
            ground_truth = get_higher_taxon(ground_truth, rank, taxonomy_db)

        gt_family = ground_truth
        # Fing family labels for the species prediction
        pred_family = copy.deepcopy(prediction)
        for i in range(len(prediction)):
            a = prediction[i][0]
            _, family = get_higher_taxon(prediction[i][0], "SPECIES", taxonomy_db)
            pred_family[i][0] = family
        pred_family_combined = _helper_combine_pred(pred_family)
        _helper_acc_alongside_conf(gt_family, pred_family_combined, "FAMILY")


def species_accuracy(
    gt_label: str,
    rank: str,
    prediction: list[list[str, str]],
    species_list: pd.DataFrame,
):
    """Calculate species wise accuracy"""

    global accuracy_sp  # Each key as [correct, total, accuracy]

    if rank == "SPECIES":
        species_pred_top1 = prediction[0][0]
        synonym_pred_top1 = species_list.loc[
            species_list["gbif_species"] == species_pred_top1
        ]["search_species"].values[0]

        # Correct prediction
        if gt_label == species_pred_top1 or gt_label == synonym_pred_top1:
            if gt_label not in accuracy_sp.keys():
                accuracy_sp[gt_label] = [1, 1, float(100)]
            else:
                accuracy_sp[gt_label][0] += 1
                accuracy_sp[gt_label][1] += 1
                accuracy_sp[gt_label][2] = round(
                    accuracy_sp[gt_label][0] / accuracy_sp[gt_label][1], 2
                )
        # Incorrect prediction
        else:
            if gt_label not in accuracy_sp.keys():
                accuracy_sp[gt_label] = [0, 1, float(0)]
            else:
                accuracy_sp[gt_label][1] += 1
                accuracy_sp[gt_label][2] = round(
                    accuracy_sp[gt_label][0] / accuracy_sp[gt_label][1], 2
                )


def fine_grained_classification_eval(
    data_dir: str,
    plot_dir: str,
    checklist: str,
    taxonomy: str,
    exclusion_sp_file: str,
    skip_small_crops: bool = False,
    min_crop_dim: int = 200,
):
    """Main function for evaluating fine-grained moth classification predictions"""

    # List of excluded species
    exclusion_sp = pickle.load(open(exclusion_sp_file, "rb"))

    # Total skipped crops for excluded species
    global skip_crops_excluded
    skip_crops_excluded = 0

    # Read relevant directory(ies)
    metadata_dir = os.path.join(data_dir, "ami_traps_dataset", "metadata")

    # All-region global variables for accuracy
    global gt_moths, gt_family, gt_genus, gt_species, sp_top1, sp_top5, sp_top10, gs_top1, gs_top5, gs_top10, fm_top1, fm_top5, fm_top10
    gt_family, gt_genus, gt_species, gt_moths = 0, 0, 0, 0
    sp_top1, sp_top5, sp_top10 = 0, 0, 0
    gs_top1, gs_top5, gs_top10 = 0, 0, 0
    fm_top1, fm_top5, fm_top10 = 0, 0, 0

    # Region-wise global variables for accuracy
    global gt_family_neamerica, gt_genus_neamerica, gt_species_neamerica, neamerica_sp_top1, neamerica_sp_top5, neamerica_gs_top1, neamerica_gs_top5, neamerica_fm_top1, neamerica_fm_top5, gt_family_weurope, gt_genus_weurope, gt_species_weurope, weurope_sp_top1, weurope_sp_top5, weurope_gs_top1, weurope_gs_top5, weurope_fm_top1, weurope_fm_top5, gt_family_camerica, gt_genus_camerica, gt_species_camerica, camerica_sp_top1, camerica_sp_top5, camerica_gs_top1, camerica_gs_top5, camerica_fm_top1, camerica_fm_top5
    gt_family_neamerica, gt_genus_neamerica, gt_species_neamerica = 0, 0, 0
    gt_family_weurope, gt_genus_weurope, gt_species_weurope = 0, 0, 0
    gt_family_camerica, gt_genus_camerica, gt_species_camerica = 0, 0, 0
    neamerica_sp_top1, neamerica_sp_top5 = 0, 0
    neamerica_gs_top1, neamerica_gs_top5 = 0, 0
    neamerica_fm_top1, neamerica_fm_top5 = 0, 0
    weurope_sp_top1, weurope_sp_top5 = 0, 0
    weurope_gs_top1, weurope_gs_top5 = 0, 0
    weurope_fm_top1, weurope_fm_top5 = 0, 0
    camerica_sp_top1, camerica_sp_top5 = 0, 0
    camerica_gs_top1, camerica_gs_top5 = 0, 0
    camerica_fm_top1, camerica_fm_top5 = 0, 0

    # Region-wise data statistics
    global neamerica_family, neamerica_genus, neamerica_species, camerica_family, camerica_genus, camerica_species, weurope_family, weurope_genus, weurope_species
    neamerica_family, neamerica_genus, neamerica_species = [], [], []
    camerica_family, camerica_genus, camerica_species = [], [], []
    weurope_family, weurope_genus, weurope_species = [], [], []

    # Variables for region-wise accuracy
    global accuracy_w_conf_sp, accuracy_w_conf_gs, accuracy_w_conf_fm
    accuracy_w_conf_sp = pd.DataFrame(
        0,
        columns=["conf-" + str(thresh) for thresh in range(10, 100, 10)],
        index=["correct", "conf-total", "reject", "moths-total"],
    )
    accuracy_w_conf_gs = pd.DataFrame(
        0,
        columns=["conf-" + str(thresh) for thresh in range(10, 100, 10)],
        index=["correct", "conf-total", "reject", "moths-total"],
    )
    accuracy_w_conf_fm = pd.DataFrame(
        0,
        columns=["conf-" + str(thresh) for thresh in range(10, 100, 10)],
        index=["correct", "conf-total", "reject", "moths-total"],
    )

    # Define global variable for crop count histogram
    global hist_dict
    hist_dict, hist_dict["FAMILY"], hist_dict["GENUS"], hist_dict["SPECIES"] = (
        {},
        {},
        {},
        {},
    )

    # Global variable for calculating species-wise accuracy
    global accuracy_sp
    accuracy_sp = {}

    # [Optional] Remove small crops
    skip_small_crops = False
    if skip_small_crops:
        print(
            f"Crops with length less than {min_crop_dim}px are removed in this analysis."
        )

    # Read the global moth species list
    species_list = pd.read_csv(checklist)
    taxonomy_db = pd.read_csv(taxonomy)

    # Get the image list and associated predctions
    # pred_dir = os.path.join(data_dir, "ami_traps_dataset", "model_predictions", "baseline")
    model = "northamerica_vit_b_baseline_run1"  # TEMP CHANGE
    print(f"Running analysis for {model}.")
    pred_dir = os.path.join(
        data_dir, "ami_traps_dataset", "model_predictions", "all-architectures", model
    )  # TEMP CHANGE
    image_pred_list = os.listdir(pred_dir)

    # Iterate over each image predictions
    for image_pred in image_pred_list:
        pred_data = json.load(open(os.path.join(pred_dir, image_pred)))

        # Get the region
        img_basename = image_pred.split("_")[0]
        metadata_file = os.path.join(metadata_dir, img_basename + ".json")
        metadata = json.load(open(metadata_file))
        region = metadata["region"]

        # Iterate over each bounding box
        for bbox in pred_data:
            gt_label = bbox["ground_truth"][0]
            gt_rank = bbox["ground_truth"][1]
            prediction = bbox["moth_classification"]
            bbox_coord = bbox["bbox_coordinates"]

            # [Optional] Remove small crops
            if skip_small_crops:
                flag = check_crop_size(bbox_coord, image_pred, data_dir, min_crop_dim)
                if flag:  # skip this crop
                    continue

            # Get only the moth crops
            if (
                gt_label != "Non-Moth"
                and gt_label != "Unidentifiable"
                and gt_label != "Unclassified"
            ):
                count_crops_at_taxon_level(gt_label, gt_rank, exclusion_sp)
                # histogram_at_taxon_level(gt_label, gt_rank, region, taxonomy_db, exclusion_sp)
                _helper_finegrained_accuracy(
                    gt_label,
                    gt_rank,
                    prediction,
                    region,
                    species_list,
                    taxonomy_db,
                    exclusion_sp,
                )
                # accuracy_alongside_confidence(gt_label, gt_rank, prediction, taxonomy_db)
                # species_accuracy(gt_label, gt_rank, prediction, species_list)

    # Save species-wise accuracy data
    # accuracy_sp = dict(sorted(accuracy_sp.items(), key=lambda item: item[1][1], reverse=True))
    # with open(os.path.join(plot_dir, "species_accuracy.json"), "w") as outfile: json.dump(accuracy_sp, outfile, indent=4)

    # Save accuracy v/s confidence data
    # for thresh in range(10, 100, 10):
    #     accuracy_w_conf_sp.loc["moths-total"]["conf-"+str(thresh)] = gt_species
    #     accuracy_w_conf_gs.loc["moths-total"]["conf-"+str(thresh)] = gt_genus
    #     accuracy_w_conf_fm.loc["moths-total"]["conf-"+str(thresh)] = gt_family
    # accuracy_w_conf_sp.to_csv(os.path.join(plot_dir, "sp_acc_rej_vs_conf.csv"))
    # accuracy_w_conf_gs.to_csv(os.path.join(plot_dir, "gs_acc_rej_vs_conf.csv"))
    # accuracy_w_conf_fm.to_csv(os.path.join(plot_dir, "fm_acc_rej_vs_conf.csv"))

    # Print, plot and save histogram plot for diff. taxon levels
    # hist_dict_family = dict(sorted(hist_dict["FAMILY"].items(), key=lambda item: item[1], reverse=True))
    # with open(os.path.join(plot_dir, "hist_family.json"), "w") as outfile: json.dump(hist_dict_family, outfile, indent=4)
    # plt.figure(figsize=(8,14))
    # plt.bar(hist_dict_family.keys(), hist_dict_family.values())
    # plt.xticks(list(hist_dict_family.keys()), rotation=90)
    # plt.xlabel("Family name")
    # plt.ylabel("Moth count")
    # plt.title("Moth crops distribution at family level")
    # plt.savefig(os.path.join(plot_dir, "family_hist.png"))

    # hist_dict_genus = dict(sorted(hist_dict["GENUS"].items(), key=lambda item: item[1], reverse=True))
    # with open(os.path.join(plot_dir, "hist_genus.json"), "w") as outfile: json.dump(hist_dict_genus, outfile, indent=4)
    # plt.figure(figsize=(18,16))
    # plt.bar(hist_dict_genus.keys(), hist_dict_genus.values())
    # plt.xticks(list(hist_dict_genus.keys()), rotation=90)
    # plt.xlabel("Genus name")
    # plt.ylabel("Moth count")
    # plt.title("Moth crops distribution at genus level")
    # plt.savefig(os.path.join(plot_dir, "genus_hist.png"))

    # hist_dict_species = dict(sorted(hist_dict["SPECIES"].items(), key=lambda item: item[1], reverse=True))
    # with open(os.path.join(plot_dir, "hist_species.json"), "w") as outfile: json.dump(hist_dict_species, outfile, indent=4)
    # plt.figure(figsize=(28,20))
    # plt.bar(hist_dict_species.keys(), hist_dict_species.values())
    # plt.xticks(list(hist_dict_species.keys()), rotation=90)
    # plt.xlabel("species name")
    # plt.ylabel("Moth count")
    # plt.title("Moth crops distribution at species level")
    # plt.savefig(os.path.join(plot_dir, "species_hist.png"))

    # Print moth data collection statistics
    print(
        f"\nFine-grained moth annotation statistics:\
        \nMoths - {gt_moths}\
        \nOrder - {gt_moths} ({round(gt_moths/gt_moths*100,2)}%)\
        \nFamily - {gt_family} ({round(gt_family/gt_moths*100,2)}%)\
        \nGenus - {gt_genus} ({round(gt_genus/gt_moths*100,2)}%)\
        \nSpecies - {gt_species} ({round(gt_species/gt_moths*100,2)}%)\
        \n"
    )

    print(f"Total skipped crops for missing species {skip_crops_excluded}.")

    print(
        f"\nMoth classification accuracy for the region (top-1, top-5, top-10):\
        \nFamily: {round(fm_top1/gt_family*100,2)}%, {round(fm_top5/gt_family*100,2)}%, {round(fm_top10/gt_family*100,2)}%\
        \nGenus: {round(gs_top1/gt_genus*100,2)}%, {round(gs_top5/gt_genus*100,2)}%, {round(gs_top10/gt_genus*100,2)}%\
        \nSpecies: {round(sp_top1/gt_species*100,2)}%, {round(sp_top5/gt_species*100,2)}%, {round(sp_top10/gt_species*100,2)}%\
        \n"
    )

    # print(
    #     f"\nMoth class statistics at different taxon levels:\
    #     \nFamily: classes - {len(hist_dict_family.keys())};  crops - {sum(hist_dict_family.values())}\
    #     \nGenus: classes - {len(hist_dict_genus.keys())};  crops - {sum(hist_dict_genus.values())}\
    #     \nSpecies: classes - {len(hist_dict_species.keys())};  crops - {sum(hist_dict_species.values())}\
    #     \n"
    # )

    # print(
    #     f"Region-wise annotations:\
    #     \nNE-America: Family - {len(neamerica_family)}, Genus - {len(neamerica_genus)}, Species - {len(neamerica_species)}\
    #     \nW-Europe: Family - {len(weurope_family)}, Genus - {len(weurope_genus)}, Species - {len(weurope_species)}\
    #     \nC-America: Family - {len(camerica_family)}, Genus - {len(camerica_genus)}, Species - {len(camerica_species)}\
    #     \n"
    # )

    # print(
    # f"\nRegion-wise accuracy (top-1, top-5):\
    #     \nFamily: NE-America {round(neamerica_fm_top1/gt_family_neamerica*100,2)}%, {round(neamerica_fm_top5/gt_family_neamerica*100,2)}%;\
    #     W-Europe {round(weurope_fm_top1/gt_family_weurope*100,2)}%, {round(weurope_fm_top5/gt_family_weurope*100,2)}%;\
    #     C-America {round(camerica_fm_top1/gt_family_camerica*100,2)}%, {round(camerica_fm_top5/gt_family_camerica*100,2)}%;\
    #     \nGenus: NE-America {round(neamerica_gs_top1/gt_genus_neamerica*100,2)}%, {round(neamerica_gs_top5/gt_genus_neamerica*100,2)}%;\
    #     W-Europe {round(weurope_gs_top1/gt_genus_weurope*100,2)}%, {round(weurope_gs_top5/gt_genus_weurope*100,2)}%;\
    #     C-America {round(camerica_gs_top1/gt_genus_camerica*100,2)}%, {round(camerica_gs_top5/gt_genus_camerica*100,2)}%;\
    #     \nSpecies: NE-America {round(neamerica_sp_top1/gt_species_neamerica*100,2)}%, {round(neamerica_sp_top5/gt_species_neamerica*100,2)}%;\
    #     W-Europe {round(weurope_sp_top1/gt_species_weurope*100,2)}%, {round(weurope_sp_top5/gt_species_weurope*100,2)}%;\
    #     C-America {round(camerica_sp_top1/gt_species_camerica*100,2)}%, {round(camerica_sp_top5/gt_species_camerica*100,2)}%;\
    #     \n"
    # )


def binary_classification_eval(
    data_dir: str, plot_dir: str, skip_small_crops: bool = True, min_crop_dim: int = 100
):
    """Evaluate binary classification predictions"""

    # Get the image list and associated predctions
    pred_dir = os.path.join(data_dir, "ami_traps_dataset", "model_predictions")
    image_pred_list = os.listdir(pred_dir)

    # Evaluation metrics variables
    gt_moths, gt_nonmoths = 0, 0
    unidentifiable = 0
    skip_gt_moths, skip_gt_nonmoths = 0, 0
    tp, tn, fp, fn = 0, 0, 0, 0
    tp_cf, tn_cf, fp_cf, fn_cf = [], [], [], []

    # [Optional] Remove small crops
    skip_small_crops = True
    if skip_small_crops:
        print(
            f"Crops with length less than {min_crop_dim}px are removed in this analysis."
        )

    # Iterate over each image predictions
    for image_pred in image_pred_list:
        pred_data = json.load(open(os.path.join(pred_dir, image_pred)))

        # Iterate over each bounding box
        for bbox in pred_data:
            gt = bbox["ground_truth"][0]
            pred = bbox["binary_classification"][0][0]
            conf = round(float(bbox["binary_classification"][0][1]) * 100)
            bbox_coord = bbox["bbox_coordinates"]

            # [Optional] Remove small crops
            if skip_small_crops:
                flag = check_crop_size(bbox_coord, image_pred, data_dir, min_crop_dim)
                if flag:  # skip this crop
                    if gt == "Non-Moth":
                        skip_gt_nonmoths += 1
                    elif gt == "Unidentifiable":
                        pass
                    else:
                        skip_gt_moths += 1
                    continue
                else:
                    pass

            # Fill up metrics variables
            if gt == "Non-Moth":
                if pred == "nonmoth":
                    tn += 1
                    tn_cf.append(conf)
                if pred == "moth":
                    fp += 1
                    fp_cf.append(conf)
                    # save_insect_crop(bbox_coord, image_pred, data_dir, "false_positives")
                gt_nonmoths += 1
            elif gt == "Unidentifiable":
                unidentifiable += 1
                continue
            elif gt == "Unclassified":
                continue
            else:
                if pred == "moth":
                    tp += 1
                    tp_cf.append(conf)
                if pred == "nonmoth":
                    fn += 1
                    fn_cf.append(conf)
                    # save_insect_crop(bbox_coord, image_pred, data_dir, "false_negatives")
                gt_moths += 1

    # Aggregated metrics
    total_crops = gt_moths + gt_nonmoths
    accuracy = round((tp + tn) / (tp + tn + fp + fn) * 100, 2)
    precision = round((tp) / (tp + fp) * 100, 2)
    recall = round((tp) / (tp + fn) * 100, 2)
    fscore = round((2 * precision * recall) / (precision + recall), 2)
    print(
        f"\nBinary classification evaluation:\
        \nTotal insect crops - {total_crops}\
        \nGround-truth moth crops - {gt_moths} ({round(gt_moths/total_crops*100,2)}%)\
        \nGround-truth non-moth crops - {gt_nonmoths} ({round(gt_nonmoths/total_crops*100,2)}%)\
        \nAccuracy - {accuracy}%\
        \nPrecision - {precision}%\
        \nRecall - {recall}%\
        \nF1 score - {fscore}%\
        \nTP, FP, TN, FN- {tp}, {fp}, {tn}, {fn}\
        \n"
    )

    # Confidence distribution plot
    sns.set(style="darkgrid")
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.tight_layout(pad=3.0)
    fig.suptitle("Binary Classifier Confidence Distribution", fontsize=15)
    sns.histplot(
        data=tp_cf,
        binwidth=10,
        binrange=(0, 100),
        kde=True,
        color="violet",
        ax=axs[0, 0],
        label="True Positives (TP)",
    )
    sns.histplot(
        data=fp_cf,
        binwidth=10,
        binrange=(0, 100),
        kde=True,
        color="olive",
        ax=axs[0, 1],
        label="False Positives (FP)",
    )
    sns.histplot(
        data=tn_cf,
        binwidth=10,
        binrange=(0, 100),
        kde=True,
        color="orange",
        ax=axs[1, 0],
        label="True Negatives (TN)",
    )
    sns.histplot(
        data=fn_cf,
        binwidth=10,
        binrange=(0, 100),
        kde=True,
        color="teal",
        ax=axs[1, 1],
        label="False Negatives (FN)",
    )
    axs[0, 0].legend(loc="upper left")
    axs[0, 0].set(xlabel="Prediction confidence")
    axs[0, 1].legend(loc="upper left")
    axs[0, 1].set(xlabel="Prediction confidence")
    axs[1, 0].legend(loc="upper left")
    axs[1, 0].set(xlabel="Prediction confidence")
    axs[1, 1].legend(loc="upper left")
    axs[1, 1].set(xlabel="Prediction confidence")
    # plt.savefig(os.path.join(plot_dir, "binary_classification_confidence_dist.png"))


def data_statistics(data_dir: str, taxonomy_db: str):
    """Analyze the various annotations statistics"""

    # Read the taxonomy database
    taxonomy_db = pd.read_csv(taxonomy)

    # Get the image list and associated predctions
    pred_dir = os.path.join(data_dir, "ami_traps_dataset", "model_predictions")
    image_pred_list = os.listdir(pred_dir)
    metadata_dir = os.path.join(data_dir, "ami_traps_dataset", "metadata")

    # Evaluation metrics and data statistics variables
    gt_moths, gt_nonmoths, unidentifiable = 0, 0, 0
    neamerica_imgs, camerica_imgs, weurope_imgs = 0, 0, 0

    # Iterate over each image predictions
    for image_pred in image_pred_list:
        pred_data = json.load(open(os.path.join(pred_dir, image_pred)))

        # Get the region
        img_basename = image_pred.split("_")[0]
        metadata_file = os.path.join(metadata_dir, img_basename + ".json")
        metadata = json.load(open(metadata_file))
        region = metadata["region"]

        # Iterate over each bounding box
        for bbox in pred_data:
            gt = bbox["ground_truth"][0]

            # Broad classification variables
            if gt == "Non-Moth":
                gt_nonmoths += 1
            elif gt == "Unidentifiable":
                unidentifiable += 1
            elif gt == "Unclassified":
                pass
            else:
                gt_moths += 1
                rank = bbox["ground_truth"][1]
                if rank not in ["SPECIES", "GENUS", "FAMILY", "ORDER"]:
                    gt = get_higher_taxon(gt, rank, taxonomy_db)
                    rank = "FAMILY"

                # Region statistics
                if region == "NorthEasternAmerica":
                    neamerica_imgs += 1
                elif region == "WesternEurope":
                    weurope_imgs += 1
                elif region == "CentralAmerica":
                    camerica_imgs += 1
                else:
                    raise Exception("Region information unknown.")

    total_crops = gt_moths + gt_nonmoths + unidentifiable
    print(
        f"\nHigh-level annotation statistics:\
        \nTotal crops - {total_crops}\
        \nMoths - {gt_moths} ({round(gt_moths/total_crops*100,2)}%)\
        \nNon-moths - {gt_nonmoths} ({round(gt_nonmoths/total_crops*100,2)}%)\
        \nUnidentifiable - {unidentifiable} ({round(unidentifiable/total_crops*100,2)}%)\
        \n"
    )

    print(
        f"Region-wise moth annotations:\
        \nNE-America: - {neamerica_imgs}\
        \nW-Europe: - {weurope_imgs}\
        \nC-America: - {camerica_imgs}\
        \n"
    )


if __name__ == "__main__":
    ECCV2024_DATA = os.getenv("ECCV2024_DATA")
    SPECIES_LISTS_DIR = os.getenv("SPECIES_LISTS_DIR")

    plot_dir = "./plots"
    taxonomy = f"{ECCV2024_DATA}/ami-taxonomy-joined-20240119.csv"
    exclusion_sp_file = f"{ECCV2024_DATA}/excluded_sp_from_AMI-GBIF.pickle"

    # data_statistics(data_dir, taxonomy)
    # binary_classification_eval(data_dir, plot_dir)
    fine_grained_classification_eval(
        ECCV2024_DATA, plot_dir, SPECIES_LISTS_DIR, taxonomy, exclusion_sp_file
    )
