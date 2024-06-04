"""
Author: Aditya Jain
Date last modified: October 25, 2023
About: Analyse different metrics for the annotated data
"""

import json
import os

import pandas as pd

# 3rd party packages
from dotenv import load_dotenv

# Load secrets and config from optional .env file
load_dotenv()


def print_statistics():
    """Print statistics to the terminal"""

    print(
        f"\nRegion-wise annotated images:\
        \nVermont - {vermont_count}\
        \nQuebec - {quebec_count}\
        \nUK - {uk_count}\
        \nDenmark - {denmark_count}\
        \nPanama - {panama_count}\
        \nTotal - {vermont_count+quebec_count+uk_count+denmark_count+panama_count}\n"
    )

    print(
        f"General annotation statistics:\
        \nBounding boxes - {num_boxes}\
        \nNon-moths - {num_nonmoths} ({round(num_nonmoths/num_boxes*100,2)}%)\
        \nMoths - {num_moths} ({round(num_moths/num_boxes*100,2)}%)\
        \nUnidentifiables - {num_unidentify} ({round(num_unidentify/num_boxes*100,2)}%)\
        \nUnclassified - {num_unclassify} ({round(num_unclassify/num_boxes*100,2)}%)\
        \n"
    )

    print(
        f"Moth annotation statistics:\
        \nMoths - {num_moths}\
        \nOrder - {num_moths} ({round(num_moths/num_moths*100,2)}%)\
        \nFamily - {num_family} ({round(num_family/num_moths*100,2)}%)\
        \nSub-family - {num_sub_family} ({round(num_sub_family/num_moths*100,2)}%)\
        \nTribe - {num_tribe} ({round(num_tribe/num_moths*100,2)}%)\
        \nSub-tribe - {num_sub_tribe} ({round(num_sub_tribe/num_moths*100,2)}%)\
        \nGenus - {num_genus} ({round(num_genus/num_moths*100,2)}%)\
        \nSpecies - {num_species} ({round(num_species/num_moths*100,2)}%)\
        \nMultiple labels - {num_mul_label} ({round(num_mul_label/num_moths*100,2)}%)\
        \n"
    )

    print(
        f"Moth class statistics:\
        \nFamilies - {len(set(list_family))}\
        \nSub-families - {len(set(list_sub_family))}\
        \nTribes - {len(set(list_tribe))}\
        \nSub-tribes - {len(set(list_sub_tribe))}\
        \nGenera - {len(set(list_genus))}\
        \nSpecies - {len(set(list_species))}\
        \n"
    )


def update_moth_annotation_statistics(label_list: list[str], taxon_info: pd.DataFrame):
    """Update moth annotation statistics given a label list and taxonomy backbone"""

    global num_order, num_family, num_sub_family, num_tribe, num_sub_tribe
    global num_genus, num_species

    for taxon in label_list:
        # Match the name in the AMI database
        try:
            taxon_level = taxon_info.loc[taxon_info["name"] == taxon, "rank"].values
        except KeyError:
            print(f"Taxon {taxon} is not found in the database.")
            continue

        # Check the hierarchy for the taxon
        if taxon_level == "ORDER":
            list_order.append(taxon)
            num_order += 1
        elif taxon_level == "FAMILY":
            list_family.append(taxon)
            num_family += 1
        elif taxon_level == "SUBFAMILY":
            list_sub_family.append(taxon)
            num_sub_family += 1
        elif taxon_level == "TRIBE":
            list_tribe.append(taxon)
            num_tribe += 1
        elif taxon_level == "SUBTRIBE":
            list_sub_tribe.append(taxon)
            num_sub_tribe += 1
        elif taxon_level == "GENUS":
            list_genus.append(taxon)
            num_genus += 1
        elif taxon_level == "SPECIES":
            list_species.append(taxon)
            num_species += 1
        else:
            print(f"Taxon level for {taxon} is not known.")
            pass


def analyze_data(data: list[dict]):
    """Main function to analze the AMI benchmark annotation data"""

    global vermont_count, quebec_count, uk_count, denmark_count, panama_count
    global num_order, num_family, num_sub_family, num_sub_tribe, num_tribe
    global num_genus, num_species, list_order, list_family, list_sub_family
    global list_tribe, list_sub_tribe, list_genus, list_species, num_boxes
    global num_nonmoths, num_moths, num_unidentify, num_unclassify, num_mul_label

    # Region-wise counts
    vermont_count = 0
    quebec_count = 0
    uk_count = 0
    denmark_count = 0
    panama_count = 0

    # Annotation statitics
    num_boxes = 0
    num_nonmoths = 0
    num_moths = 0
    num_moths_dir_label = 0
    num_unidentify = 0
    num_unclassify = 0
    num_labels = 0
    num_order = 0
    num_family = 0
    num_sub_family = 0
    num_tribe = 0
    num_sub_tribe = 0
    num_genus = 0
    num_species = 0
    list_order = []
    list_family = []
    list_sub_family = []
    list_tribe = []
    list_sub_tribe = []
    list_genus = []
    list_species = []
    num_mul_label = 0
    img_names = []

    for i in range(len(data)):
        data_point = data[i]

        # Get region-wise counts
        deployment = data_point["data"]["deployment"]
        if "Panama" in deployment:
            panama_count += 1
        elif "UK" in deployment:
            uk_count += 1
        elif "Quebec" in deployment:
            quebec_count += 1
        elif "Denmark" in deployment:
            denmark_count += 1
        elif "Vermont" in deployment:
            vermont_count += 1
        else:
            print("No region name found in the data.")

        # Check for duplicate images
        image_url = data_point["data"]["image"]
        image_name = os.path.basename(os.path.normpath(image_url))
        if image_name in img_names:
            print(f"Image {image_name} for {deployment} has duplicate entry.")
            if "Denmark" in deployment:
                denmark_count -= 1
            else:
                print(f"{deployment} region also has duplicate images.")
        else:
            img_names.append(image_name)

        # Take results from reviewer annotations, if available
        annotator_results = data_point["annotations"][0]["result"]
        reviewer_results = data_point["annotations"][0]["reviews"]
        if reviewer_results and reviewer_results[0]["fixed_annotation_history"]:
            annotations = reviewer_results[0]["fixed_annotation_history_result"]
        else:
            annotations = annotator_results

        # Iterate over all annotations
        labels_ids = []
        taxonomy_ids = []
        for item in annotations:
            # Total bounding boxes
            if item["type"] == "rectangle":
                num_boxes += 1

            # Binary classification statistics
            if item["type"] == "labels":
                labels_ids.append(item["id"])
                num_labels += 1
                if not item["value"]["labels"]:
                    num_unclassify += 1
                elif item["value"]["labels"][0] == "Non-Moth":
                    num_nonmoths += 1
                elif item["value"]["labels"][0] == "Moth":
                    num_moths += 1
                elif item["value"]["labels"][0] == "Unidentifiable":
                    num_unidentify += 1
                else:
                    print(item["value"]["labels"][0])

            # Fine-grained moth classification statistics
            if item["type"] == "taxonomy":
                taxonomy_ids.append(item["id"])
                # Sort based on higher to lower taxon classification
                taxonomy_labels = sorted(item["value"]["taxonomy"], key=len)
                if len(taxonomy_labels) == 1:
                    # Add single annotation
                    update_moth_annotation_statistics(taxonomy_labels[0], taxon_db)
                else:
                    num_mul_label += 1
                    for i in range(len(taxonomy_labels) - 1):
                        # Check all annotations if it is a subset of the ...
                        # ... lower most classification
                        if not set(taxonomy_labels[i]).issubset(
                            set(taxonomy_labels[-1])
                        ):
                            update_moth_annotation_statistics(
                                taxonomy_labels[i], taxon_db
                            )
                    # Add the deepest label
                    update_moth_annotation_statistics(taxonomy_labels[-1], taxon_db)

        # Count the moth crops that are directly labelled
        # at taxonomy level, w/o coarse labelling
        for id in taxonomy_ids:
            if id not in labels_ids:
                num_moths_dir_label += 1

    num_unclassify += num_boxes - num_labels - num_moths_dir_label
    num_moths += num_moths_dir_label
    print_statistics()


if __name__ == "__main__":
    ECCV2024_DATA = os.getenv("ECCV2024_DATA_PATH")
    taxon_db = pd.read_csv(f"{ECCV2024_DATA}/ami-taxa-20231029.csv")

    annotation_file = f"{ECCV2024_DATA}/annotated-tasks-20240110.json"
    with open(annotation_file) as f:
        annotation_data = json.load(f)

    analyze_data(annotation_data)
