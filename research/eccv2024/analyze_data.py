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


class MothAnnotationStatistics:
    """Class to store moth annotation statistics"""

    def __init__(self, data: list[dict], taxon_db: pd.DataFrame):
        # Variables for storing annotation statistics
        self.data = data
        self.taxon_db = taxon_db
        self.vermont_cnt = 0
        self.quebec_cnt = 0
        self.uk_cnt = 0
        self.denmark_cnt = 0
        self.panama_cnt = 0
        self.order = 0
        self.family = 0
        self.sub_family = 0
        self.tribe = 0
        self.sub_tribe = 0
        self.genus = 0
        self.species = 0
        self.list_order = []
        self.list_family = []
        self.list_sub_family = []
        self.list_tribe = []
        self.list_sub_tribe = []
        self.list_genus = []
        self.list_species = []
        self.boxes = 0
        self.nonmoths = 0
        self.moths = 0
        self.unidentify = 0
        self.unclassify = 0
        self.mul_label = 0
        self.labels = 0
        self.moths_dir_label = 0

    def _update_moth_annotation_statistics(self, label_list: list[str]):
        """Update moth annotation statistics given a label list and taxonomy backbone"""

        for taxon in label_list:
            # Match the name in the AMI database
            try:
                taxon_level = self.taxon_db.loc[
                    self.taxon_db["name"] == taxon, "rank"
                ].values
            except KeyError:
                print(f"Taxon {taxon} is not found in the database.")
                continue

            # Check the hierarchy for the taxon
            if taxon_level == "ORDER":
                self.list_order.append(taxon)
                self.order += 1
            elif taxon_level == "FAMILY":
                self.list_family.append(taxon)
                self.family += 1
            elif taxon_level == "SUBFAMILY":
                self.list_sub_family.append(taxon)
                self.sub_family += 1
            elif taxon_level == "TRIBE":
                self.list_tribe.append(taxon)
                self.tribe += 1
            elif taxon_level == "SUBTRIBE":
                self.list_sub_tribe.append(taxon)
                self.sub_tribe += 1
            elif taxon_level == "GENUS":
                self.list_genus.append(taxon)
                self.genus += 1
            elif taxon_level == "SPECIES":
                self.list_species.append(taxon)
                self.species += 1
            else:
                print(f"Taxon level for {taxon} is not known.")
                pass

    def analyze_data(self):
        """Main function to analze the AMI benchmark annotation data"""
        img_names = []

        for i in range(len(self.data)):
            data_point = self.data[i]

            # Get region-wise counts
            deployment = data_point["data"]["deployment"]
            if "Panama" in deployment:
                self.panama_cnt += 1
            elif "UK" in deployment:
                self.uk_cnt += 1
            elif "Quebec" in deployment:
                self.quebec_cnt += 1
            elif "Denmark" in deployment:
                self.denmark_cnt += 1
            elif "Vermont" in deployment:
                self.vermont_cnt += 1
            else:
                print("No region name found in the data.")

            # Check for duplicate images
            image_url = data_point["data"]["image"]
            image_name = os.path.basename(os.path.normpath(image_url))
            if image_name in img_names:
                # print(f"Image {image_name} for {deployment} has duplicate entry.")
                if "Denmark" in deployment:
                    self.denmark_cnt -= 1
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
                    self.boxes += 1

                # Binary classification statistics
                if item["type"] == "labels":
                    labels_ids.append(item["id"])
                    self.labels += 1
                    if not item["value"]["labels"]:
                        self.unclassify += 1
                    elif item["value"]["labels"][0] == "Non-Moth":
                        self.nonmoths += 1
                    elif item["value"]["labels"][0] == "Moth":
                        self.moths += 1
                    elif item["value"]["labels"][0] == "Unidentifiable":
                        self.unidentify += 1
                    else:
                        print(item["value"]["labels"][0])

                # Fine-grained moth classification statistics
                if item["type"] == "taxonomy":
                    taxonomy_ids.append(item["id"])
                    # Sort based on higher to lower taxon classification
                    taxonomy_labels = sorted(item["value"]["taxonomy"], key=len)
                    if len(taxonomy_labels) == 1:
                        # Add single annotation
                        self._update_moth_annotation_statistics(taxonomy_labels[0])
                    else:
                        self.mul_label += 1
                        for i in range(len(taxonomy_labels) - 1):
                            # Check all annotations if it is a subset of the ...
                            # ... lower most classification
                            if not set(taxonomy_labels[i]).issubset(
                                set(taxonomy_labels[-1])
                            ):
                                self._update_moth_annotation_statistics(
                                    taxonomy_labels[i]
                                )
                        # Add the deepest label
                        self._update_moth_annotation_statistics(taxonomy_labels[-1])

            # Count the moth crops that are directly labelled
            # at taxonomy level, w/o coarse labelling
            for id in taxonomy_ids:
                if id not in labels_ids:
                    self.moths_dir_label += 1

        self.unclassify += self.boxes - self.labels - self.moths_dir_label
        self.moths += self.moths_dir_label

    def print_statistics(self):
        """Print statistics to the terminal"""

        # Variables for making lines shorter for better readability
        total = self.vermont_cnt + self.quebec_cnt + self.uk_cnt + self.denmark_cnt
        total += self.panama_cnt

        print(
            f"\nRegion-wise annotated images:\
            \nVermont - {self.vermont_cnt}\
            \nQuebec - {self.quebec_cnt}\
            \nUK - {self.uk_cnt}\
            \nDenmark - {self.denmark_cnt}\
            \nPanama - {self.panama_cnt}\
            \nTotal - {total}\n"
        )

        print(
            f"General annotation statistics:\
            \nBounding boxes: {self.boxes}\
            \nNon-moths: {self.nonmoths} ({round(self.nonmoths/self.boxes*100,2)}%)\
            \nMoths: {self.moths} ({round(self.moths/self.boxes*100,2)}%)\
            \nUnidentify: {self.unidentify}({round(self.unidentify/self.boxes*100,2)}%)\
            \nUnclassify: {self.unclassify}({round(self.unclassify/self.boxes*100,2)}%)\
            \n"
        )

        print(
            f"Moth annotation statistics:\
            \nMoths: {self.moths}\
            \nOrder: {self.moths} ({round(self.moths/self.moths*100,2)}%)\
            \nFamily: {self.family} ({round(self.family/self.moths*100,2)}%)\
            \nSubfamily: {self.sub_family} ({round(self.sub_family/self.moths*100,2)}%)\
            \nTribe: {self.tribe} ({round(self.tribe/self.moths*100,2)}%)\
            \nSubtribe: {self.sub_tribe} ({round(self.sub_tribe/self.moths*100,2)}%)\
            \nGenus: {self.genus} ({round(self.genus/self.moths*100,2)}%)\
            \nSpecies: {self.species} ({round(self.species/self.moths*100,2)}%)\
            \nMultilabels: {self.mul_label} ({round(self.mul_label/self.moths*100,2)}%)\
            \n"
        )

        print(
            f"Moth class statistics:\
            \nFamilies - {len(set(self.list_family))}\
            \nSub-families - {len(set(self.list_sub_family))}\
            \nTribes - {len(set(self.list_tribe))}\
            \nSub-tribes - {len(set(self.list_sub_tribe))}\
            \nGenera - {len(set(self.list_genus))}\
            \nSpecies - {len(set(self.list_species))}\
            \n"
        )


if __name__ == "__main__":
    ECCV2024_DATA = os.getenv("ECCV2024_DATA")
    if not ECCV2024_DATA:
        print("Env. variable ECCV2024_DATA is not set.")
    taxon_db = pd.read_csv(f"{ECCV2024_DATA}/ami-taxa-20231029.csv")

    annotation_file = f"{ECCV2024_DATA}/annotated-tasks-20240110.json"
    with open(annotation_file) as f:
        annotation_data = json.load(f)

    # Run and print the analysis
    moth_stats = MothAnnotationStatistics(annotation_data, taxon_db)
    moth_stats.analyze_data()
    moth_stats.print_statistics()
