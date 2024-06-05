import json
import os

import pandas as pd

GBIF_COUNT_FILE = os.getenv("GBIF_COUNT_FILE")
MASTER_SPECIES_LIST = os.getenv("MASTER_SPECIES_LIST")

# Important variables
xaxis_labels = ["4-10", "11-20", "21-50", "51-100", "101-200", "201-500", "500-1K"]
yaxis_data = [[], [], [], [], [], [], []]
min_imgs = 0  # Minimum images in AMI-Traps to consider for evaluation
dir = "./plots"
acc_data = json.load(open(os.path.join(dir, "species_accuracy.json")))
with open(GBIF_COUNT_FILE) as f:
    gbif_training_count = json.load(f)
species_list = pd.read_csv(MASTER_SPECIES_LIST)


def get_num_gbif_images(species: str, gbif_count: dict, species_list: pd.DataFrame):
    """Get number of GBF images for a species"""

    # Search for the name in the species list
    try:
        numeric_id = species_list.loc[
            species_list["gbif_species"] == species, "accepted_taxon_key"
        ].values[0]
    except Exception:
        numeric_id = species_list.loc[
            species_list["search_species"] == species, "accepted_taxon_key"
        ].values[0]

    if numeric_id == -1:
        print(f"Species {species} is not found in the database.")
        return 0

    # Find image count using taxon key
    try:
        return gbif_count[str(numeric_id)]
    except KeyError:
        print(
            f"Taxon name {species} with id {numeric_id} has no count in the data file."
        )
        return 0


for species in acc_data.keys():
    total_imgs, accuracy = acc_data[species][1], acc_data[species][2] * 100

    if total_imgs >= min_imgs:
        num_gbif_imgs = get_num_gbif_images(species, gbif_training_count, species_list)

#         if num_gbif_imgs!=0:
#             if num_gbif_imgs <= 10:
#                 yaxis_data[0].append(accuracy)
#             elif num_gbif_imgs <= 20:
#                 yaxis_data[1].append(accuracy)
#             elif num_gbif_imgs <= 50:
#                 yaxis_data[2].append(accuracy)
#             elif num_gbif_imgs <= 100:
#                 yaxis_data[3].append(accuracy)
#             elif num_gbif_imgs <= 200:
#                 yaxis_data[4].append(accuracy)
#             elif num_gbif_imgs <= 500:
#                 yaxis_data[5].append(accuracy)
#             else:
#                 yaxis_data[6].append(accuracy)

# plt.boxplot(yaxis_data, labels=xaxis_labels)
# plt.savefig(os.path.join(dir, "boxplot.png"), bbox_inches="tight")

# for x_categ in yaxis_data:
#     print(len(x_categ))
