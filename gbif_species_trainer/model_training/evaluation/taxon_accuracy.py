"""
Author         : Aditya Jain
Last modified  : May 23rd, 2023
About          : Calculates top1 accuracy for every taxon at different taxonomy levels
"""
import pandas as pd


def taxon_accuracy(accuracy_data: dict[str, dict], label_info: dict[str, list[str]]):
    """
    returns top1 accuracy for every taxon at different ranks
    """
    final_data = {}
    final_data[
        "About"
    ] = "[Top1 Accuracy, Total Test Points] for each taxon at different ranks"

    # Family
    family_list = label_info["family"]
    f_data = accuracy_data["family"]
    family_data = {}

    for key in f_data.keys():
        family_data[family_list[key]] = [f_data[key][0], f_data[key][3]]

    family_data = dict(
        sorted(family_data.items(), key=lambda item: item[1], reverse=True)
    )
    final_data["family"] = family_data

    # Genus
    genus_list = label_info["genus"]
    g_data = accuracy_data["genus"]
    genus_data = {}

    for key in g_data.keys():
        genus_data[genus_list[key]] = [g_data[key][0], g_data[key][3]]

    genus_data = dict(
        sorted(genus_data.items(), key=lambda item: item[1], reverse=True)
    )
    final_data["genus"] = genus_data

    # Species
    species_list = label_info["species"]
    s_data = accuracy_data["species"]
    species_data = {}

    for key in s_data.keys():
        species_data[species_list[key]] = [s_data[key][0], s_data[key][3]]

    species_data = dict(
        sorted(species_data.items(), key=lambda item: item[1], reverse=True)
    )
    final_data["species"] = species_data

    return final_data


def add_taxon_accuracy_to_species_checklist(
    species_checklist: pd.DataFrame, taxa_accuracy: dict[str, dict]
):
    """
    Appends the calculated accuracy to the species checklist for every species entry
    """

    species_accuracy = taxa_accuracy["species"]
    checklist_columns = species_checklist.columns.values.tolist()
    checklist_columns.extend(["accuracy", "num_of_train_images", "num_of_test_images"])
    updated_checklist = pd.DataFrame(columns=checklist_columns, dtype=object)
    train_images_per = 0.75  # percentage of images used for training
    test_images_per = 0.15  # percentage of images used for testing

    for _, row in species_checklist.iterrows():
        species = row["gbif_species_name"]
        row_data = row.tolist()

        if species in species_accuracy.keys():
            accuracy = float(species_accuracy[species][0])
            num_test_points = int(species_accuracy[species][1])
            num_train_points = int(
                train_images_per * (num_test_points / test_images_per)
            )
        else:
            accuracy = float(-1)
            num_test_points = -1
            num_train_points = -1

        row_data.extend([accuracy, num_train_points, num_test_points])
        updated_checklist = pd.concat(
            [
                updated_checklist,
                pd.DataFrame(
                    [row_data],
                    columns=checklist_columns,
                ),
            ],
            ignore_index=True,
        )

    return updated_checklist
