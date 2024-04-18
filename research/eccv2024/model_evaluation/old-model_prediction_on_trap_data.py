"""
Author: Aditya Jain
Date last modified: November 12, 2023
About: Predictions of GBIF-trained models on AMI benchmark data
"""

import os
import json
import pandas as pd
import argparse
from PIL import Image
import torch
from torchvision import transforms

from model_inference import ModelInference


def moth_ids_to_names(moth_pred: list[list], species_list: pd.DataFrame):
    """Convert numeric to species name lables"""
    new_model_pred = []

    for pred in moth_pred:
        numeric_id, conf = int(pred[0]), pred[1]

        # Find the class name for the corresponding numeric id
        name_id = "NA"
        try:
            name_id = species_list.loc[
                species_list["accepted_taxon_key"] == numeric_id, "gbif_species"
            ].values[0]
        except:
            print(f"Taxon key {numeric_id} is not found in the database.")

        new_model_pred.append([name_id, conf])

    return new_model_pred


def predict_on_trap_data(
    data_dir: str,
    binary_model: str,
    binary_model_type: str,
    moth_model: str,
    moth_model_type: str,
    category_map_binary_model: str,
    category_map_moth_model: str,
    region: str,
    global_species_list: str,
):
    """Main function for model prediction on trap data"""

    # Get the image list and other relevant data
    model_dir = os.path.join(data_dir, "models")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ami_data_dir = os.path.join(data_dir, "ami_traps_dataset")
    image_dir = os.path.join(ami_data_dir, "images")
    image_list = os.listdir(image_dir)
    labels_dir = os.path.join(ami_data_dir, "labels")
    metadata_dir = os.path.join(ami_data_dir, "metadata")

    # Build the binary classification model
    model_path = os.path.join(model_dir, binary_model)
    categ_map_path = os.path.join(model_dir, category_map_binary_model)
    binary_classifier = ModelInference(
        model_path, binary_model_type, categ_map_path, device
    )

    # Build the moth classification model
    # model_path = os.path.join(model_dir, moth_model)
    model_path = os.path.join(model_dir, "all-architectures", moth_model) # TEMP CHANGE
    categ_map_path = os.path.join(model_dir, category_map_moth_model)
    moth_classifier = ModelInference(
        model_path, moth_model_type, categ_map_path, device
    )

    # Get ground truth class names
    gt_class_list = json.load(open(os.path.join(ami_data_dir, "notes.json")))[
        "categories"
    ]

    # Read the global moth species list
    species_list = pd.read_csv(global_species_list)

    # Create the prediction folder if it does not exist
    # model_pred_dir = os.path.join(ami_data_dir, "model_predictions")
    model_pred_dir = os.path.join(ami_data_dir, "model_predictions", "all-architectures", moth_model.split(".")[0]) # TEMP CHANGE
    if not os.path.exists(model_pred_dir):
        os.makedirs(model_pred_dir)

    # Make prediction over each image iteratively
    for image_name in image_list:
        img_basename = os.path.splitext(image_name)[0]

        # Fetch the region name for the image
        metadata_file = os.path.join(metadata_dir, img_basename + ".json")
        metadata = json.load(open(metadata_file))        

        # If in the region, make model prediction
        if metadata["region"] == region:
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

            # Dictionary for storing prediction results
            prediction_results = []
        
            # Get the ground truth bounding box and label
            labels = open(os.path.join(labels_dir, img_basename + ".txt"), "r")

            # Iterate over each annotation separately
            for line in labels:
                label_id, x, y, w, h = (
                    int(line.split()[0]),
                    float(line.split()[1]),
                    float(line.split()[2]),
                    float(line.split()[3]),
                    float(line.split()[4]),
                )

                # Get the ground truth class name
                for class_entry in gt_class_list:
                    if class_entry["id"] == label_id:
                        label_name = class_entry["name"]
                        label_rank = class_entry["rank"]
                        break

                # Get the insect crop
                x_start = int((x - w / 2) * img_width)
                y_start = int((y - h / 2) * img_height)
                w_px, h_px = int(w * img_width), int(h * img_height)
                cropped_image = image[
                    :, y_start : y_start + h_px, x_start : x_start + w_px
                ]

                # Binary model prediction
                binary_pred = binary_classifier.predict(cropped_image)

                # Moth model prediction
                if label_rank != "NA":
                    moth_pred = moth_classifier.predict(cropped_image)
                    moth_pred = moth_ids_to_names(moth_pred, species_list)
                else:
                    moth_pred = "Not required"

                # Store and save the results
                bbox_result = {
                    "ground_truth": [label_name, label_rank],
                    "binary_classification": binary_pred,
                    "moth_classification": moth_pred,
                    "bbox_coordinates": [str(x), str(y), str(w), str(h)]
                }
                prediction_results.append(bbox_result)

            # Write the prediction and ground truth on the disk
            with open(
                os.path.join(
                    model_pred_dir, img_basename + "_" + moth_model_type + ".json"
                ),
                "w",
            ) as f:
                json.dump(prediction_results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        help="Path to the root directory containing the annotation data.",
        required=True,
    )
    parser.add_argument(
        "--binary_model",
        help="Name of the binary classification model.",
        required=True,
    )
    parser.add_argument(
        "--binary_model_type",
        help="Type of the binary classification model.",
        required=True,
    )
    parser.add_argument(
        "--moth_model",
        help="Name of the moth classification model.",
        required=True,
    )
    parser.add_argument(
        "--moth_model_type",
        help="Type of the moth classification model.",
        required=True,
    )
    parser.add_argument(
        "--category_map_binary_model",
        help="Name of the category map for the binary classification model.",
        required=True,
    )
    parser.add_argument(
        "--category_map_moth_model",
        help="Name of the category map for the moth classification model.",
        required=True,
    )
    parser.add_argument(
        "--region",
        help="Region to which the model belongs.",
        required=True,
    )
    parser.add_argument(
        "--global_species_list",
        help="Path to the global moth species list.",
        required=True,
    )

    args = parser.parse_args()

    predict_on_trap_data(
        args.data_dir,
        args.binary_model,
        args.binary_model_type,
        args.moth_model,
        args.moth_model_type,
        args.category_map_binary_model,
        args.category_map_moth_model,
        args.region,
        args.global_species_list,
    )
