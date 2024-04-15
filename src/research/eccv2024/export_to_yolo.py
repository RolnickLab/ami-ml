"""
Author: Aditya Jain
Date last modified: November 2, 2023
About: Export AMI annotated data to the yolo format
"""

import json
import os
import pandas as pd
import urllib.request
from tqdm import tqdm
from PIL import Image

unidentfiable_cnt = 0

def get_raw_image_dim(img_path: str):
    """Get raw image dimensions"""

    # Read the raw image
    try:
        raw_image = Image.open(img_path)
        img_width, img_height = raw_image.size
    except:
        raise Exception(f"Issue with image {img_path}")

    return img_width, img_height


def convert_to_yolo_dim(x: float, y: float, w: float, h: float):
    """Convert box exported from Label Studio to YOLO format"""

    x_yolo = min((x + w / 2) / 100, 1)
    y_yolo = min((y + h / 2) / 100, 1)
    w_yolo = min(w / 100, 1)
    h_yolo = min(h / 100, 1)

    return round(x_yolo, 5), round(y_yolo, 5), round(w_yolo, 5), round(h_yolo, 5)


def create_categories_dict(yolo_data_dir: str):
    """Create a dictionary using json file with id and names of the classes"""

    filepath = os.path.join(yolo_data_dir, "notes.json")
    classes = json.load(open(filepath))
    inv_dictionary = dict(list(enumerate([x["name"] for x in classes["categories"]])))
    return {v: k for k, v in inv_dictionary.items()}


def get_only_taxon_labels(annotations: list[dict]):
    """Process the original annotation list
    to get only taxonomy labels when available
    """

    processed_list = []
    item_ids = []

    # First add all "taxonomy" labels
    for item in annotations:
        if item["type"] == "taxonomy":
            processed_list.append(item)
            item_ids.append(item["id"])

    # Add additional "labels" items only if
    # it does not have its corresponding
    # "taxonomy" label
    for item in annotations:
        if item["type"] == "labels" and item["id"] not in item_ids \
        and item["value"]["labels"] and item["value"]["labels"][0]!="Unclassified":
            processed_list.append(item)

    return processed_list


def filter_label_name(annotation: dict):
    """Choose the lower-most class for the fine-grained classification
    or the course label for the binary classification
    """

    if annotation["type"] == "labels":
        if annotation["value"]["labels"]: return annotation["value"]["labels"][0]

    elif annotation["type"] == "taxonomy":
        # Sort annotation based on higher to lower taxon classification
        taxonomy_labels = sorted(annotation["value"]["taxonomy"], key=len)
        return taxonomy_labels[-1][-1]

        # TO DO: The above return doesn't take into account two cases:
        # 1. The two labels assigned at the same level
        # (currently assigning second one; for no reason)
        # 2. The two labels might diverge at a higher taxon

    else:
        raise Exception("Annotation can onle be of type labels or taxonomy.")


def check_for_synonyms(label: str, sp_list: pd.DataFrame):
    """Return GBIF accepted names for synonyms"""
    
    a = label
    pass

def export_to_yolo(data: list[dict], output_dir: str, sp_list: pd.DataFrame):
    """Main function for exporting the data to yolo format"""

    global unidentfiable_cnt

    # Create the images folder if it does not exist
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Create the label folder if it does not exist
    labels_dir = os.path.join(output_dir, "labels")
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # Create the metadata folder if it does not exist
    metadata_dir = os.path.join(output_dir, "metadata")
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    # Get 0-indexed categories
    categories = create_categories_dict(output_dir)

    # Iterate over each raw annotated image
    for i in tqdm(range(len(data))):
        # Select the data point
        data_point = data[i]

        # Fetch image if it does not exist
        image_url = data_point["data"]["image"]
        image_name = os.path.basename(os.path.normpath(image_url))
        image_path = os.path.join(image_dir, image_name)
        if not os.path.isfile(image_path):
            print(f"Image {image_name} didn't exist.")
            try:
                urllib.request.urlretrieve(image_url, image_path)
            except Exception as e:
                print("Error fetching {image_name}. The error is: {e}.", flush=True)

        # Take results from reviewer annotations, if available
        annotator_results = data_point["annotations"][0]["result"]
        reviewer_results = data_point["annotations"][0]["reviews"]
        if reviewer_results and reviewer_results[0]["fixed_annotation_history"]:
            annotations = reviewer_results[0]["fixed_annotation_history_result"]
        else:
            annotations = annotator_results

        # Filter only taxon labels when available
        taxon_annotations = get_only_taxon_labels(annotations)

        # Iterate over all annotations and save them
        annot_yolo_format = ""
        label_filename = os.path.splitext(image_name)[0] + ".txt"
        for item in taxon_annotations:
            # Get bounding box coordinates
            xywh = convert_to_yolo_dim(
                item["value"]["x"],
                item["value"]["y"],
                item["value"]["width"],
                item["value"]["height"],
            )

            # Check for synonym names and return only GBIF accepted names
            label_name = filter_label_name(item)   
            label_name = check_for_synonyms(label_name, sp_list)         

            #Find the numeric label id from the json file
            try:
                label_id = str(categories[label_name])
            except:
                raise (f"{label_name} class is not found in the notes.json label file")

            # Add entry to the file
            yolo_row = label_id + " " + str(xywh)[1:-1].replace(",", "")
            annot_yolo_format += yolo_row + "\n"

        # Write the annotation file to the disk
        try:
            with open(os.path.join(labels_dir, label_filename), "w") as f:
                f.write(annot_yolo_format)
                f.close()
        except Exception as e:
            print(e)

        # Find the region information
        deployment = data_point["data"]["deployment"]
        if "Panama" in deployment:
            region = "CentralAmerica"
        elif "UK" in deployment or "Denmark" in deployment:
            region = "WesternEurope"
        elif "Quebec" in deployment or "Vermont" in deployment:
            region = "NorthEasternAmerica"
        else:
            raise Exception("Region information not found in the data")

        # Write the image metadata information to file
        image_width, image_height = get_raw_image_dim(image_path)
        metadata = {"region": region, "image_width": image_width, "image_height": image_height}
        metadata_filename = os.path.splitext(image_name)[0] + ".json"
        with open(os.path.join(metadata_dir, metadata_filename), "w") as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    # Input data
    annotation_file = (
        "/home/mila/a/aditya.jain/scratch/eccv2024_data/annotated-tasks-20240110.json"
    )
    annotation_data = json.load(open(annotation_file))
    yolo_data_dir = "/home/mila/a/aditya.jain/scratch/eccv2024_data/ami_traps_dataset"
    species_list = pd.read_csv("/home/mila/a/aditya.jain/mothAI/species_lists/quebec-vermont-uk-denmark-panama_checklist_20231124.csv")
    export_to_yolo(annotation_data, yolo_data_dir, species_list)
