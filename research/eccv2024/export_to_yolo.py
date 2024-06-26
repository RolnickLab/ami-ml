"""
Author: Aditya Jain
Date last modified: November 2, 2023
About: Export AMI annotated data to the yolo format
"""

import json
import os
import urllib.request
from pathlib import Path

# 3rd party packages
from PIL import Image
from tqdm import tqdm


def _get_raw_image_dim(img_path: str):
    """Get raw image dimensions"""

    # Read the raw image
    try:
        raw_image = Image.open(img_path)
        img_width, img_height = raw_image.size
    except FileNotFoundError:
        raise Exception(f"Image file not found: {img_path}")
    except Exception as e:
        raise Exception(f"Error reading image: {img_path}. {str(e)}")

    return img_width, img_height


def _convert_to_yolo_dim(x: float, y: float, w: float, h: float):
    """Convert box exported from Label Studio to YOLO format"""

    x_yolo = min((x + w / 2) / 100, 1)
    y_yolo = min((y + h / 2) / 100, 1)
    w_yolo = min(w / 100, 1)
    h_yolo = min(h / 100, 1)

    return round(x_yolo, 5), round(y_yolo, 5), round(w_yolo, 5), round(h_yolo, 5)


def _create_categories_dict(yolo_data_dir: str):
    """Create a dictionary using json file with id and names of the classes"""

    filepath = Path(yolo_data_dir) / "notes.json"
    with open(filepath) as f:
        classes = json.load(f)
    inv_dictionary = dict(list(enumerate([x["name"] for x in classes["categories"]])))
    return {v: k for k, v in inv_dictionary.items()}


def _get_only_taxon_labels(annotations: list[dict]):
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
        if (
            item["type"] == "labels"
            and item["id"] not in item_ids
            and item["value"]["labels"]
            and item["value"]["labels"][0] != "Unclassified"
        ):
            processed_list.append(item)

    return processed_list


def _filter_label_name(annotation: dict):
    """Choose the lower-most class for the fine-grained classification
    or the course label for the binary classification
    """

    if annotation["type"] == "labels":
        if annotation["value"]["labels"]:
            return annotation["value"]["labels"][0]

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


def export_to_yolo(data: list[dict], output_dir: str):
    """Main function for exporting the data to yolo format"""

    # Create the images folder if it does not exist
    image_dir = Path(output_dir) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Create the label folder if it does not exist
    labels_dir = Path(output_dir) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Create the metadata folder if it does not exist
    metadata_dir = Path(output_dir) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Get 0-indexed categories
    categories = _create_categories_dict(output_dir)

    # Iterate over each raw annotated image
    for i in tqdm(range(len(data))):
        # Select the data point
        data_point = data[i]

        # Fetch image if it does not exist
        image_url = data_point["data"]["image"]
        image_name = Path(image_url).name
        image_path = image_dir / image_name
        if not image_path.is_file():
            print(f"Image {image_name} didn't exist.")
            try:
                urllib.request.urlretrieve(image_url, str(image_path))
            except Exception as e:
                print(f"Error fetching {image_name}. The error is: {e}.", flush=True)

        # Take results from reviewer annotations, if available
        annotator_results = data_point["annotations"][0]["result"]
        reviewer_results = data_point["annotations"][0]["reviews"]
        if reviewer_results and reviewer_results[0]["fixed_annotation_history"]:
            annotations = reviewer_results[0]["fixed_annotation_history_result"]
        else:
            annotations = annotator_results

        # Filter only taxon labels when available
        taxon_annotations = _get_only_taxon_labels(annotations)

        # Iterate over all annotations and save them
        annot_yolo_format = ""
        label_filename = Path(image_name).stem + ".txt"
        for item in taxon_annotations:
            # Get bounding box coordinates
            xywh = _convert_to_yolo_dim(
                item["value"]["x"],
                item["value"]["y"],
                item["value"]["width"],
                item["value"]["height"],
            )

            # Get the label name
            label_name = _filter_label_name(item)

            # Find the numeric label id from the json file
            try:
                label_id = str(categories[label_name])
            except KeyError:
                raise Exception(
                    f"{label_name} class is not found in the notes.json label file"
                )

            # Add entry to the file
            yolo_row = label_id + " " + str(xywh)[1:-1].replace(",", "")
            annot_yolo_format += yolo_row + "\n"

        # Write the annotation file to the disk
        try:
            with open(labels_dir / label_filename, "w") as f:
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
        image_width, image_height = _get_raw_image_dim(str(image_path))
        metadata = {
            "region": region,
            "image_width": image_width,
            "image_height": image_height,
        }
        metadata_filename = Path(image_name).stem + ".json"
        with open(metadata_dir / metadata_filename, "w") as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    ECCV2024_DATA = os.getenv("ECCV2024_DATA")
    SPECIES_LISTS_DIR = os.getenv("SPECIES_LISTS_DIR")
    MASTER_SPECIES_LIST = os.getenv("MASTER_SPECIES_LIST")

    annotation_file = f"{ECCV2024_DATA}/annotated-tasks-20240110.json"
    with open(annotation_file) as f:
        annotation_data = json.load(f)
    yolo_data_dir = f"{ECCV2024_DATA}/ami_traps_dataset"
    export_to_yolo(annotation_data, yolo_data_dir)
