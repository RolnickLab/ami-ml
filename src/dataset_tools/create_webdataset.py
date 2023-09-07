#!/usr/bin/env python
# coding: utf-8

""" Create webdataset
"""

import json
import os

import click
import pandas as pd
import PIL
import torch
import webdataset as wds
from PIL import Image
from torchvision import transforms
from utils import set_random_seeds


def prepare_json_data(sample_metadata, columns_to_json):
    metadata = {}
    columns_to_json = columns_to_json.split(",")

    for column in columns_to_json:
        metadata[column] = sample_metadata[column]

    return metadata


def square_crop(image, x, y, width, height):
    """
    Try to crop a square region centered at the bounding box provided. The square
    region cropped has a padding corresponding to 10% of the side of a tight
    square cropping around the bbox. If the square region would crop outside the
    image size, the region will be capped to be at the borders. In this case, the
    region is not guaranteed to be square.

    Args:
      image: a PIL image
      x (float): left coordinate
      y (float): upper coordinate
      width (float): bbox width
      height (float): bbox height

    Returns:
      image: a cropped image
    """

    image_width, image_height = image.size
    x = x * image_width
    y = y * image_height
    width = width * image_width
    height = height * image_height

    side = 1.2 * max(width, height)
    center_x = x + width / 2
    center_y = y + height / 2

    upper = max(0.0, center_y - side / 2)
    left = max(0.0, center_x - side / 2)
    lower = min(image_height, center_y + side / 2)
    right = min(image_width, center_x + side / 2)

    image = image.crop((int(left), int(upper), int(right), int(lower)))

    return image


def crop_to_bbox(image, md_preds):
    if md_preds["max_detection_conf"] > 0.0:
        for det in md_preds["detections"]:
            if det["conf"] == md_preds["max_detection_conf"]:
                (x, y, width, height) = det["bbox"]
                image = square_crop(image, x, y, width, height)

                return image

    return image


def get_resize_transform(resize_min_size):
    return transforms.Compose(
        [transforms.Resize(resize_min_size), transforms.ToTensor()]
    )


def get_category_map(
    dataset_df, label_column: str, category_map_json: str, save_category_map_json: str
):
    if category_map_json is not None:
        with open(category_map_json, "r") as f:
            categories_map = json.load(f)
    else:
        categories = sorted(list(dataset_df[label_column].unique()))
        categories_map = {str(categ): id for id, categ in enumerate(categories)}

        if save_category_map_json is not None:
            with open(save_category_map_json, "w") as f:
                json.dump(categories_map, f)

    return categories_map


def dataset_samples(
    annotations_csv: str,
    dataset_dir: str,
    shuffle_images: bool,
    label_column: str,
    image_path_column: str,
    category_map_json: str,
    save_category_map_json: str,
    megadetector_results_json: str,
    columns_to_json: str,
    resize_min_size: int,
):
    if megadetector_results_json is not None:
        with open(megadetector_results_json, "r") as f:
            md_results = json.load(f)
        md_results = pd.DataFrame(md_results["images"])
        md_results = md_results.set_index("file")
    else:
        md_results = None

    dataset_df = pd.read_csv(annotations_csv)
    if shuffle_images:
        dataset_df = dataset_df.sample(frac=1)
    categories_map = get_category_map(
        dataset_df, label_column, category_map_json, save_category_map_json
    )

    if resize_min_size is not None:
        resize_transform = get_resize_transform(resize_min_size)

    for _, row in dataset_df.iterrows():
        fpath = os.path.join(dataset_dir, row[image_path_column])
        if not os.path.isfile(fpath):
            print(f"File {fpath} not found", flush=True)
            continue

        if resize_min_size is not None or md_results is not None:
            try:
                image = Image.open(fpath)
                image = image.convert("RGB")
            except PIL.UnidentifiedImageError:
                print(f"Unidentified Image Error on file {fpath}", flush=True)
                continue
            except OSError:
                print(f"OSError Error on file {fpath}", flush=True)
                continue

            if md_results is not None:
                try:
                    md_preds = md_results.loc[row[image_path_column]]
                    image = crop_to_bbox(image, md_preds)
                except KeyError as err:
                    print("Skipping crop for the image: ", err)

            if resize_min_size is not None:
                image = resize_transform(image)

            image = image.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
            image_data = image.numpy().transpose(1, 2, 0)
        else:
            with open(fpath, "rb") as f:
                image_data = f.read()
        sample = {
            "__key__": os.path.splitext(row[image_path_column])[0]
            .lower()
            .replace(".", "_"),
            "jpg": image_data,
            "cls": categories_map[str(row[label_column])],
        }

        if columns_to_json is not None:
            json_data = prepare_json_data(row, columns_to_json)
            sample["json"] = json_data

        yield sample


@click.command(context_settings={"show_default": True})
@click.option(
    "--annotations_csv",
    type=str,
    required=True,
    help="Path to csv file containing the annotations",
)
@click.option(
    "--dataset_dir",
    type=str,
    required=True,
    help="Path to directory containing dataset images",
)
@click.option(
    "--webdataset_patern",
    type=str,
    required=True,
    help="Webdataset output file pattern",
)
@click.option(
    "--image_path_column",
    type=str,
    required=True,
    help="CSV column containing image file path",
)
@click.option(
    "--label_column",
    type=str,
    required=True,
    help="CSV column containing image label",
)
@click.option(
    "--max_shard_size",
    type=int,
    default=100 * 1024 * 1024,
    help="Maximun size of each shard",
)
@click.option(
    "--resize_min_size",
    type=int,
    help=(
        "Size which the shortest image side will be resized to. If it is not"
        " given, the original image is used withou resizing."
    ),
)
@click.option(
    "--shuffle_images",
    type=bool,
    default=True,
    help="Shufle images before to write to tar files",
)
@click.option(
    "--category_map_json",
    type=str,
    help=(
        "JSON containing the categories id map. If not provided, the"
        " category map will be infered from annotations csv."
    ),
)
@click.option(
    "--save_category_map_json",
    type=str,
    help=(
        "JSON containing the categories id map. If not provided, the"
        " category map will be infered from annotations csv."
    ),
)
@click.option(
    "--columns_to_json",
    type=str,
    help="List of columns from CSV file to save as metadata in a json file.",
)
@click.option(
    "--megadetector_results_json",
    type=str,
    help=(
        "Path to json file containing megadetector results. If provided, the"
        " images will be cropped to a squared region around the bbox with"
        " the highest confidence."
    ),
)
@click.option(
    "--random_seed",
    type=int,
    default=42,
    help="Random seed for reproductible experiments",
)
def main(
    annotations_csv: str,
    dataset_dir: str,
    webdataset_patern: str,
    image_path_column: str,
    label_column: str,
    max_shard_size: int,
    shuffle_images: bool,
    resize_min_size: int,
    category_map_json: str,
    save_category_map_json: str,
    columns_to_json: str,
    megadetector_results_json: str,
    random_seed: int,
):
    set_random_seeds(random_seed)
    with wds.ShardWriter(webdataset_patern, maxsize=max_shard_size) as sink:
        dataset = dataset_samples(
            annotations_csv,
            dataset_dir,
            shuffle_images,
            label_column,
            image_path_column,
            category_map_json,
            save_category_map_json,
            megadetector_results_json,
            columns_to_json,
            resize_min_size,
        )
        for sample in dataset:
            sink.write(sample)


if __name__ == "__main__":
    main()
