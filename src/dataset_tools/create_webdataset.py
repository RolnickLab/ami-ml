#!/usr/bin/env python
# coding: utf-8

"""
Create webdataset
"""

import json
import os

import pandas as pd
import PIL
import torch
import webdataset as wds
from PIL import Image
from torchvision import transforms

from src.dataset_tools.utils import set_random_seeds, square_crop


def _prepare_json_data(sample_metadata, columns_to_json):
    metadata = {}
    columns_to_json = columns_to_json.split(",")

    for column in columns_to_json:
        metadata[column] = sample_metadata[column]

    return metadata


def _crop_to_bbox(image, md_preds):
    if len(md_preds["detections"]) > 0:
        max_detection_conf = 0.0
        (x, y, width, height) = 0, 0, 0, 0
        for det in md_preds["detections"]:
            if det["conf"] > max_detection_conf:
                max_detection_conf = det["conf"]
                (x, y, width, height) = det["bbox"]
        if max_detection_conf > 0.0:
            image = square_crop(image, x, y, width, height)

            return image

    return image


def _get_resize_transform(resize_min_size):
    if resize_min_size:
        return transforms.Compose(
            [transforms.Resize(resize_min_size), transforms.ToTensor()]
        )
    return None


def _get_category_map(
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


def _dataset_preprocessing(
    annotations_csv: str,
    category_map_json: str,
    label_column: str,
    megadetector_results_json: str,
    save_category_map_json: str,
    shuffle_images: bool,
) -> (dict, pd.DataFrame, dict):
    md_results = None

    if megadetector_results_json is not None:
        with open(megadetector_results_json, "r") as f:
            md_results = json.load(f)
        md_results = pd.DataFrame(md_results["images"])
        md_results = md_results.set_index("file")

    dataset_df = pd.read_csv(annotations_csv)

    if shuffle_images:
        dataset_df = dataset_df.sample(frac=1)

    categories_map = _get_category_map(
        dataset_df, label_column, category_map_json, save_category_map_json
    )

    return categories_map, dataset_df, md_results


# TODO : Refactor _create_samples
# Pretty sure there's a way to refactor for less complexity
# Instead of checking `resize_min_size` and `md_results` each loop,
def _create_samples(
    dataset_path: str,
    categories_map: dict,
    dataset_df: pd.DataFrame,
    md_results: pd.DataFrame,
    label_column: str,
    image_path_column: str,
    columns_to_json: str,
    resize_min_size: int,
):
    resize_transform = _get_resize_transform(resize_min_size)

    for _, row in dataset_df.iterrows():
        fpath = os.path.join(dataset_path, row[image_path_column])
        if not os.path.isfile(fpath):
            print(f"File {fpath} not found", flush=True)
            continue

        if resize_min_size is not None or md_results is not None:
            try:
                with Image.open(fpath) as image:
                    image = image.convert("RGB")
                    if md_results is not None:
                        try:
                            md_preds = md_results.loc[row[image_path_column]]
                            image = _crop_to_bbox(image, md_preds)
                        except KeyError:
                            print(
                                "Skipping crop for the image: ", row[image_path_column]
                            )
                        except TypeError:
                            print(
                                "Skipping crop for the image: ", row[image_path_column]
                            )
                    if resize_min_size is not None:
                        image = resize_transform(image)

                    image = image.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
                    image_data = image.numpy().transpose(1, 2, 0)
            except PIL.UnidentifiedImageError:
                print(f"Unidentified Image Error on file {fpath}", flush=True)
                continue
            except OSError:
                print(f"OSError Error on file {fpath}", flush=True)
                continue
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
            json_data = _prepare_json_data(row, columns_to_json)
            sample["json"] = json_data

        yield sample


def create_webdataset(
    annotations_csv: str,
    dataset_path: str,
    webdataset_pattern: str,
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
    with wds.ShardWriter(webdataset_pattern, maxsize=max_shard_size) as sink:
        (
            categories_map,
            dataset_df,
            md_results,
        ) = _dataset_preprocessing(
            annotations_csv=annotations_csv,
            category_map_json=category_map_json,
            label_column=label_column,
            megadetector_results_json=megadetector_results_json,
            save_category_map_json=save_category_map_json,
            shuffle_images=shuffle_images,
        )
        dataset = _create_samples(
            categories_map=categories_map,
            dataset_df=dataset_df,
            md_results=md_results,
            dataset_path=dataset_path,
            image_path_column=image_path_column,
            label_column=label_column,
            columns_to_json=columns_to_json,
            resize_min_size=resize_min_size,
        )
        for sample in dataset:
            sink.write(sample)
