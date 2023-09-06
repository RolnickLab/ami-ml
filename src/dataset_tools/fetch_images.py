#!/usr/bin/env python
# coding: utf-8

""" Image fetching script using Darwin Core Archive files
"""

import json
import os
import shutil
import urllib.request
from functools import partial
from multiprocessing import Pool

import click
from utils import get_image_path, load_dwca_data


def get_and_verify_image_path(image_data, dataset_path: str):
    image_path = os.path.join(dataset_path, get_image_path(image_data))

    dirs = os.path.dirname(image_path)
    if not os.path.isdir(dirs):
        try:
            os.makedirs(dirs)
        except OSError:
            print(f"Directory {dirs} can not be created")
            return None

    return image_path


def try_copy_from_cache(image_path: str, dataset_path: str, cache_path: str):
    if cache_path is not None:
        image_cache = os.path.relpath(image_path, dataset_path)
        image_cache = os.path.join(cache_path, image_cache)

        if os.path.isfile(image_cache) and not os.path.isfile(image_path):
            shutil.copyfile(image_cache, image_path)
            return True

    return False


def fetch_image(image_data, dataset_path: str, cache_path: str):
    url = image_data["identifier"]
    image_path = get_and_verify_image_path(image_data, dataset_path)

    if image_path is not None:
        cached = try_copy_from_cache(image_path, dataset_path, cache_path)

        if not cached and not os.path.isfile(image_path):
            try:
                urllib.request.urlretrieve(url, image_path)
                print(f"Image fetched from {url}")
            except Exception as e:
                print(f"Error fetching {url}")
                print(e)


@click.command(context_settings={"show_default": True})
@click.option(
    "--dataset_path", type=str, required=True, help="Folder to save images to"
)
@click.option(
    "--cache_path",
    type=str,
    help=(
        "Folder containing cached images. If provided, the script will try copy"
        " images from cache before trying fetch them."
    ),
)
@click.option(
    "--num_workers",
    type=int,
    default=8,
    help="Number of processes to download in images in parallel",
)
@click.option(
    "--dwca_file",
    type=str,
    required=True,
    help="Darwin Core Archive file",
)
@click.option(
    "--subset_list",
    type=str,
    help=(
        "JSON file with the list of keys to be fetched."
        "If provided, only occorrences with these keys will be fetched. Use the option"
        " --subset_key to define the field to be used for filtering."
    ),
)
@click.option(
    "--subset_key",
    type=str,
    default="speciesKey",
    help=(
        "DwC-A field to use for filtering occurrences to be fetched. "
        "See --subset_list"
    ),
)
def main(
    dwca_file: str,
    num_workers: int,
    dataset_path: str,
    cache_path: str,
    subset_list: str,
    subset_key: str,
):
    gbif_metadata = load_dwca_data(dwca_file)
    if subset_list is not None:
        with open(subset_list) as f:
            keys_list = json.load(f)
            keys_list = [int(x) for x in keys_list]
        gbif_metadata = gbif_metadata[gbif_metadata[subset_key].isin(keys_list)].copy()

    _, list_images = zip(*gbif_metadata.iterrows())

    fetch_image_f = partial(
        fetch_image, dataset_path=dataset_path, cache_path=cache_path
    )
    with Pool(processes=num_workers) as pool:
        pool.map(fetch_image_f, list_images)


if __name__ == "__main__":
    main()
