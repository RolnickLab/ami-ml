#!/usr/bin/env python
# coding: utf-8

""" Image fetching script using Darwin Core Archive files
"""

import os
import shutil
import urllib.request
from functools import partial
from multiprocessing import Pool

import click
import pandas as pd
from dwca.read import DwCAReader


def get_and_verify_image_path(image_data, dataset_path: str):
    write_loc = os.path.join(dataset_path, image_data["datasetKey"])
    if not os.path.isdir(write_loc):
        try:
            os.makedirs(write_loc)
        except OSError:
            print(f"Directory {write_loc} can not be created")
            return None

    image_path = write_loc + os.sep + str(image_data["coreid"])
    if image_data["count"] > 0:
        image_path = image_path + "_" + str(image_data["count"])

    return image_path + ".jpg"


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
    cached = try_copy_from_cache(image_path, dataset_path, cache_path)

    if not cached and not os.path.isfile(image_path):
        try:
            urllib.request.urlretrieve(url, image_path)
            print(f"Image fetched from {url}")
        except Exception as e:
            print(f"Error fetching {url}")
            print(e)


def load_dwca_data(dwca_file: str):
    with DwCAReader(dwca_file) as dwca:
        media_df = dwca.pd_read("multimedia.txt", parse_dates=True, on_bad_lines="skip")
        occ_df = dwca.pd_read("occurrence.txt", parse_dates=True, on_bad_lines="skip")

    media_df = media_df[["coreid", "identifier"]].copy()
    occ_df = occ_df[["id", "datasetKey", "speciesKey"]].copy()

    images = pd.merge(media_df, occ_df, how="inner", left_on="coreid", right_on="id")
    images["count"] = images.groupby("coreid").cumcount()

    return images


@click.command(context_settings={"show_default": True})
@click.option(
    "--dataset_path", type=str, required=True, help="Folder to save images to"
)
@click.option(
    "--cache_path",
    type=str,
    help=(
        "Folder containing cached images. If provied, the script will try copy"
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
def main(dwca_file: str, num_workers: int, dataset_path: str, cache_path: str):
    gbif_metadata = load_dwca_data(dwca_file)

    _, list_images = zip(*gbif_metadata.iterrows())

    fetch_image_f = partial(
        fetch_image, dataset_path=dataset_path, cache_path=cache_path
    )
    with Pool(processes=num_workers) as pool:
        pool.map(fetch_image_f, list_images)


if __name__ == "__main__":
    main()
