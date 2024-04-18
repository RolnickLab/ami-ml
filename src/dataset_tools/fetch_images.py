#!/usr/bin/env python
# coding: utf-8

""" Image fetching script using Darwin Core Archive files
"""

import json
import os
import shutil
from functools import partial
from multiprocessing import Pool

import pandas as pd
import requests

from src.dataset_tools.utils import get_image_path, load_dwca_data, set_random_seeds


def _get_and_verify_image_path(image_data, dataset_path: str):
    image_path = os.path.join(dataset_path, get_image_path(image_data))

    dirs = os.path.dirname(image_path)
    if not os.path.isdir(dirs):
        try:
            os.makedirs(dirs)
        except OSError:
            print(f"Directory {dirs} can not be created")
            return None

    return image_path


def _try_copy_from_cache(image_path: str, dataset_path: str, cache_path: str):
    if cache_path is not None:
        image_cache = os.path.relpath(image_path, dataset_path)
        image_cache = os.path.join(cache_path, image_cache)

        if os.path.isfile(image_cache) and not os.path.isfile(image_path):
            shutil.copyfile(image_cache, image_path)
            return True

    return False


def _url_retrieve(url, file_path, timeout):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:124.0) "
            "Gecko/20100101 Firefox/124.0"
        )
    }

    r = requests.get(url, timeout=(timeout, timeout * 10), headers=headers)
    open(file_path, "wb").write(r.content)


def _fetch_image(image_data, dataset_path: str, cache_path: str, timeout: int):
    url = image_data["identifier"]
    image_path = _get_and_verify_image_path(image_data, dataset_path)

    if image_path is not None:
        cached = _try_copy_from_cache(image_path, dataset_path, cache_path)

        if not cached and not os.path.isfile(image_path):
            try:
                _url_retrieve(url, image_path, timeout)
                print(f"Image fetched from {url}")
            except Exception as e:
                print(f"Error fetching {url}")
                print(e)


def _subset_list(gbif_metadata, subset_key, subset_list):
    with open(subset_list) as f:
        keys_list = json.load(f)
        keys_list = [int(x) for x in keys_list]
    gbif_metadata = gbif_metadata[gbif_metadata[subset_key].isin(keys_list)].copy()
    return gbif_metadata


def _group_by_category(gbif_metadata, num_images_per_category, subset_key):
    all_occ = (
        gbif_metadata.groupby(subset_key)
        .filter(lambda x: len(x) < num_images_per_category)
        .copy()
    )
    subsampling_occ = (
        gbif_metadata.groupby(subset_key)
        .filter(lambda x: len(x) >= num_images_per_category)
        .groupby(subset_key)
        .sample(num_images_per_category)
        .copy()
    )
    gbif_metadata = pd.concat([all_occ, subsampling_occ])
    return gbif_metadata


def fetch_images(
    dwca_file: str,
    num_workers: int,
    dataset_path: str,
    cache_path: str,
    subset_list: str,
    subset_key: str,
    num_images_per_category: int,
    request_timeout: int,
    random_seed: int,
):
    set_random_seeds(random_seed)
    gbif_metadata = load_dwca_data(dwca_file)
    if subset_list is not None:
        gbif_metadata = _subset_list(gbif_metadata, subset_key, subset_list)

    if num_images_per_category > 0:
        gbif_metadata = _group_by_category(
            gbif_metadata, num_images_per_category, subset_key
        )

    _, list_images = zip(*gbif_metadata.iterrows())

    fetch_image_f = partial(
        _fetch_image,
        dataset_path=dataset_path,
        cache_path=cache_path,
        timeout=request_timeout,
    )
    with Pool(processes=num_workers) as pool:
        pool.map(fetch_image_f, list_images)
