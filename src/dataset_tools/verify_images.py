#!/usr/bin/env python
# coding: utf-8

""" Image verification script using Darwin Core Archive files
"""

import json
import os
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import pandas as pd
import PIL
from PIL import Image

from src.dataset_tools.utils import get_image_path, load_dwca_data

ERROR_CSV = ".error.csv"


def _get_image_info(image_path):
    file_size = -1
    image_width = -1
    image_height = -1
    fetch_date = -1
    corrupted = False
    if os.path.isfile(image_path):
        try:
            image = Image.open(image_path)
            image = image.convert("RGB")

            file_size = os.path.getsize(image_path)
            fetch_date = os.path.getmtime(image_path)
            fetch_date = datetime.fromtimestamp(fetch_date).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            image_width, image_height = image.size

            print(f"Verified image {image_path}", flush=True)

        except PIL.UnidentifiedImageError:
            print(f"Unidentified Image Error on file {image_path}", flush=True)
            corrupted = True
        except OSError:
            print(f"OSError Error on file {image_path}", flush=True)
            corrupted = True

    return file_size, image_width, image_height, fetch_date, corrupted


def _verify_image(image_data, dataset_path: str):
    image_path = os.path.join(dataset_path, image_data["image_path"])
    file_size, width, height, fetch_date, corrupted = _get_image_info(image_path)

    verification_metadata = {
        "image_path": image_data["image_path"],
        "width": width,
        "height": height,
        "fetch_date": fetch_date,
        "file_size": file_size,
        "corrupted": corrupted,
    }
    return verification_metadata


def verify_images(
    dwca_file: str,
    resume_from_ckpt: str,
    save_freq: int,
    num_workers: int,
    dataset_path: str,
    results_csv: str,
    subset_list: str,
    subset_key: str,
):
    gbif_metadata = load_dwca_data(dwca_file)
    if subset_list is not None:
        with open(subset_list) as f:
            keys_list = json.load(f)
            keys_list = [int(x) for x in keys_list]
        gbif_metadata = gbif_metadata[gbif_metadata[subset_key].isin(keys_list)].copy()

    gbif_metadata = gbif_metadata[~gbif_metadata.datasetKey.isna()]
    gbif_metadata["image_path"] = gbif_metadata.apply(get_image_path, axis=1)

    verif_df = pd.DataFrame()
    errors_df = pd.DataFrame()
    gbif_metadata_unverified = gbif_metadata

    if resume_from_ckpt is not None:
        errors_df, gbif_metadata_unverified, verif_df = _resume_from_ckpt(
            gbif_metadata, resume_from_ckpt
        )

    _, images_list = zip(*gbif_metadata_unverified.iterrows())
    partial_size = save_freq if save_freq > 0 else len(images_list)
    images_list = [
        images_list[i : i + partial_size]
        for i in range(0, len(images_list), partial_size)
    ]

    verify_image_f = partial(_verify_image, dataset_path=dataset_path)

    for images_list_partial in images_list:
        with Pool(processes=num_workers) as pool:
            results = pool.map(verify_image_f, images_list_partial)
            errors = [x for x in results if x["corrupted"]]
            results = [x for x in results if x["file_size"] > 0]

            verif_df_partial = pd.DataFrame(results)
            errors_partial = pd.DataFrame(errors)

            if not verif_df_partial.empty:
                verif_df = pd.concat([verif_df, verif_df_partial], ignore_index=True)
                temp_df = pd.merge(
                    gbif_metadata, verif_df, how="inner", on="image_path"
                )
                temp_df.to_csv(results_csv, index=False)
                print(f"Partial verification saved to {results_csv}")

            if not errors_partial.empty:
                errors_partial = errors_partial["image_path"]
                errors_df = pd.concat([errors_df, errors_partial], ignore_index=True)
                errors_df.to_csv(results_csv + ERROR_CSV, index=False)

    verif_df = pd.merge(gbif_metadata, verif_df, how="inner", on="image_path")
    verif_df.to_csv(results_csv, index=False)
    errors_df.to_csv(results_csv + ERROR_CSV, index=False)
    print(f"Final verification results saved to {results_csv}")


def _resume_from_ckpt(gbif_metadata, resume_from_ckpt):
    verif_df = pd.read_csv(resume_from_ckpt)
    if os.path.isfile(resume_from_ckpt + ERROR_CSV):
        errors_df = pd.read_csv(resume_from_ckpt + ERROR_CSV)
    else:
        errors_df = pd.DataFrame()
    if not verif_df.empty:
        verif_df = verif_df[["image_path", "width", "height", "fetch_date"]].copy()
        gbif_metadata_unverified = gbif_metadata[
            ~gbif_metadata.image_path.isin(verif_df.image_path)
        ]
    else:
        gbif_metadata_unverified = gbif_metadata
    return errors_df, gbif_metadata_unverified, verif_df
