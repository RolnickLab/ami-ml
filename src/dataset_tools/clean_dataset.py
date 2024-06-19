#!/usr/bin/env python
# coding: utf-8

""" Clean dataset applying a set of rules to improve the quality of the dataset
"""
import pandas as pd

from src.dataset_tools.utils import get_image_path, load_dwca_data


def _load_data(dwca_file: str, verified_data_csv: str, life_stage_predictions: str):
    metadata = load_dwca_data(dwca_file)
    metadata["image_path"] = metadata.apply(get_image_path, axis=1)
    verified_data = pd.read_csv(verified_data_csv)
    verified_metadata = pd.merge(
        verified_data[["image_path", "width", "height", "fetch_date"]],
        metadata,
        how="inner",
        on="image_path",
    )
    if life_stage_predictions is not None:
        life_stage_preds = pd.read_csv(life_stage_predictions)
        verified_metadata = pd.merge(
            verified_metadata, life_stage_preds, on="image_path", how="left"
        )
    else:
        verified_metadata["life_stage_prediction"] = None

    return verified_metadata


def _remove_non_adults(current_filtered, verified_metadata):
    nan_lifestage = len(verified_metadata[verified_metadata.lifeStage.isna()])
    verified_metadata = verified_metadata[
        (verified_metadata.lifeStage.isin(["Adult", "Imago"]))
        | (verified_metadata.life_stage_prediction.isin(["Adult"]))
    ]
    previous = current_filtered
    current_filtered = len(verified_metadata)
    print(
        f"Non-adult removed: {previous - current_filtered} "
        f"(No lifeStage info: {nan_lifestage})",
    )
    return verified_metadata


def _remove_thumbnails(current_filtered, thumb_size, verified_metadata):
    verified_metadata = verified_metadata[~(verified_metadata.width < thumb_size)]
    verified_metadata = verified_metadata[~(verified_metadata.height < thumb_size)]
    previous = current_filtered
    current_filtered = len(verified_metadata)
    print("Thumbnail removed: ", previous - current_filtered)
    return current_filtered, verified_metadata


def _ignore_dataset(current_filtered, ignore_dataset_by_key, verified_metadata):
    dataset_keys = ignore_dataset_by_key.split(",")
    verified_metadata = verified_metadata[
        ~verified_metadata.datasetKey.isin(dataset_keys)
    ]
    previous = current_filtered
    current_filtered = len(verified_metadata)
    print("Removed by datasetKey: ", previous - current_filtered)
    return current_filtered, verified_metadata


def _remove_duplicates(current_filtered, verified_metadata):
    verified_metadata = verified_metadata.drop_duplicates(
        subset=["identifier"], keep=False
    )
    previous = current_filtered
    current_filtered = len(verified_metadata)
    print("Duplicate URL removed: ", previous - current_filtered)
    return current_filtered, verified_metadata


def clean_dataset(
    dwca_file: str,
    verified_data_csv: str,
    remove_duplicate_url: bool,
    ignore_dataset_by_key: str,
    remove_tumbnails: bool,
    thumb_size: int,
    remove_non_adults: bool,
    life_stage_predictions: str,
):
    verified_metadata = _load_data(dwca_file, verified_data_csv, life_stage_predictions)
    current_filtered = len(verified_metadata)
    print("Verified images: ", current_filtered)

    if remove_duplicate_url:
        current_filtered, verified_metadata = _remove_duplicates(
            current_filtered, verified_metadata
        )

    if ignore_dataset_by_key is not None:
        current_filtered, verified_metadata = _ignore_dataset(
            current_filtered, ignore_dataset_by_key, verified_metadata
        )

    if remove_tumbnails:
        current_filtered, verified_metadata = _remove_thumbnails(
            current_filtered, thumb_size, verified_metadata
        )

    if remove_non_adults:
        verified_metadata = _remove_non_adults(current_filtered, verified_metadata)

    verified_metadata_clean = verified_metadata.copy()
    metadata_clean_filename = verified_data_csv[:-4] + "_clean.csv"
    verified_metadata_clean.to_csv(metadata_clean_filename, index=False)
    print(f"Clean dataset saved to {metadata_clean_filename}")
