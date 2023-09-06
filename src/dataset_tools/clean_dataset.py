#!/usr/bin/env python
# coding: utf-8

""" Clean dataset applying a set of rules to improve the quality of the dataset
"""
import click
import pandas as pd
from utils import get_image_path, load_dwca_data


def load_data(dwca_file: str, verified_data_csv: str):
    metadata = load_dwca_data(dwca_file)
    metadata["image_path"] = metadata.apply(get_image_path, axis=1)
    verified_data = pd.read_csv(verified_data_csv)
    verified_metadata = pd.merge(
        verified_data[["image_path", "width", "height", "fetch_date"]],
        metadata,
        how="inner",
        on="image_path",
    )

    return verified_metadata


@click.command(context_settings={"show_default": True})
@click.option(
    "--dwca_file",
    type=str,
    required=True,
    help="Darwin Core Archive file",
)
@click.option(
    "--verified_data_csv",
    type=str,
    required=True,
    help="CSV file containing verified image info",
)
@click.option(
    "--remove_duplicate_url",
    type=bool,
    default=True,
    help="Whether occurrences with duplicate URLs should be removed.",
)
@click.option(
    "--remove_tumbnails",
    type=bool,
    default=True,
    help=(
        "Whether small images should be removed. Use the option --thumb_size to "
        "determine the minimum side size."
    ),
)
@click.option(
    "--thumb_size",
    type=int,
    default=64,
    help="Minimum side size to an image not be considered as thumbnail",
)
@click.option(
    "--ignore_dataset_by_key",
    type=str,
    default=(
        "f3130a8a-4508-42b4-9737-fbda77748438,"
        "4bfac3ea-8763-4f4b-a71a-76a6f5f243d3,"
        "7e380070-f762-11e1-a439-00145eb45e9a"
    ),
    help=(
        "DatasetKeys separeted by comma. Some datasets might be ignored due to the "
        "poor quality of their images."
    ),
)
@click.option(
    "--remove_non_adults",
    type=bool,
    default=True,
    help="Whether keeping only occurrences with lifeStage identified as Adult or Imago",
)
def main(
    dwca_file: str,
    verified_data_csv: str,
    remove_duplicate_url: bool,
    ignore_dataset_by_key: str,
    remove_tumbnails: bool,
    thumb_size: int,
    remove_non_adults: bool,
):
    verified_metadata = load_data(dwca_file, verified_data_csv)
    current_filtered = len(verified_metadata)
    print("Verified images: ", current_filtered)

    if remove_duplicate_url:
        verified_metadata = verified_metadata.drop_duplicates(
            subset=["identifier"], keep=False
        )
        previous = current_filtered
        current_filtered = len(verified_metadata)
        print("Duplicate URL removed: ", previous - current_filtered)

    if ignore_dataset_by_key is not None:
        dataset_keys = ignore_dataset_by_key.split(",")
        verified_metadata = verified_metadata[
            ~verified_metadata.datasetKey.isin(dataset_keys)
        ]
        previous = current_filtered
        current_filtered = len(verified_metadata)
        print("Removed by datasetKey: ", previous - current_filtered)

    if remove_tumbnails:
        verified_metadata = verified_metadata[~(verified_metadata.width < thumb_size)]
        verified_metadata = verified_metadata[~(verified_metadata.height < thumb_size)]
        previous = current_filtered
        current_filtered = len(verified_metadata)
        print("Thumbnail removed: ", previous - current_filtered)

    if remove_non_adults:
        nan_lifestage = len(verified_metadata[verified_metadata.lifeStage.isna()])
        verified_metadata = verified_metadata[
            verified_metadata.lifeStage.isin(["Adult", "Imago"])
        ]
        previous = current_filtered
        current_filtered = len(verified_metadata)
        print(
            f"Non-adult removed: {previous - current_filtered} "
            f"(No lifeStage info: {nan_lifestage})",
        )

    verified_metadata_clean = verified_metadata.copy()
    metadata_clean_filename = verified_data_csv[:-4] + "_clean.csv"
    verified_metadata_clean.to_csv(metadata_clean_filename, index=False)
    print(f"Clean dataset saved to {metadata_clean_filename}")


if __name__ == "__main__":
    main()
