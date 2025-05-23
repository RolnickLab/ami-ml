#!/usr/bin/env python
# coding: utf-8

""" Conversion of ML structured data to the WebDataset format
"""

import json
import os
from pathlib import Path
from typing import Generator

import dotenv
import pandas as pd
import PIL
import webdataset as wds
from PIL import Image

dotenv.load_dotenv()


def _get_image(image_path: Path) -> Image.Image | None:
    """Read an image from the given path.
    Args:
        image_path (Path): Path to the image file.
    Returns:
        Image.Image: The loaded image.
    """

    image = None

    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
    except PIL.UnidentifiedImageError:
        print(f"Unidentified Image Error on file {image_path}", flush=True)
    except OSError:
        print(f"OSError Error on file {image_path}", flush=True)

    return image


def _create_samples(dataset_dir: str, category_map: dict, split_type: str) -> Generator:
    """Create samples for the webdataset.

    Args:
        dataset_dir (str): Directory containing the dataset.
        category_map (dict): Mapping of taxon keys to labels.
        split_type (str): Type of split (train, val, test).
    """

    # Read the split files
    dataset_df = pd.read_csv(Path(dataset_dir) / (split_type + ".csv"))

    for _, row in dataset_df.iterrows():
        taxon_key, filename = str(row["taxonkey"]), row["filename"]
        image = _get_image(Path(dataset_dir) / "ami_traps" / taxon_key / filename)
        label = category_map.get(taxon_key, None)
        if not label:
            print(f"Label not found for taxon key {taxon_key}", flush=True)
        if image and label:
            sample = {"__key__": Path(filename).stem, "jpg": image, "cls": label}
            yield sample


def _write_samples_to_sink(
    samples,
    webdataset_dir: Path,
    split_type: str,
    max_shard_size: int = 25 * 1024 * 1024,
) -> None:
    """Write samples to the sink.

    Args:
        samples (Generator): Generator object containing the files to be written.
        webdataset_dir (str): Directory to save the webdataset.
        split_type (str): Type of split (train, val, test).
        max_shard_size (int): Maximum size of each shard in bytes.

    Returns:
        None: The function saves the samples in the specified directory.
    """
    webdataset_pattern = str(Path(webdataset_dir) / f"{split_type}-%06d.tar")
    with wds.ShardWriter(webdataset_pattern, maxsize=max_shard_size) as sink:
        for sample in samples:
            sink.write(sample)


def convert_to_webdataset(fine_tuning_data_dir: str, category_map_f: str) -> None:
    """Main function to convert the fine-tuning camera trap data to webdataset format.

    Args:
        fine_tuning_data_dir (str): Directory containing the fine-tuning data.
        category_map_f (str): Path to the category map file.

    Returns:
        None: The function processes and saves the webdataset in the specified directory.
    """

    # Read the category map
    with open(category_map_f, "r", encoding="utf-8") as f:
        category_map = json.load(f)

    # Create samples for the webdataset
    train_samples = _create_samples(
        fine_tuning_data_dir,
        category_map,
        "train",
    )
    val_samples = _create_samples(fine_tuning_data_dir, category_map, "val")
    test_samples = _create_samples(fine_tuning_data_dir, category_map, "test")

    # Save the samples to the webdataset format
    _write_samples_to_sink(
        train_samples, Path(fine_tuning_data_dir) / "webdataset" / "train", "train"
    )
    _write_samples_to_sink(
        val_samples, Path(fine_tuning_data_dir) / "webdataset" / "val", "val"
    )
    _write_samples_to_sink(
        test_samples, Path(fine_tuning_data_dir) / "webdataset" / "test", "test"
    )


if __name__ == "__main__":
    FINE_TUNING_UK_DENMARK_DATA_DIR = os.getenv(
        "FINE_TUNING_UK_DENMARK_DATA_DIR", "./fine_tuning_data/"
    )
    WEUROPE_CATEGORY_MAP = os.getenv(
        "WEUROPE_CATEGORY_MAP", "./neamerica_category_map.csv"
    )
    convert_to_webdataset(FINE_TUNING_UK_DENMARK_DATA_DIR, WEUROPE_CATEGORY_MAP)
