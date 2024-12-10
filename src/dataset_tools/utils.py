#!/usr/bin/env python
# coding: utf-8

import os
import random

import numpy as np
import pandas as pd
from dwca.read import DwCAReader


def get_image_path(image_data):
    image_path = str(image_data["datasetKey"]) + os.sep + str(image_data["coreid"])
    if image_data["count"] > 0:
        image_path = image_path + "_" + str(image_data["count"])

    return image_path + ".jpg"


def load_dwca_data(dwca_file: str):
    with DwCAReader(dwca_file) as dwca:
        media_df = dwca.pd_read(
            "multimedia.txt", parse_dates=True, on_bad_lines="skip", low_memory=False
        )
        occ_df = dwca.pd_read(
            "occurrence.txt", parse_dates=True, on_bad_lines="skip", low_memory=False
        )

    media_df = media_df[["coreid", "identifier"]].copy()
    occ_df = occ_df[
        [
            "id",
            "datasetKey",
            "speciesKey",
            "acceptedTaxonKey",
            "lifeStage",
            "decimalLatitude",
            "decimalLongitude",
            "eventDate",
        ]
    ].copy()

    images = pd.merge(media_df, occ_df, how="inner", left_on="coreid", right_on="id")
    images["count"] = images.groupby("coreid").cumcount()

    return images


def set_random_seeds(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)


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
