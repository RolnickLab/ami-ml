#!/usr/bin/env python
# coding: utf-8

import os

import pandas as pd
from dwca.read import DwCAReader


def get_image_path(image_data):
    image_path = str(image_data["datasetKey"]) + os.sep + str(image_data["coreid"])
    if image_data["count"] > 0:
        image_path = image_path + "_" + str(image_data["count"])

    return image_path + ".jpg"


def load_dwca_data(dwca_file: str):
    with DwCAReader(dwca_file) as dwca:
        media_df = dwca.pd_read("multimedia.txt", parse_dates=True, on_bad_lines="skip")
        occ_df = dwca.pd_read("occurrence.txt", parse_dates=True, on_bad_lines="skip")

    media_df = media_df[["coreid", "identifier"]].copy()
    occ_df = occ_df[
        [
            "id",
            "datasetKey",
            "speciesKey",
            "lifeStage",
            "decimalLatitude",
            "decimalLongitude",
            "eventDate",
        ]
    ].copy()

    images = pd.merge(media_df, occ_df, how="inner", left_on="coreid", right_on="id")
    images["count"] = images.groupby("coreid").cumcount()

    return images
