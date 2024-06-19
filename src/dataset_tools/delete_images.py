#!/usr/bin/env python
# coding: utf-8

""" Delete images from a list
"""

import os

import pandas as pd


def delete_images(error_images_csv: str, base_path: str):
    errors_df = pd.read_csv(error_images_csv, header=0, names=["filename"])
    if base_path is not None:
        errors_df["filename"] = errors_df["filename"].apply(
            lambda x: os.path.join(base_path, x)
        )
    for _, row in errors_df.iterrows():
        if os.path.isfile(row["filename"]):
            os.remove(row["filename"])
