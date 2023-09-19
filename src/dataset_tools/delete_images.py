#!/usr/bin/env python
# coding: utf-8

""" Delete images from a list
"""

import os

import click
import pandas as pd


@click.command(context_settings={"show_default": True})
@click.option(
    "--error-images-csv",
    type=str,
    required=True,
    help="File to save image verification info",
)
@click.option(
    "--base-path",
    type=str,
    help="Root path for the image list",
)
def main(error_images_csv: str, base_path: str):
    errors_df = pd.read_csv(error_images_csv, header=0, names=["filename"])
    if base_path is not None:
        errors_df["filename"] = errors_df["filename"].apply(
            lambda x: os.path.join(base_path, x)
        )
    for _, row in errors_df.iterrows():
        if os.path.isfile(row["filename"]):
            os.remove(row["filename"])


if __name__ == "__main__":
    main()
