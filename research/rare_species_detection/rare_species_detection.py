# !/usr/bin/env python
# coding: utf-8

"""
This is a module docstring
"""

import typer

from ... import Classification, Localization
from .fetch_images import FetchImages


def main(api_url: str, output_dir: str):
    """Main function"""

    # Download camera trap image through the API
    fetch_images = FetchImages(api_url, output_dir)
    fetch_images.fetch_single_image()

    # Import the localization model
    localization = Localization()

    # Import the classification model
    classification = Classification()


if __name__ == "__main__":
    main()
