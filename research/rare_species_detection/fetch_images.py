#!/usr/bin/env python
# coding: utf-8

"""
Fetch raw images using the AMI Data Platform (ADP) API
"""

import pathlib
from pathlib import Path

import requests


class FetchImages:
    """Fetch images from the AMI Data Platform"""

    def __init__(self, url: str, output_dir: str):
        self.url = url
        self.output_dir = output_dir

    def download_all_images(self):
        """Fetch all images and store to disk"""
        pass

    def fetch_single_image(self):
        """Fetch only a single image"""
        pass
