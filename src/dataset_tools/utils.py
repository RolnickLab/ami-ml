#!/usr/bin/env python
# coding: utf-8

import os


def get_image_path(image_data):
    image_path = str(image_data["datasetKey"]) + os.sep + str(image_data["coreid"])
    if image_data["count"] > 0:
        image_path = image_path + "_" + str(image_data["count"])

    return image_path + ".jpg"
