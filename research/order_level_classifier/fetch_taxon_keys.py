#!/usr/bin/env python
# coding: utf-8

"""Find the GBIF taxon keys for the taxonomic groups"""

import json
import os

import dotenv
from pygbif import species as species_api

dotenv.load_dotenv()

ORDER_CLASSIFIER_ACCEPTED_KEYS_PART2 = os.getenv(
    "ORDER_CLASSIFIER_ACCEPTED_KEYS_PART2",
)

taxon_names = ["Mantodea"]

taxon_keys = []
# Data for Lepidoptera exists, hence excluded in this download

# Match for the taxon keys
for taxon in taxon_names:
    taxon_information = species_api.name_backbone(
        name=taxon, strict=True, clazz="Insecta"
    )
    try:
        taxon_key = str(taxon_information["usageKey"])
    except TypeError:
        taxon_key = None

    print(f"Taxon key for {taxon} is {taxon_key}")
    taxon_keys.append(taxon_key)

# Save the list on disk
filename = ORDER_CLASSIFIER_ACCEPTED_KEYS_PART2
with open(filename, "w") as json_file:
    json.dump(taxon_keys, json_file, indent=2)
