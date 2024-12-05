#!/usr/bin/env python
# coding: utf-8

"""Find the GBIF taxon keys for the taxonomic groups"""

import json
import os

import dotenv
from pygbif import species as species_api

dotenv.load_dotenv()

ORDER_CLASSIFIER_ACCEPTED_KEYS = os.environ.get("ORDER_CLASSIFIER_ACCEPTED_KEYS")

taxon_names = [
    "Diptera",
    "Hemiptera",
    "Odonata",
    "Coleoptera",
    "Araneae",
    "Orthoptera",
    "Ichneumonoidea",
    "Formicidae",
    "Apoidea",
    "Trichoptera",
    "Neuroptera",
    "Opiliones",
    "Ephemeroptera",
    "Plecoptera",
    "Blattodea",
    "Dermaptera",
    "Mantodea",
]
taxon_keys = []
# Data for Lepidoptera exists, hence excluded in this download

# Match for the taxon keys
for taxon in taxon_names:
    taxon_information = species_api.name_backbone(
        name=taxon, strict=True, clazz="Insecta"
    )
    taxon_key = ...
    taxon_keys.append(taxon_key)

# Save the list on disk
filename = ORDER_CLASSIFIER_ACCEPTED_KEYS
with open(filename, "w") as json_file:
    json.dump(filename, taxon_keys, indent=2)
