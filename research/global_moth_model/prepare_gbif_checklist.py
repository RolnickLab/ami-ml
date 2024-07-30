#!/usr/bin/env python
# coding: utf-8

"""Prepare the GBIF checklist for the global moth model"""

import os
from pathlib import Path

# System packages
import pandas as pd

# 3rd party packages
from dotenv import load_dotenv

# Load secrets and config from optional .env file
load_dotenv()


def remove_non_species_taxon(checklist: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all non-species taxa from the checklist
    """

    # Keep only rows where the taxa rank is "SPECIES"
    checklist = checklist.loc[checklist["taxonRank"] == "SPECIES"]

    return checklist


def remove_butterflies(checklist: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all butterflies from the checklist
    """

    # List of butterfly families
    butterfly_fm = [
        "Hesperiidae",
        "Lycaenidae",
        "Nymphalidae",
        "Papilionidae",
        "Pieridae",
        "Riodinidae",
        "Hedylidae",
    ]

    # Remove butterfly families
    checklist = checklist.loc[~checklist["family"].isin(butterfly_fm)]

    return checklist


if __name__ == "__main__":
    GLOBAL_MODEL_DIR = os.getenv("GLOBAL_MODEL_DIR")

    # Remove non-species taxa
    checklist = "gbif_leps_checklist_07242024_original.csv"
    checklist_pd = pd.read_csv(Path(GLOBAL_MODEL_DIR) / checklist)
    leps_checklist_pd = remove_non_species_taxon(checklist_pd)
    leps_checklist_pd.to_csv(
        Path(GLOBAL_MODEL_DIR) / "gbif_leps_checklist_07242024.csv", index=False
    )

    # Remove butterflies
    checklist = "gbif_leps_checklist_07242024.csv"
    checklist_pd = pd.read_csv(Path(GLOBAL_MODEL_DIR) / checklist)
    moth_checklist_pd = remove_butterflies(checklist_pd)
    moth_checklist_pd.to_csv(
        Path(GLOBAL_MODEL_DIR) / "gbif_moth_checklist_07242024.csv", index=False
    )
