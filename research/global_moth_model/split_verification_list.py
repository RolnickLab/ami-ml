#!/usr/bin/env python
# coding: utf-8

"""Split the image verification list to multiple parts"""

import os
from pathlib import Path

# System packages
import pandas as pd

# 3rd party packages
from dotenv import load_dotenv

# Load secrets and config from optional .env file
load_dotenv()

# Load the list
img_verf_df = pd.read_csv(os.getenv("VERIFICATION_RESULTS"))
img_verf_lstage_nan_df = img_verf_df[img_verf_df.lifeStage.isnull()].copy()

# Slice the list
num_entries = img_verf_lstage_nan_df.shape[0]
half = int(num_entries / 2)
img_verf_lstage_nan_p1 = img_verf_lstage_nan_df.iloc[:half, :].copy()
img_verf_lstage_nan_p2 = img_verf_lstage_nan_df.iloc[half:, :].copy()

# Save the scripts
save_dir = os.getenv("GLOBAL_MODEL_DIR")
fname = Path(os.getenv("VERIFICATION_RESULTS")).stem
img_verf_lstage_nan_p1.to_csv(Path(save_dir) / str(fname + "_p1" + ".csv"), index=False)
img_verf_lstage_nan_p2.to_csv(Path(save_dir) / str(fname + "_p2" + ".csv"), index=False)
