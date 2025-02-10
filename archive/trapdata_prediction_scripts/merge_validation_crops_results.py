#! /usr/bin/env python3

import os
import sys
import datetime
import pathlib

import pandas as pd


AMIDATA = pathlib.Path(os.environ["AMIDATA"])
validation_crops = AMIDATA / "validation_crops"

models = [
    dict(slug="aug22", path="quebec-vermont-moth-model_v02_efficientnetv2-b3_2022-09-08-15-44.pt-.json"),
    dict(slug="mixedres", path="vermont-moth-model_v07_resnet50_2022-12-22-07-54.pt-.json"),
]

df1 = pd.read_json(f"{validation_crops}/classify_annotation-Quebec-quebec-vermont-moth-model_v02_efficientnetv2-b3_2022-09-08-15-44.pt-.json").T
df2 = pd.read_json(f"{validation_crops}/classify_annotation-Quebec-quebec-vermont-moth-model_v07_resnet50_2022-12-22-07-54.pt-.json").T

def get_name(s):
    return s.split("-")[-1].rsplit(".", 1)[0]

def get_rank(s):
    first_part = s.split("-")[0]
    return first_part.split("_")[-1]

df = df1.join(df2, lsuffix="_aug22", rsuffix="_mixedres")
df.index.rename("filename", inplace=True)
df = df.reset_index()

df.loc[:, "true_label"] = df["filename"].apply(get_name)
df.loc[:, "true_label_rank"] = df["filename"].apply(get_rank)


df["correct_aug22"] = False
df["correct_mixedres"]  = False

total_species_ids = len(df[(df.true_label_rank == "species")])
print("Num species IDs:", total_species_ids)

df.loc[
    (df.true_label_rank == "species") & (df.true_label == df.label_aug22), 
    "correct_aug22"
    ] = True
total_aug22 = sum(df.correct_aug22)
print("aug22", total_aug22, total_species_ids, total_aug22/total_species_ids)

df.loc[
    (df.true_label_rank == "species") & (df.true_label == df.label_mixedres), 
    "correct_mixedres"
    ] = True
total_mixedres = sum(df.correct_mixedres)
print("mixedres", total_mixedres, total_species_ids, total_mixedres/total_species_ids)

date = datetime.date.today().strftime("%Y%m%d")
df.to_csv(f"{validation_crops}/moth_classification_results_comparison-{date}.csv", index=False)

# import ipdb; ipdb.set_trace()
