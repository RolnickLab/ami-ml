#!/usr/bin/env python
# coding: utf-8

""" Predict life stage for moths
"""

import os

import click
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def get_image_transforms(input_size=224, preprocess_mode="torch"):
    if preprocess_mode == "torch":
        # imagenet preprocessing
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif preprocess_mode == "tf":
        # global butterfly preprocessing (-1 to 1)
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    else:
        # (0 to 1)
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class CSVDataset(Dataset):
    def __init__(
        self,
        data_csv,
        dataset_dir,
        input_size,
        preprocess_mode,
        keep_only_nan_life_stage=True,
    ):
        self.img_labels = pd.read_csv(data_csv)
        self.dataset_dir = dataset_dir
        if keep_only_nan_life_stage:
            self.img_labels = self.img_labels[self.img_labels.lifeStage.isnull()].copy()
        self.transform = get_image_transforms(input_size, preprocess_mode)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_path = os.path.join(
            self.dataset_dir, self.img_labels["image_path"].iloc[idx]
        )
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = self.transform(image)

        return image


@click.command(context_settings={"show_default": True})
@click.option(
    "--verified-data-csv",
    type=str,
    required=True,
    help="CSV file containing verified image info",
)
@click.option(
    "--dataset-path",
    type=str,
    required=True,
    help="Path to directory containing dataset images.",
)
@click.option("--input-size", type=int, default=300, help="Input size of the model")
@click.option(
    "--preprocessing-mode",
    type=click.Choice(["tf", "torch", "float32"]),
    default="tf",
    help=(
        "Mode for scaling input: tf scales image between -1 and 1;"
        " torch normalizes inputs using ImageNet mean and std"
        " float32 uses image on scale 0-1"
    ),
)
@click.option(
    "--predict-nan-life-stage",
    type=bool,
    default=True,
    help=(
        "You can use this parameter to specify whether you want to make predictions"
        " only for images with the 'life_stage' tag as NaN"
    ),
)
@click.option(
    "--batch-size", type=int, default=32, help="Batch size used during training."
)
def main(
    verified_data_csv: str,
    dataset_path: str,
    input_size: int,
    preprocessing_mode: str,
    predict_nan_life_stage: bool,
    batch_size: int,
):
    test_data = CSVDataset(
        verified_data_csv,
        dataset_path,
        input_size,
        preprocessing_mode,
        predict_nan_life_stage,
    )
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for i, data in enumerate(test_dataloader):
        print(data)
        print(data.shape)
        if i > 0:
            break


if __name__ == "__main__":
    main()
