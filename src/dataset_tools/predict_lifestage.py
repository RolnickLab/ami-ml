#!/usr/bin/env python
# coding: utf-8

""" Predict life stage for moths
"""

import json
import os
from typing import Union

import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import wandb


def _build_model(model_name, num_classes, model_path, device):
    if model_name == "efficientnetv2-b3":
        model = timm.create_model(
            "tf_efficientnetv2_b3", pretrained=True, num_classes=num_classes
        )
    else:
        raise RuntimeError(f"Model {model_name} not implemented")

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    return model


def _get_image_transforms(input_size=224, preprocess_mode="torch"):
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
        self.transform = _get_image_transforms(input_size, preprocess_mode)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_id = self.img_labels["image_path"].iloc[idx]
        image_path = os.path.join(self.dataset_dir, image_id)
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = self.transform(image)

        return image, image_id


def _save_predictions(
    model,
    dataset,
    device,
    log_frequence,
    category_map: dict,
    results_csv: str,
    batch_size: int,
    wandb_log: bool = True,
) -> None:
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        ids_cpu = []

        model.eval()
        for i, data in enumerate(dataset):
            images, ids = data
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            y_pred = torch.cat((y_pred, outputs), 0)
            ids_cpu += list(ids)

            # Log to W & B
            if wandb_log:
                wandb.log({"images_processed": (i + 1) * batch_size})

            if (i % log_frequence == 0) or (i == len(dataset) - 1):
                print(f"Finished eval step {i}", flush=True)
                y_pred_cpu = y_pred.cpu().argmax(axis=1)

                # Calculate the predictions to be written to disk
                df = pd.DataFrame(
                    {"image_path": ids_cpu, "life_stage_prediction": y_pred_cpu}
                )
                df["life_stage_prediction"] = df["life_stage_prediction"].astype(int)
                df["life_stage_prediction"] = df["life_stage_prediction"].map(
                    category_map
                )

                # Save the file to disk if there is none
                if not os.path.isfile(results_csv):
                    df.to_csv(results_csv, index=False)
                # Read and concatenate the data if the file already exists
                else:
                    df_exist = pd.read_csv(results_csv)
                    df_merged = pd.concat([df_exist, df], ignore_index=True)
                    df_merged.to_csv(results_csv, index=False)

                # Re-initialize temporary variables
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                ids_cpu = []


def predict_lifestage(
    verified_data_csv: str,
    dataset_path: str,
    input_size: int,
    preprocessing_mode: str,
    predict_nan_life_stage: bool,
    batch_size: int,
    model_name: str,
    num_classes: int,
    model_path: str,
    log_frequence: int,
    category_map_json: str,
    results_csv: str,
    wandb_entity: Union[str, None],
    wandb_project: Union[str, None],
    wandb_run: Union[str, None],
):
    # Weights and Biases logging
    if wandb_entity:
        wandb.init(entity=wandb_entity, project=wandb_project, name=wandb_run)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    test_data = CSVDataset(
        verified_data_csv,
        dataset_path,
        input_size,
        preprocessing_mode,
        predict_nan_life_stage,
    )
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model = _build_model(model_name, num_classes, model_path, device)
    with open(category_map_json, "r") as f:
        category_map = json.load(f)
    category_map = {v: k for k, v in category_map.items()}

    _save_predictions(
        model,
        test_dataloader,
        device,
        log_frequence,
        category_map,
        results_csv,
        batch_size,
    )

    print("Finished life stage prediction.", flush=True)
