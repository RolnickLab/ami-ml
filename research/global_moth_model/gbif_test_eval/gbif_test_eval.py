#!/usr/bin/env python
# coding: utf-8

"""Get species-wise accuracy for GBIF test set"""

import json
import os
from pathlib import Path

import torch
from dataloader import build_webdataset_pipeline
from dotenv import load_dotenv
from model_inference import ModelInference

SP_ACC = {}  # "key" : [correct, total]


def cal_species_accuracy(outputs: list[str], labels: list[str]):
    """Calculate per species accuracy"""

    for key_l, key_o in zip(labels, outputs):
        if key_l == key_o:
            if key_l in SP_ACC.keys():
                SP_ACC[key_l][0] += 1
                SP_ACC[key_l][1] += 1
            else:
                SP_ACC[key_l] = [1, 1]
        else:
            if key_l in SP_ACC.keys():
                SP_ACC[key_l][1] += 1
            else:
                SP_ACC[key_l] = [0, 1]


def evaluate(wbd_url: str, model_f: str, category_map_f: str, batch_size: int = 128):
    """Main evaluation function"""

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model_class = ModelInference(
        model_f, "timm_resnet50", category_map_f, device, topk=1
    )
    model = model_class.model

    # Load test data loader
    test_dataloader = build_webdataset_pipeline(
        sharedurl=wbd_url,
        input_size=128,
        batch_size=batch_size,
        is_training=False,
        preprocess_mode="torch",
    )
    processed_imgs = 0

    with torch.no_grad():
        model.eval()
        for images, labels in test_dataloader:
            # Get the top 1 prediction
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            processed_imgs += len(labels)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            _, ids = torch.topk(outputs, 1)
            ids = torch.squeeze(ids).cpu().tolist()
            labels = labels.cpu().tolist()

            # Get categories from indices
            outputs = model_class.get_category(ids)
            labels = model_class.get_category(labels)

            # Calculate per species accuracy
            cal_species_accuracy(outputs, labels)

            if processed_imgs % (batch_size * 10) == 0:
                print(f"{processed_imgs} images are processed.", flush=True)

        print("Finished model evaluation.", flush=True)


if __name__ == "__main__":
    load_dotenv()
    TEST_WBDS_URL = os.getenv("TEST_WBDS_URL")
    GLOBAL_MODEL = os.getenv("GLOBAL_MODEL")
    CATEGORY_MAP = os.getenv("CATEGORY_MAP_JSON")
    GLOBAL_MODEL_DIR = os.getenv("GLOBAL_MODEL_DIR")
    batch_size = 1024
    evaluate(TEST_WBDS_URL, GLOBAL_MODEL, CATEGORY_MAP, batch_size)

    gbif_test_acc = Path(GLOBAL_MODEL_DIR) / "gbif_test_accuracy.json"
    with open(gbif_test_acc, "w") as file:
        json.dump(SP_ACC, file, indent=2)
