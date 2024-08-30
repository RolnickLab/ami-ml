#!/usr/bin/env python
# coding: utf-8

"""Get species-wise accuracy for GBIF test set"""

import os

import torch
from dataloader import build_webdataset_pipeline
from dotenv import load_dotenv
from model_inference import ModelInference


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

    with torch.no_grad():
        model.eval()
        for images, labels in test_dataloader:
            # Get the top 1 prediction
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            _, ids = torch.topk(outputs, 1)
            ids = torch.squeeze(ids).cpu().tolist()
            labels = labels.cpu().tolist()

            # Get categories from indices
            outputs = model_class.get_category(ids)
            labels = model_class.get_category(labels)


if __name__ == "__main__":
    load_dotenv()
    TEST_WBDS_URL = os.getenv("TEST_WBDS_URL")
    GLOBAL_MODEL = os.getenv("GLOBAL_MODEL")
    CATEGORY_MAP = os.getenv("CATEGORY_MAP_JSON")
    batch_size = 16
    evaluate(TEST_WBDS_URL, GLOBAL_MODEL, CATEGORY_MAP, batch_size)
