#!/usr/bin/env python
# coding: utf-8

"""Temperature scaling for confidence calibration"""

import gc

import dotenv
import torch
import torch.nn as nn
import torch.optim as optim
import typer

from src.classification.dataloader import build_webdataset_pipeline
from src.classification.utils import build_model

dotenv.load_dotenv()


class TemperatureScaling(nn.Module):
    """Temperature scaling class"""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))  # Initialize T = 1

    def forward(self, logits):
        return logits / self.temperature


def expected_calibration_error():
    """Calculate expected calibration error"""
    pass


def calibration_error_on_gbif_test():
    """Calculate ECE on GBIF test set before and after temperature scaling"""
    pass


def calibration_error_on_ami_traps():
    """Calculate ECE on AMI-Traps dataset before and after temperature scaling"""
    pass


def reliability_diagram():
    """Plot reliability diagram"""


def tune_temperature(
    model: torch.nn.Module, val_dataloader: torch.utils.data.DataLoader, device: str
):
    """Main temperature tuning function"""

    model.eval()
    logits_list = []
    labels_list = []
    loop_num = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            loop_num += 1
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            logits_list.append(logits)
            labels_list.append(labels)

    # Concatenate all the logits and labels
    logits = torch.cat(logits_list).cpu()
    labels = torch.cat(labels_list).cpu()
    del logits_list
    del labels_list
    gc.collect()

    temp_scaling = TemperatureScaling()
    optimizer = optim.LBFGS([temp_scaling.temperature], lr=0.01, max_iter=50)
    print("create optimizer", flush=True)

    def closure():
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(temp_scaling(logits), labels)
        loss.backward(retain_graph=True)
        return loss

    optimizer.step(closure)

    print(f"Optimal Temperature: {temp_scaling.temperature.item()}")
    return temp_scaling


def main(
    model_weights: str = typer.Option(),
    model_type: str = typer.Option(),
    num_classes: int = typer.Option(),
    val_webdataset: str = typer.Option(),
    image_input_size: int = typer.Option(),
    batch_size: int = typer.Option(),
    preprocess_mode: str = typer.Option(),
):
    """Main function for confidence calibration"""

    # Model initialization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"The available device is {device}.")
    model = build_model(device, model_type, num_classes, model_weights)

    # Dataloader
    val_dataloader = build_webdataset_pipeline(
        val_webdataset,
        image_input_size,
        batch_size,
        preprocess_mode,
    )

    # Tune temperature parameter
    _ = tune_temperature(model, val_dataloader, device)

    # TODO Calculate ECE on test set before temperature scaling

    # TODO Plot reliability diagram on test set before temperature scaling

    # TODO Calculate ECE on test set after temperature scaling

    # TODO Plot reliability diagram on test set after temperature scaling


if __name__ == "__main__":
    # MODEL_WEIGHTS = os.getenv("QUEBEC_VERMONT_WEIGHTS")
    # CONF_CALIB_VAL_WBDS = os.getenv("CONF_CALIB_VAL_WBDS")

    # main(
    #     model_weights=MODEL_WEIGHTS,
    #     model_type="resnet50",
    #     num_classes=2497,
    #     val_webdataset=CONF_CALIB_VAL_WBDS,
    #     image_input_size=128,
    #     batch_size=64,
    #     preprocess_mode="torch",
    # )
    typer.run(main)
