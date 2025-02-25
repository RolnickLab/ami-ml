#!/usr/bin/env python
# coding: utf-8

"""Temperature scaling for confidence calibration"""

import gc

import dotenv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import typer

from src.classification.dataloader import build_webdataset_pipeline
from src.classification.utils import build_model

# import os
dotenv.load_dotenv()


class TemperatureScaling(nn.Module):
    """Temperature scaling class"""

    def __init__(self, initial_t: float = 1.0):
        super().__init__()
        print(f"The initial value of temp scaling is {initial_t}.")
        self.temperature = nn.Parameter(torch.ones(1) * initial_t)  # Initialize T = 1

    def forward(self, logits):
        return logits / self.temperature


def expected_calibration_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
    plot_reliability_diagram: bool = False,
    reliability_diagram_title: str = "Reliability Diagram",
):
    """
    Computes the Expected Calibration Error (ECE).

    Args:
        logits (torch.Tensor): Model's raw outputs before softmax (shape: [N, num_classes]).
        labels (torch.Tensor): True class labels (shape: [N]).
        n_bins (int): Number of bins for confidence calibration.

    Returns:
        float: Expected Calibration Error (ECE).
    """

    softmax_probs = torch.softmax(logits, dim=1)
    confidences, predictions = torch.max(
        softmax_probs, dim=1
    )  # Get max confidence per sample
    accuracies = predictions.eq(labels)  # 1 if correct, 0 otherwise

    # Create bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)  # Equal-width bins
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.tensor(0.0, device=logits.device)

    # Plotting related variables
    avg_confidences = []
    avg_accuracies = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()  # Fraction of samples in the bin

        if prop_in_bin > 0:
            acc_in_bin = accuracies[in_bin].float().mean()  # Accuracy in bin
            avg_conf_in_bin = confidences[in_bin].mean()  # Avg confidence in bin
            ece += prop_in_bin * torch.abs(avg_conf_in_bin - acc_in_bin)  # Weighted sum

            # Update plotting variables
            avg_confidences.append(avg_conf_in_bin.item())
            avg_accuracies.append(acc_in_bin.item())
            bin_counts.append(prop_in_bin.item())
        else:
            avg_confidences.append(0)
            avg_accuracies.append(0)
            bin_counts.append(0)

    if plot_reliability_diagram:
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")  # Diagonal line
        plt.bar(
            avg_confidences,
            avg_accuracies,
            width=0.07,
            color="blue",
            alpha=0.7,
            label="Model Reliability",
        )

        # Formatting
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title(reliability_diagram_title)
        plt.legend(loc="upper left")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True)
        plt.savefig(reliability_diagram_title + ".png")

    return ece.item()  # Return scalar ECE value


def calibration_error_on_gbif_test(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: str,
    optimal_t: float,
    plot_reliability_diagram: bool = True,
) -> tuple[float, float]:
    """Calculate ECE on GBIF test set before and after calibration"""

    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in test_dataloader:
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

    # Calculate ECE before and after calibration
    ece_before_calibration = expected_calibration_error(
        logits,
        labels,
        plot_reliability_diagram=True,
        reliability_diagram_title="before_calibration",
    )
    ece_after_calibration = expected_calibration_error(
        logits / optimal_t,
        labels,
        plot_reliability_diagram=True,
        reliability_diagram_title="after_calibration",
    )

    return ece_before_calibration, ece_after_calibration


def calibration_error_on_ami_traps():
    """Calculate ECE on AMI-Traps dataset before and after temperature scaling"""
    pass


def reliability_diagram():
    """Plot reliability diagram"""


def tune_temperature(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    device: str,
    initial_t: float = 1.5,
    max_optim_iterations=50,
    learning_rate=0.01,
) -> float:
    """Main temperature tuning function"""

    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            logits_list.append(logits)
            labels_list.append(labels)

    # Concatenate all the logits and labels
    logits = torch.cat(logits_list).to(device)
    labels = torch.cat(labels_list).to(device)
    del logits_list
    del labels_list
    gc.collect()

    temp_scaling = TemperatureScaling(initial_t=initial_t).to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.LBFGS(
        [temp_scaling.temperature], lr=learning_rate, max_iter=max_optim_iterations
    )

    def closure():
        optimizer.zero_grad()
        loss = loss_function(temp_scaling(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    optimal_temperature = temp_scaling.temperature.item()
    print(f"Optimal Temperature: {optimal_temperature}")

    # Calculate ECE before and after calibration
    val_ece_before_calibration = expected_calibration_error(logits, labels)
    val_ece_after_calibration = expected_calibration_error(
        logits / optimal_temperature, labels
    )
    print(
        f"Validation ECE before and after calibration is {val_ece_before_calibration} and {val_ece_after_calibration} respectively.",
        flush=True,
    )

    return optimal_temperature


def main(
    model_weights: str = typer.Option(),
    model_type: str = typer.Option(),
    num_classes: int = typer.Option(),
    val_webdataset: str = typer.Option(),
    test_webdataset: str = typer.Option(),
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
    test_dataloader = build_webdataset_pipeline(
        test_webdataset,
        image_input_size,
        batch_size,
        preprocess_mode,
    )

    # Tune temperature parameter
    optimal_temperature = tune_temperature(model, val_dataloader, device, initial_t=1.0)
    # optimal_temperature = 0.92076

    # Calculate ECE on test set before and after temperature scaling
    error_before_calibration, error_after_calibration = calibration_error_on_gbif_test(
        model, test_dataloader, device, optimal_temperature
    )
    print(
        f"GBIF Test ECE before and after calibration is {error_before_calibration} and {error_after_calibration} respectively.",
        flush=True,
    )


if __name__ == "__main__":
    # MODEL_WEIGHTS = os.getenv("QUEBEC_VERMONT_WEIGHTS")
    # CONF_CALIB_VAL_WBDS = os.getenv("TEST_CONF_CALIB_VAL_WBDS")
    # CONF_CALIB_TEST_WBDS = os.getenv("TEST_CONF_CALIB_TEST_WBDS")

    # main(
    #     model_weights=MODEL_WEIGHTS,
    #     model_type="resnet50",
    #     num_classes=2497,
    #     val_webdataset=CONF_CALIB_VAL_WBDS,
    #     test_webdataset=CONF_CALIB_TEST_WBDS,
    #     image_input_size=128,
    #     batch_size=64,
    #     preprocess_mode="torch",
    # )
    typer.run(main)
