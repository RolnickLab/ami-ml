"""
Command Line Interface for the training module

To add a new command, create a new function below following these instructions:

- Create a new Command key constant
   <NEW_COMMAND_NAME>_CMD = "<NEW_COMMAND_NAME>_cmd"

- Add the new Command key constant to the COMMAND_KEYS frozenset

- Add the command name and help text in the COMMANDS and COMMANDS_HELP dictionaries

- Create a new function named <COMMAND_NAME>_command(), alongside the appropriate
  options
    - Make sure to reuse appropriate options before creating duplicate ones.
    - The contents of the CLI command should be minimal : execution of an imported
      function

- If unsure, take some time to look at how other commands have been implemented

- Make sure to use lazy loading when importing modules that are only used by 1 command
"""

import typing as tp
from typing import Optional

import click

from src.classification.constants import (
    ADAMW,
    AVAILABLE_LOSS_FUNCIONS,
    AVAILABLE_LR_SCHEDULERS,
    AVAILABLE_MODELS,
    AVAILABLE_OPTIMIZERS,
    CROSS_ENTROPY_LOSS,
)

SupportedModels = tp.Literal[*AVAILABLE_MODELS]
SupportedLossFunctions = tp.Literal[*AVAILABLE_LOSS_FUNCIONS]
SupportedOptimizers = tp.Literal[*AVAILABLE_OPTIMIZERS]
SupportedLearningRateSchedulers = tp.Literal[*AVAILABLE_LR_SCHEDULERS]

# Command key constants
# Make sure to add them to COMMAND_KEYS frozenset
TRAIN_CMD = "train_cmd"

# This is most useful to automatically test the CLI
COMMAND_KEYS = frozenset([TRAIN_CMD])

# Command dictionary
COMMANDS = {
    TRAIN_CMD: "train-model",
}

# Command help text dictionary
COMMANDS_HELP = {TRAIN_CMD: "Train a classification model"}


# # # # # # #
# Commands  #
# # # # # # #

# The order of declaration of the commands affect the order
# in which they appear in the CLI


#
# Train Model Command
#
@click.command(
    name=COMMANDS[TRAIN_CMD],
    help=COMMANDS_HELP[TRAIN_CMD],
    context_settings={"show_default": True},
)
@click.option(
    "--random_seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
@click.option(
    "--model_type",
    type=click.Choice(tp.get_args(SupportedModels)),
    required=True,
    help="Model architecture",
)
@click.option(
    "--num_classes",
    type=int,
    required=True,
    help="Number of model's output classes",
)
@click.option(
    "--existing_weights",
    type=str,
    default=None,
    help="Existing weights to be loaded, if available",
)
@click.option(
    "--total_epochs",
    type=int,
    default=30,
    help="Total number of training epochs",
)
@click.option(
    "--warmup_epochs",
    type=int,
    default=0,
    help="Number of warmup epochs, if using a learning rate scehduler",
)
@click.option(
    "--train_webdataset",
    type=str,
    required=True,
    help="Webdataset files for the training set",
)
@click.option(
    "--val_webdataset",
    type=str,
    required=True,
    help="Webdataset files for the validation set",
)
@click.option(
    "--test_webdataset",
    type=str,
    required=True,
    help="Webdataset files for the test set",
)
@click.option(
    "--image_input_size",
    type=int,
    default=128,
    help="Image input size for training and inference",
)
@click.option(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size for training",
)
@click.option(
    "--preprocess_mode",
    type=click.Choice(["torch", "tf", "other"]),
    default="torch",
    help="Preprocessing mode for normalization",
)
@click.option(
    "--optimizer_type",
    type=click.Choice(tp.get_args(SupportedOptimizers)),
    default=ADAMW,
    help="Optimizer type",
)
@click.option(
    "--learning_rate",
    type=float,
    default=0.001,
    help="Initial learning rate",
)
@click.option(
    "--learning_rate_scheduler_type",
    type=click.Choice(tp.get_args(SupportedLearningRateSchedulers)),
    default=None,
    help="Learning rate scheduler",
)
@click.option(
    "--weight_decay",
    type=float,
    default=1e-5,
    help="Weight decay for regularization",
)
@click.option(
    "--loss_function_type",
    type=click.Choice(tp.get_args(SupportedLossFunctions)),
    default=CROSS_ENTROPY_LOSS,
    help="Loss function",
)
@click.option(
    "--label_smoothing",
    type=float,
    default=0.1,
    help="Label smoothing for model regularization. No smoothing if 0.0",
)
def train_model_command(
    random_seed: int,
    model_type: str,
    num_classes: int,
    existing_weights: Optional[str],
    total_epochs: int,
    warmup_epochs: int,
    train_webdataset: str,
    val_webdataset: str,
    test_webdataset: str,
    image_input_size: int,
    batch_size: int,
    preprocess_mode: str,
    optimizer_type: str,
    learning_rate: float,
    learning_rate_scheduler_type: str,
    weight_decay: float,
    loss_function_type: str,
    label_smoothing: float,
):
    from src.classification.train import train_model

    train_model(
        random_seed=random_seed,
        model_type=model_type,
        num_classes=num_classes,
        existing_weights=existing_weights,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        train_webdataset=train_webdataset,
        val_webdataset=val_webdataset,
        test_webdataset=test_webdataset,
        image_input_size=image_input_size,
        batch_size=batch_size,
        preprocess_mode=preprocess_mode,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        learning_rate_scheduler_type=learning_rate_scheduler_type,
        weight_decay=weight_decay,
        loss_function_type=loss_function_type,
        label_smoothing=label_smoothing,
    )


# # # # # # # # # # # # # #
# Main CLI configuration  #
# # # # # # # # # # # # # #
class OrderCommands(click.Group):
    """This class is necessary to order the commands the way we want to."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(self.commands)


@click.group(cls=OrderCommands)
def cli():
    """This is the main command line interface for the classification tools."""


# Following is an automated way to add all functions containing the word `command`
# in their name instead of manually having to add them.
all_objects = globals()
functions = [
    obj for name, obj in all_objects.items() if callable(obj) and "command" in name
]

for command_function in functions:
    cli.add_command(command_function)

if __name__ == "__main__":
    cli()
