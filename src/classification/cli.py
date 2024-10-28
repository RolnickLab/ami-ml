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

import click

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
def train_model_command(random_seed: int):
    from src.classification.train_model import train_model

    train_model(random_seed=random_seed)


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
    pass


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
