import re

import pytest
from click.testing import CliRunner

from src.dataset_tools.cli import COMMAND_KEYS, COMMANDS, COMMANDS_HELP, cli


def _normalize_ws(text: str) -> str:
    """Collapse runs of whitespace to a single space.

    Click wraps long help text across lines based on terminal width, so the
    raw ``--help`` output cannot be compared directly against the single-line
    help strings. Normalising whitespace on both sides makes the comparison
    insensitive to line wrapping.
    """
    return re.sub(r"\s+", " ", text).strip()


command_test_parameters = [COMMANDS[cmd_key] for cmd_key in COMMAND_KEYS]
command_and_help_test_parameters = [
    (COMMANDS[cmd_key], COMMANDS_HELP[cmd_key]) for cmd_key in COMMAND_KEYS
]


def test_main_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in command_test_parameters:
        assert command in result.output


@pytest.mark.parametrize("command", command_test_parameters)
def test_command_no_arg(command):
    runner = CliRunner()
    result = runner.invoke(cli, [command])
    assert result.exit_code != 0
    assert "Missing option" in result.output


@pytest.mark.parametrize("command, command_help", command_and_help_test_parameters)
def test_command_help(command, command_help):
    runner = CliRunner()
    result = runner.invoke(cli, [command, "--help"])
    assert result.exit_code == 0
    assert command in result.output
    assert _normalize_ws(command_help) in _normalize_ws(result.output)


def test_number_of_command_constants():
    num_of_keys = len(COMMAND_KEYS)
    num_of_commands = len(COMMANDS)
    num_of_help = len(COMMANDS_HELP)
    assert num_of_commands == num_of_help, "Each command should have its own help text"
    assert (
        num_of_keys == num_of_commands
    ), "The COMMAND_KEYS frozenset should be the same length as the number of COMMANDS"
