import pytest
from click.testing import CliRunner

from src.dataset_tools.cli import COMMAND_KEYS, COMMANDS, COMMANDS_HELP, cli

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
    assert command_help in result.output
