# System packages
import os
import pathlib
import unittest

# 3rd party packages
import dotenv

# Load secrets and config from optional .env file
dotenv.load_dotenv()

# Optional environment variable
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")


def write_log(message: str) -> pathlib.Path:
    try:
        output_logs_dir = os.environ["OUTPUT_LOGS_DIR"]
    except KeyError:
        raise KeyError("OUTPUT_LOGS_DIR environment variable not set")

    path = pathlib.Path(output_logs_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    filepath = path / "log.txt"

    with open(filepath, "a") as f:
        f.write(message + "\n")

    return filepath


class TestWriteLog(unittest.TestCase):
    def test_write_log(self):
        message = "Hello world"
        filepath = write_log(message)
        self.assertTrue(filepath.exists())

        with open(filepath, "r") as f:
            self.assertEqual(f.read().strip(), message)


if __name__ == "__main__":
    # Run tests
    unittest.main()
