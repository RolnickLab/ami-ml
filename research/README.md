# AMI Research
This directory contains code related to the research associated with the AMI project.

## Usage
An example environment variables file is provided in `.env.example`. To use it, rename the file to `.env` and update the values as needed.

The research scripts depend on the optional `research` extras. Install them from the repository root with `uv sync --extra research`, then run a script either inside the activated environment:

1. `source .venv/bin/activate`
2. `cd research/`
3. `python eccv2024/analyze_data.py`

or directly with `uv run`:

```bash
uv run python research/eccv2024/analyze_data.py
```
