# AMI Research
This directory contains code related to the research associated with the AMI project.

## Setup
[Conda](https://conda.io/projects/conda/en/latest/index.html) is used to manage the project environment and [Poetry](https://python-poetry.org/) for managing the project dependencies. To setup the environment, run the following commands:
1. [Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Create conda environment using the `environment.yml`: `conda env create -f environment.yml`
3. Activate the conda environment: `conda activate ami-ml`
4. Install packages in the root of the repository using Poetry: `poetry install`

## Usage
An example environment file is provided in `.env.example`. To use it, rename the file to `.env` and update the values as needed.

Example of running a script: Navigate to the `research` directory and run the following command:
```bash
python eccv2024/analyze_data.py
```
