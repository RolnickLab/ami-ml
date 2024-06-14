# ML for Automated Monitoring of Insects

Software, algorithms and research related to the Automated Monitoring of Insects using deep learning and other machine learning methods.

<table>
<tr>
  <td>
<img width="200px" alt="Monitoring station deployment in field" src="https://user-images.githubusercontent.com/158175/212795444-3f638f4b-78f9-4f94-adf0-f2269427b441.png">
</td>
<td>
  <img width="200px" alt="Emerald moths detected in processed images" src="https://user-images.githubusercontent.com/158175/212794681-45a51172-1431-4475-87a8-9468032d6f7d.png">
</td>
  <td>
<img width="200px" alt="Monitoring station deployment in field" src="https://github.com/RolnickLab/ami-ml/assets/158175/1e6f9a7e-9744-43f6-be85-f53e9b684d27">
</td>
  <td>
<img width="200px" alt="Monitoring station deployment in field" src="https://github.com/RolnickLab/ami-ml/assets/158175/42db2783-5ccd-4de5-9f27-1f18b2b7f544">
</td>
</tr>
</table>

## Setup

Poetry is used to manage the dependencies common to all scripts and sub-projects. Some sub-projects may manage their own dependencies if necessary.

1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Clone this repository
2. Create a `.env` or copy `.env.example` and update the values
3. Run `poetry install` in the root of the repository
4. Install pre-commit hooks `poetry run pre-commit install`

### [Optional] Conda + Poetry
An optional way to setup the environment is to use [Conda](https://conda.io/projects/conda/en/latest/index.html) for creating and managing the environment, while using [Poetry](https://python-poetry.org/) for managing the packages and dependencies. Run the following steps to setup:
1. [Install Conda](https://docs.anaconda.com/free/miniconda/)
2. Create conda environment using the `environment.yml`: `conda env create -f environment.yml`
3. Activate the conda environment: `conda activate ami-ml`
4. Install packages in the root of the repository using Poetry: `poetry install`

## Usage

Activate the virtual environment before running scripts

```bash
poetry shell
```

Example for running a script (in the poetry shell):

```bash
python src/localization/inference_localization.py \
  --data_dir ~/TRAPIMAGES/Sample/ \
  --ckpt_path ~/Downloads/fasterrcnn_mobilenet_v3_large_fpn_uqfh7u9w.pt \
  --model_type fasterrcnn_mobilenet_v3_large_fpn
```

Alternatively, one can run the scripts without activating poetry's shell:

```bash
 poetry run python <script>
```
