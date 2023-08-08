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
3. Run `poetry install` in the root of the repository
4. Install pre-commit hooks `poetry run pre-commit install`

To run scripts in the virtual environment, activate with `poetry shell` or run scripts with `poetry run <script>`.

## Usage

### General

Activate the virtual environment before running scripts

```bash
poetry shell
```

### Object Detection

The code in `src/localization` allows to:

- train a torchvision localization model, with `training.py`
- run inferences on a set of images with a given localization model, with `inference_localization.py`
- visualize annotations/predictions with the `annotation_explorer.py` app
- run inferences on a set of images with Meta's Segment Anything Model (SAM), with `inference_sam.py`
- visualize the crops produced by SAM with the `crop_explorer.py` app
- visualize the Precision/Recall curve given the ground truths and the model predictions with the `threshold_explorer.py` app
- convert model predictions (as a json file) to ground truths that can be used for training, with `preds_to_ground_truth.py`

Example (in the poetry shell):

```bash
python src/localization/inference_localization.py \
  --data_dir ~/TRAPIMAGES/Sample/ \
  --ckpt_path ~/Downloads/fasterrcnn_mobilenet_v3_large_fpn_uqfh7u9w.pt \
  --model_type fasterrcnn_mobilenet_v3_large_fpn
```

The functioning of each script is detailed in its docstring. Scripts' options can be listed
directly in the CLI with the `--help` flag:

```bash
python src/localization/training.py --help
```
