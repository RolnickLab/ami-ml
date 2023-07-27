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
4. Run `poetry shell` to activate the virtual environment
5. OR run scripts using `poetry run python <script>`


## Usage


Activate the virtual environment

```bash
poetry shell
```

### Object Detection

Run object detection using a pre-trained model on a directory of images

```bash
python src/localization/inference_localization.py \
  --data_dir ~/TRAPIMAGES/Sample/
```

Review results in a GUI

```bash
python src/localization/annotations_explorer.py \
  --img_dir ~/TRAPIMAGES/Sample/ \
  --annotations_path ~/TRAPIMAGES/Sample/predictions_.json
```

Run detection using a custom trained model

```bash
python src/localization/inference_localization.py \
  --data_dir ~/TRAPIMAGES/Sample/ \
  --ckpt_path ~/Downloads/fasterrcnn_mobilenet_v3_large_fpn_uqfh7u9w.pt \
  --model_type fasterrcnn_mobilenet_v3_large_fpn
```
