# Object Detection

## Overview

Here is an overview of the scripts available in this folder:

| script                      | function                                                                                                                                                                                                         |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `training.py`               | train a [torchvision](https://pytorch.org/vision/main/models.html#object-detection) localization model                                                                                                           |
| `inference_localization.py` | run inferences on a set of images with a given localization model                                                                                                                                                |
| `annotation_explorer.py`    | visualize the models' predictions in a GUI app                                                                                                                                                                   |
| `inference_sam.py`          | run inferences on a set of images with Meta's [Segment Anything Model](https://github.com/facebookresearch/segment-anything) (SAM). It can be used both as an object detector or to segment objects in the image |
| `crop_explorer.py`          | visualize the segmented objects produced by SAM and filter out the bad crops                                                                                                                                     |
| `synthesize_images.py`      | create synthetic images given SAM's crops and background images                                                                                                                                                  |
| `threshold_explorer.py`     | visualize the Precision/Recall curve, given ground truths and model predictions                                                                                                                                  |
| `preds_to_ground_truth.py`  | convert model predictions (as a json file) to ground truths that can be used for training                                                                                                                        |
| `synthesize_images.py`      | create synthetic images given SAM's crops and background images                                                                                                                                                  |

The purpose and use of each script is detailed in its docstring. Scripts' options can be listed
directly in the CLI with the `--help` flag:

```bash
python src/localization/training.py --help
```

## Pipeline: creation of synthetic training data

The challenge of creating large and good-quality training datasets for object detection has been adressed by synthesizing the images.
The pipeline consists in 3 steps:

1. Cropping moths from trap images with SAM (`inference_sam.py`)
2. Reviewing the crops (`crop_explorer.py`)
3. Creating the synthetic dataset (`synthesize_images.py`)

The idea is that reviewing crops is _much_ faster than drawing bounding boxes.
Models trained on synthetic data have shown good performance on natural data.

## Threshold Analysis

Choosing the threshold in an important step for the successful deployment of a model
If a test set is available, the `threshold_explorer.py` GUI can help with that.
Having obtained the model's predictions on the test set with `inference_localization.py`,
this script will compute and display the Precision/Recall curve at the given IoU (Intersection over Union)
threshold(s).
The IoU thresholds are used to match inferred bounding boxes and ground truth boxes.
Two boxes match if their IoU is over the threshold.

- A bbox with a matching ground truth is a true positive
- A bbox without a matching ground truth is a false positive
- A ground truth without a bbox is a false negative

**The GUI allows to see what precision and recall would be expected at a given score threshold.**

The word _threshold_ has been used a lot in this paragraph: there's the score threshold and then there's the IoU threshold, which are not to be confused.
