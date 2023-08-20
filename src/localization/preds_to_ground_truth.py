"""Converts predictions to groud truths by applying a threshold on the score (below the
threshold, bboxes are considered negatives and discarted, while above the score, they
are considered positives and are kept).

Usage:
    python3 preds_to_ground_truth.py [OPTIONS]

Options
    --preds_json_path : str
        Path to the json file containing the predictions.
        The keys of the json are the image IDs. For each image, a list with 3 elements
        is expected, in the following order:
        - a list of bounding boxes in the [x1, y1, x2, y2] format, with 0 <= x1 < x2 <=
        W and 0 <= y1 < y2 <= H
        - a list of labels
        - a list with the corresponding scores

    --score_thr : float
    --ground_truth_filename : str, optional
        The ground truths are saved in a json file at the same location as the
        predictions. If no filename is given, the predictions filename is used with the
        '_gt' suffix.

The format of the generated json file with the ground truths is the same as the given
predictions file, but withouth the scores, and only including boxes (and corresponding
labels) with score above the threshold. This file can be used for training a model.
"""


import json
import os
import typing as tp

import click
from utils import preds_to_ground_truth


@click.command(context_settings={"show_default": True})
@click.option(
    "--preds_json_path", type=click.Path(exists=True, dir_okay=False), required=True
)
@click.option("--score_thr", type=float, required=True)
@click.option("--ground_truth_filename", type=str, default=None)
def main(
    preds_json_path: str,
    score_thr: float,
    ground_truth_filename: tp.Optional[str],
):
    with open(preds_json_path) as f:
        preds = json.load(f)

    ground_thruths = preds_to_ground_truth(preds, score_thr)

    if ground_truth_filename is None:
        ground_truth_filename = os.path.splitext(os.path.basename(preds_json_path))[0]
        ground_truth_filename = ground_truth_filename + "_gt.json"

    if os.path.splitext(ground_truth_filename)[1] == "":
        ground_truth_filename = ground_truth_filename + ".json"

    gt_json_path = os.path.join(os.path.dirname(preds_json_path), ground_truth_filename)

    with open(gt_json_path, "w") as f:
        json.dump(ground_thruths, f)

    return


if __name__ == "__main__":
    main()
