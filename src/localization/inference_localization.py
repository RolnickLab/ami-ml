"""This script is used to run inferences with a given model on a set of jpg images. The
result of the inferences is saved in a json file, in the images' directory. The json
file contains, for each image, a list with three elements:
- a list of bounding boxes in the [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
    0 <= y1 < y2 <= H
- a list of labels (always 1 for the moth localization task, since there's only 1 class)
- a list with the corresponding scores
The number of bounding boxes that are saved for each image can be controlled with the
options --score_thr and --max_bboxes

Usage:
    python3 inference_localization.py [OPTIONS]

Options:
    --ckpt_path (str)
    --data_dir (str): where the jpg images are located.
    --model_type (str): by default, fasterrcnn_resnet50_fpn. Needs to match the model
    given at the checkpoint path.
    --batch_size (int): by default, 1.
    --num_workers (int): by default, 1.
    --device (str): cuda or cpu. By default, cuda.
    --score_thr (float): by default, None (i.e. all boxes are kept).
    --max_bboxes (int): by default, None (i.e. all boxes are kept).
    --sampling_rate (int): by default, 1. Can set a higher value if you don't want to
    run inferences on every image, but just on one in {sampling_rate}
    --stats (bool): by default, True.
    --anchor_size (int): by default, (32, 64, 128, 256, 512). This is only relevant when
    loading a fasterrcnn_mobilenet_v3_large_fpn. Each anchor size must be given by
    itself (e.g. "--anchor_size 32 --anchor_size 64 [...]")
"""
import json
import os
import time
import typing as tp

import click
import numpy as np
import torch
from data.custom_datasets import InferenceDataset
from torch import nn
from tqdm import tqdm
from utils import Devices, SupportedModels, compute_model_size, load_model


@click.command(context_settings={"show_default": True})
@click.option(
    "--model_type",
    type=click.Choice(tp.get_args(SupportedModels)),
    default="fasterrcnn_resnet50_fpn",
)
@click.option("--batch_size", type=int, default=1)
@click.option("--num_workers", type=int, default=1)
@click.option("--device", type=click.Choice(tp.get_args(Devices)), default="cuda")
@click.option("--score_thr", type=float, default=None)
@click.option("--max_bboxes", type=int, default=None)
@click.option("--sampling_rate", type=int, default=1)
@click.option(
    "--ckpt_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--data_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option("--stats", type=bool, default=True)
@click.option(
    "--anchor_size",
    "anchor_sizes",
    type=int,
    multiple=True,
    default=(32, 64, 128, 256, 512),
    help="fasterrcnn_mobilenet_v3_large_fpn only",
)
def main(
    data_dir: str,
    ckpt_path: tp.Optional[str],
    model_type: SupportedModels,
    device: Devices,
    score_thr: tp.Optional[float],
    max_bboxes: tp.Optional[int],
    batch_size: int,
    num_workers: int,
    sampling_rate: int,
    stats: bool,
    anchor_sizes: tp.Tuple[int, ...],
):
    model = load_model(
        model_type, device=device, ckpt_path=ckpt_path, anchor_sizes=anchor_sizes
    )

    if stats:
        size = compute_model_size(model)
        print(f"Size of loaded {model_type} is: {size:.2f}MB")

    dataset = InferenceDataset(data_dir, sampling_rate=sampling_rate)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    preds = inference(
        model, data_loader, score_thr=score_thr, max_bboxes=max_bboxes, stats=stats
    )
    model_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    with open(os.path.join(data_dir, "predictions_" + model_name + ".json"), "w") as f:
        json.dump(preds, f)

    return


@torch.no_grad()
def inference(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    score_thr: tp.Optional[float] = None,
    max_bboxes: tp.Optional[int] = None,
    stats: bool = True,
):
    model.eval()
    device = next(model.parameters()).device
    inference_times = []
    preds = {}

    for batch, ids in tqdm(data_loader):
        start_time = time.time()
        batch_preds = model(batch.to(device))
        inference_times.append(time.time() - start_time)

        for img_id, img_preds in zip(ids, batch_preds):
            bboxes = img_preds["boxes"].round().int()
            labels = img_preds["labels"]
            scores = img_preds["scores"]

            if score_thr is not None:
                bboxes = bboxes[scores > score_thr]
                labels = labels[scores > score_thr]
                scores = scores[scores > score_thr]

            if max_bboxes is not None:
                scores, indices_sorted = scores.sort(descending=True)
                bboxes = bboxes[indices_sorted][:max_bboxes]
                labels = labels[indices_sorted][:max_bboxes]
                scores = scores[:max_bboxes]

            preds[img_id] = [bboxes.tolist(), labels.tolist(), scores.tolist()]

    if stats:
        batch_size = data_loader.batch_size
        print(f"Image average inference time: {np.mean(inference_times)/batch_size}")

    return preds


if __name__ == "__main__":
    main()
