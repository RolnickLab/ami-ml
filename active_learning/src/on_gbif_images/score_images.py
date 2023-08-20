"""
    

"""
import os
import typing as tp

import click
import torch
from torch import nn
from tqdm import tqdm
from utils import Resnet50, build_webdataset_pipeline

SupportedAcquisitionFunc = tp.Literal["entropy", "mutual_info", "varion_ratios"]


@click.command(context_settings={"show_default": True})
@click.option("--web_dataset_path", type=click.Path(exists=True), required=True)
@click.option(
    "--ckpt_path",
    "ckpt_paths",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    multiple=True,
)
@click.option("--num_classes", type=int, required=True)
@click.option(
    "--scoring_func",
    "scoring_functions"
    type=click.Choice(tp.get_args(SupportedAcquisitionFunc)),
    required=True,
    multiple=True,
)
@click.option("--image_resize", type=int, default=300)
@click.option("--batch_size", type=int, default=128)
@click.option("--num_workers", type=int, default=1)
@torch.no_grad()
def score_images(
    web_dataset_path: str,
    ckpt_paths: tp.List[str],
    num_classes: int,
    scoring_functions: SupportedAcquisitionFunc,
    image_resize: int,
    batch_size: int,
    num_workers: int,
):
    # Configure dataloader from webdataset
    dataloader = build_webdataset_pipeline(
        sharedurl=web_dataset_path,
        input_size=image_resize,
        batch_size=batch_size,
        set_type="test",
        num_workers=num_workers,
        preprocess_mode="torch",
        test_set_num=4,
    )

    # Load models
    models = load_models(ckpt_paths, num_classes)

    scores = {}

    for image_batch, image_names in tqdm(dataloader):
        batch_predictions = ensemble_inference(models, image_batch)
        batch_scores = score_batch(batch_predictions, scoring_functions)
        # add results to scores dict
    return


def ensemble_inference(models: tp.List[nn.Module], image_batch):
    for model in models:
        device = next(model.parameters()).device
        batch_predictions = model(image_batch.to(device))
    pass

def score_batch(batch_predictions, scoring_functions):
    pass


def load_models(ckpt_paths: tp.List[str], num_classes: int) -> tp.List[nn.Module]:
    """All models are loaded to gpu (if available). This approach would fail if too
    many models are given. The alternatives are:
    A) for each batch, load a model to gpu, infer, load the next model, and so on;
    B) load a model, infer over the whole dataset, and so on; this means that model
    predictions need to be saved for the whole dataset, which seems undesirable
    """

    # Device
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load weights from checkpoints and move to device
    models = []
    for ckpt_path in ckpt_paths:
        checkpoint = torch.load(ckpt_path)
        model = Resnet50(num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(device)
        models.append(model)

        name = os.path.basename(ckpt_path)
        print(f"Loaded model {name} to {device}")

    return models


if __name__ == "__main__":
    score_images()
