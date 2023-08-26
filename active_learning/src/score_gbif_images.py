"""


"""
import os
import typing as tp
import numpy
import random

import click
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from utils.datasets import InferenceDataset
from utils.resnet50 import Resnet50
from utils.scoring_functions import SupportedScoringFunc, score_batch


@click.command(context_settings={"show_default": True})
@click.option(
    "--image_list_csv", type=click.Path(exists=True, dir_okay=False), required=True
)
@click.option(
    "--image_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--ckpt_path",
    "ckpt_paths",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    multiple=True,
)
@click.option(
    "--save_path",
    type=click.Path(exists=True, file_okay=False),
    help="directory where the computed scores will be save",
    required=True,
)
@click.option(
    "--save_name",
    type=str,
    help="filename for csv file",
    required=True,
)
@click.option("--num_classes", type=int, required=True)
@click.option(
    "--scoring_func",
    "scoring_functions",
    type=click.Choice(tp.get_args(SupportedScoringFunc)),
    required=True,
    multiple=True,
)
@click.option("--image_resize", type=int, default=300)
@click.option("--batch_size", type=int, default=128)
@click.option("--num_workers", type=int, default=1)
@torch.no_grad()
def score_images(
    image_list_csv: str,
    image_dir: str,
    ckpt_paths: tp.List[str],
    save_path: str,
    save_name: str,
    num_classes: int,
    scoring_functions: tp.List[SupportedScoringFunc],
    image_resize: int,
    batch_size: int,
    num_workers: int,
):
    # Build data loader
    dataset = InferenceDataset(
        image_list_csv,
        image_dir,
        image_resize,
        sampling_rate=1,
        preprocess_mode="torch",
    )

    g = torch.Generator().manual_seed(0)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Load models
    models = load_models(ckpt_paths, num_classes)

    scores = {}

    # Run inferences and score images
    for image_batch, image_names in tqdm(data_loader):
        batch_preds = ensemble_inference(models, image_batch)
        batch_scores = score_batch(batch_preds, scoring_functions)
        scores.update(scores_to_dict(image_names, scoring_functions, batch_scores))

    # Save results
    scores = pd.DataFrame(scores).T
    images_info = pd.read_csv(image_list_csv).drop_duplicates()
    merged = images_info.merge(
        scores, left_on="filename", right_index=True, how="inner"
    )
    merged.to_csv(os.path.join(save_path, save_name), index=False)

    return


def ensemble_inference(
    models: tp.List[nn.Module], image_batch: torch.Tensor
) -> torch.Tensor:
    """
    Parameters
    ----------
    models : tp.List[nn.Module]
        List of models in eval mode.
    image_batch : torch.Tensor
        Size is (batch_size, channels, width, height)

    Returns
    -------
    torch.Tensor
        Size is (nb_models, batch_size, nb_classes)
    """
    ensemble_predictions = []

    for model in models:
        device = next(model.parameters()).device
        model_predictions = model(image_batch.to(device))
        # size: batch_size x nb_classes
        model_predictions = nn.functional.softmax(model_predictions, dim=1).cpu()
        ensemble_predictions.append(model_predictions)

    ensemble_predictions = torch.stack(ensemble_predictions, dim=0)

    return ensemble_predictions


def scores_to_dict(
    image_keys: tp.List[str],
    scoring_functions: SupportedScoringFunc,
    batch_scores: torch.Tensor,
):
    scores_dict = {}
    for i, image_key in enumerate(image_keys):
        image_scores = {}
        for j, scoring_func in enumerate(scoring_functions):
            image_scores[scoring_func] = batch_scores[i, j].item()
        scores_dict[image_key] = image_scores

    return scores_dict


def load_models(ckpt_paths: tp.List[str], num_classes: int) -> tp.List[nn.Module]:
    """All models are loaded to gpu (if available). This approach would fail if too
    many models are given. The alternatives are:
    A) for each batch, load a model to gpu, infer, load the next model, and so on;
    B) load a model, infer over the whole dataset, and so on; this means that model
    predictions need to be saved for the whole dataset, which seems undesirable
    """

    # Device
    if torch.cuda.is_available():
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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    score_images()
