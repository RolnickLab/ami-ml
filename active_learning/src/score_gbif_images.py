"""


"""
import os
import typing as tp

import click
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
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=num_workers, batch_size=batch_size
    )

    # Load models
    models = load_models(ckpt_paths, num_classes)

    scores = {}

    for image_batch, image_names in tqdm(data_loader):
        batch_preds = ensemble_inference(models, image_batch)
        batch_scores = score_batch(batch_preds, scoring_functions)
        breakpoint()
        # add results to scores dict
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


if __name__ == "__main__":
    score_images()
