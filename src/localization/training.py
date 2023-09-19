#!/usr/bin/env python
# coding: utf-8

""" Training script.

Usage:
    python3 train_localization.py [OPTIONS]

The options can be visualized with 'python3 train_localization.py --help'.

This script has several limitations:
    - Resuming a training from a checkpoint is not supported for now
    - Using a config file instead of command-line arguments would be preferable
    - Support for learning rate schedulers and optimizers limited to those used here:
    https://github.com/pytorch/vision/tree/e35793a1a4000db1f9f99673437c514e24e65451/references/detection

"""

import os
import time
import typing as tp

import click
import torch
from torchvision import disable_beta_transforms_warning

disable_beta_transforms_warning()
import torchvision.transforms.v2 as T
import wandb
from data.custom_datasets import SplitDataset, TrainingDataset
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from utils import SupportedModels, bounding_box_to_tensor, load_model, set_random_seed

SupportedLRSchedulers = tp.Literal["multisteplr", "cosineannealinglr"]


def train_model_one_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    lr_scheduler: tp.Optional[optim.lr_scheduler.LRScheduler],
    epoch: int,
    warmup: bool,
):
    # Initialization
    model.train()
    num_batches = len(train_dataloader)
    train_loss = 0
    device = next(model.parameters()).device

    # Loop over batches
    for image_batch, target_batch in tqdm(
        train_dataloader, desc=f"Epoch {epoch+1}", position=1, leave=False
    ):
        image_batch = list(image.to(device) for image in image_batch)
        target_batch = [{k: v.to(device) for k, v in t.items()} for t in target_batch]

        output = model(image_batch, target_batch)
        batch_loss = sum(loss for loss in output.values())
        train_loss += batch_loss.item()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # During warmup, lr_scheduler step after each batch instead of epoch
        if warmup:
            lr_scheduler.step()

    train_loss = train_loss / num_batches

    if warmup is False and lr_scheduler is not None:
        lr_scheduler.step()

    return train_loss


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler: tp.Optional[optim.lr_scheduler.LRScheduler],
    lr_scheduler_warmup: tp.Optional[optim.lr_scheduler.LRScheduler],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    warmup_epochs: tp.Optional[int],
    early_stop: int,
):
    # Initialization
    device = next(model.parameters()).device
    metric = MeanAveragePrecision(max_detection_thresholds=[5, 10, 100])
    best_map = 0
    early_stp_count = 0

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training", position=0):
        start_time = time.time()
        metric.reset()

        if lr_scheduler_warmup is not None and epoch < warmup_epochs:
            warmup = True
            lr_scheduler_for_epoch = lr_scheduler_warmup
        else:
            warmup = False
            lr_scheduler_for_epoch = lr_scheduler

        # Training epoch
        train_loss = train_model_one_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler_for_epoch,
            epoch=epoch,
            warmup=warmup,
        )

        # Validation at every epoch
        with torch.no_grad():
            model.eval()
            for image_batch, target_batch in tqdm(
                val_dataloader, desc="End-of-epoch validation", position=2, leave=False
            ):
                image_batch = list(image.to(device) for image in image_batch)
                target_batch = [
                    {k: v.to(device) for k, v in t.items()} for t in target_batch
                ]

                preds = model(image_batch)
                target_batch_as_tensor = bounding_box_to_tensor(target_batch)
                metric.update(preds, target_batch_as_tensor)

            map = metric.compute()

        # Save checkpoint if improvement
        if map["map_75"] > best_map:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict()
                if lr_scheduler is not None
                else None,
                "train_loss": train_loss,
                "wandb_run_id": wandb.run.id,
            }
            best_map = map["map_75"]
            early_stp_count = 0
        else:
            early_stp_count += 1

        # Logging metrics
        epoch_time = (time.time() - start_time) / 60  # time in minutes

        if (
            lr_scheduler_for_epoch is not None
        ):  # Assumes same learning rate for all parameters
            current_lr = lr_scheduler_for_epoch.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        wandb.log(
            {
                "training loss": train_loss,
                "epoch": epoch,
                "map": map["map"].item(),
                "map_50": map["map_50"].item(),
                "map_75": map["map_75"].item(),
                "time per epoch": epoch_time,
                "learning_rate": current_lr,
            }
        )

        if early_stp_count >= early_stop:
            break

    return checkpoint


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(
    optimizer: torch.optim.Optimizer, warmup_iters: int, warmup_factor: float
):
    """This returns a scheduler for the linear warmup phase, in which the learning
    rate grows linearly from 'warmup_factor' to the learning rate given to the opimizer.
    """

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def prepare_dataloader(
    data_dir: str,
    train_val_ratio: int,
    model_type: SupportedModels,
    batch_size: int,
    num_workers: int,
) -> tp.Tuple[DataLoader, DataLoader]:
    """Returns the training and validation data loaders, which have different transforms
    (data augmentation is only applied on the training set)
    """
    if model_type == "ssdlite320_mobilenet_v3_large":
        transforms = T.Compose(
            [
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.SanitizeBoundingBox(),
            ]
        )
    else:
        transforms = T.Compose([T.RandomHorizontalFlip(p=0.5), T.ToTensor()])

    dataset = TrainingDataset(data_dir, transform=None)
    train_set_size = int(train_val_ratio / 100 * len(dataset))
    valid_set_size = len(dataset) - train_set_size
    train_set, val_set = random_split(dataset, [train_set_size, valid_set_size])
    train_set = SplitDataset(train_set, transform=transforms)
    val_set = SplitDataset(val_set)

    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_dataloader, valid_dataloader


@click.command(context_settings={"show_default": True})
@click.option(
    "--model_type",
    type=click.Choice(tp.get_args(SupportedModels)),
    default="fasterrcnn_resnet50_fpn",
)
@click.option("--batch_size", type=int, default=16)
@click.option("--num_workers", type=int, default=1)
@click.option("--pretrained", default=False)
@click.option("--pretrained_backbone", default=False)
@click.option("--train_val_ratio", type=int, default=80)
@click.option("--num_epochs", required=True, type=int)
@click.option(
    "--early_stop",
    help="number of epochs after which to stop training if MAP@75 does not improve",
    required=True,
    type=int,
)
@click.option("--lr", type=float, default=0.01, help="learning rate")
@click.option("--momentum", type=float, default=0.9)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option(
    "--lr_scheduler",
    type=click.Choice(tp.get_args(SupportedLRSchedulers)),
    default=None,
)
@click.option("--warmup_epochs", type=int, default=None)
@click.option(
    "--lr_step",
    "lr_steps",
    type=int,
    multiple=True,
    default=None,
    help="at what epoch to multiply lr by lr_gamma (multisteplr scheduler only)",
)
@click.option(
    "--lr_gamma",
    type=float,
    default=0.1,
    help="decrease lr by a factor of lr_gamma (multisteplr scheduler only)",
)
@click.option(
    "--anchor_size",
    "anchor_sizes",
    type=int,
    multiple=True,
    default=(32, 64, 128, 256, 512),
    help="fasterrcnn_mobilenet_v3_large_fpn only",
)
@click.option(
    "--trainable_backbone_layers",
    type=int,
    default=3,
    help="fasterrcnn_mobilenet_v3_large_fpn only",
)
@click.option(
    "--ckpt_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--data_dir",
    type=click.Path(exists=True, file_okay=False),
    help="directory containing the images and the json file with annotations",
    required=True,
)
@click.option("--dataset_name", type=str, default=None)
@click.option(
    "--save_dir",
    type=click.Path(exists=True, file_okay=False),
    help="directory where the model will be saved",
    required=True,
)
@click.option("--random_seed", help="For reproducibility", default=42, type=int)
@click.option(
    "--wandb_project",
    help="see wandb.init() documentation",
    default="Localization-Model",
)
@click.option(
    "--wandb_entity", help="see wandb.init() documentation", default="moth-ai"
)
@click.option(
    "--model_to_wandb",
    help="whether to upload the model on wandb",
    default=True,
    type=bool,
)
def main(
    data_dir: str,
    dataset_name: tp.Optional[str],
    save_dir: str,
    ckpt_path: tp.Optional[str],
    wandb_project: str,
    wandb_entity: str,
    model_to_wandb: bool,
    model_type: SupportedModels,
    pretrained: bool,
    pretrained_backbone: bool,
    batch_size: int,
    num_workers: int,
    num_epochs: int,
    early_stop: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    lr_scheduler: tp.Optional[SupportedLRSchedulers],
    lr_steps: tp.Optional[tp.Sequence[int]],
    lr_gamma: float,
    anchor_sizes: tp.Tuple[int, ...],
    trainable_backbone_layers: int,
    warmup_epochs: tp.Optional[int],
    train_val_ratio: int,
    random_seed: tp.Optional[int],
):
    set_random_seed(random_seed)

    # Initialize wandb api
    config = {
        "model_type": model_type,
        "pretrained": pretrained,
        "pretrained_backbone": pretrained_backbone,
        "batch_size": batch_size,
        "optimizer": "SGD",
        "learning_rate": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "lr_scheduler": lr_scheduler,
        "warmup_epochs": warmup_epochs,
        "lr_steps": lr_steps if lr_scheduler == "multisteplr" else None,
        "lr_gamma": lr_gamma if lr_scheduler == "multisteplr" else None,
        "dataset": dataset_name
        if dataset_name is not None
        else os.path.basename(os.path.normpath(data_dir)),
        "train_val_ratio": train_val_ratio,
        "random_seed": random_seed,
        "num_epochs": num_epochs,
        "early_stop": early_stop,
        "trainable_backbone_layers": trainable_backbone_layers
        if model_type == "fasterrcnn_mobilenet_v3_large_fpn"
        else None,
        "anchor_sizes": anchor_sizes
        if model_type == "fasterrcnn_mobilenet_v3_large_fpn"
        else None,
    }
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config=config,
        settings=wandb.Settings(start_method="fork"),
    )

    # Dataloaders
    train_dataloader, valid_dataloader = prepare_dataloader(
        data_dir, train_val_ratio, model_type, batch_size, num_workers
    )

    # Model
    model = load_model(
        model_type=model_type,
        pretrained=pretrained,
        pretrained_backbone=pretrained_backbone,
        num_classes=2,
        ckpt_path=ckpt_path,
        anchor_sizes=anchor_sizes,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if lr_scheduler == "multisteplr":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, lr_steps, gamma=lr_gamma
        )
    elif lr_scheduler == "cosineannealinglr":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    if warmup_epochs is not None:
        warmup_iters = len(train_dataloader) * warmup_epochs
        warmup_factor = 0.001
        lr_scheduler_warmup = warmup_lr_scheduler(
            optimizer, warmup_iters, warmup_factor
        )
    else:
        lr_scheduler_warmup = None

    # Training
    checkpoint = train_model(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_scheduler_warmup=lr_scheduler_warmup,
        train_dataloader=train_dataloader,
        val_dataloader=valid_dataloader,
        num_epochs=num_epochs,
        early_stop=early_stop,
        warmup_epochs=warmup_epochs,
    )

    # Save checkpoint locally and on wandb
    save_path = os.path.join(
        save_dir, model_type + "_" + checkpoint["wandb_run_id"] + ".pt"
    )
    torch.save(checkpoint, save_path)
    if model_to_wandb:
        wandb.log_artifact(save_path, name=model_type, type="model")


if __name__ == "__main__":
    main()
