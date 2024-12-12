#!/usr/bin/env python
# coding: utf-8

"""
Author        : Aditya Jain
Last modified : May 10th, 2023
About         : This is the main training file 
"""

import wandb
import torch
from torch import nn
import json
import os
import numpy as np
import torch.optim as optim
import pandas as pd
import time
import argparse
import random
import datetime

from data import dataloader
from models.build_model import build_model
from training_params.loss import Loss
from training_params.optimizer import Optimizer
from training_params.lr_scheduler import LRScheduler
from evaluation.micro_accuracy import (
    MicroAccuracyBatch,
    add_batch_microacc,
    final_micro_accuracy,
)
from evaluation.macro_accuracy import (
    MacroAccuracyBatch,
    add_batch_macroacc,
    final_macro_accuracy,
)
from evaluation.taxon_accuracy import (
    taxon_accuracy,
    add_taxon_accuracy_to_species_checklist,
)


def train_model(args):
    """main function for training"""

    # Reading data from configuration file required for training
    config_data = json.load(open(args.config_file))
    print(json.dumps(config_data, indent=3))

    wandb.init(
        project=config_data["training"]["wandb"]["project"],
        entity=config_data["training"]["wandb"]["entity"],
        settings=wandb.Settings(start_method="fork"),
        config=config_data,
    )

    image_resize = config_data["training"]["image_resize"]
    batch_size = config_data["training"]["batch_size"]
    label_list = config_data["dataset"]["label_info"]
    epochs = config_data["training"]["epochs"]
    loss_name = config_data["training"]["loss"]["name"]
    early_stop = config_data["training"]["early_stopping"]
    start_val_loss = config_data["training"]["start_val_loss"]

    label_read = json.load(open(label_list))
    num_classes = len(label_read["species"])

    model_type = config_data["model"]["type"]
    preprocess_mode = config_data["model"]["preprocess_mode"]
    optimizer = config_data["training"]["optimizer"]["name"]
    learning_rate = config_data["training"]["optimizer"]["learning_rate"]
    momentum = config_data["training"]["optimizer"]["momentum"]
    lr_scheduler = config_data["training"]["lr_scheduler"]["name"]

    dtstr = datetime.datetime.now()
    dtstr = dtstr.strftime("%Y-%m-%d-%H-%M")
    model_save_path = config_data["training"]["model_save_path"]
    model_name = config_data["training"]["model_name"]
    
    default_save_as = os.path.join(model_save_path, model_name + "_" + dtstr + f"_{wandb.run.id}.pt")
    save_ckpt_on_epochs = config_data["training"]["save_ckpt_on_epochs"]

    # Read other dataset specific files
    taxon_hierarchy = config_data["dataset"]["taxon_hierarchy"]
    label_info = config_data["dataset"]["label_info"]
    species_checklist = pd.read_csv(config_data["dataset"]["species_checklist"])

    # Loading model
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = build_model(num_classes, model_type)

    # Making use of multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Loading data
    # Training data loader
    train_dataloader = dataloader.build_webdataset_pipeline(
        sharedurl=args.train_webdataset_url,
        input_size=image_resize,
        batch_size=batch_size,
        set_type="train",
        num_workers=args.dataloader_num_workers,
        preprocess_mode=preprocess_mode,
    )

    # Validation data loader
    val_dataloader = dataloader.build_webdataset_pipeline(
        sharedurl=args.val_webdataset_url,
        input_size=image_resize,
        batch_size=batch_size,
        set_type="validation",
        num_workers=args.dataloader_num_workers,
        preprocess_mode=preprocess_mode,
    )

    # Test data loader
    test_dataloader = dataloader.build_webdataset_pipeline(
        sharedurl=args.test_webdataset_url,
        input_size=image_resize,
        batch_size=batch_size,
        set_type="test",
        num_workers=args.dataloader_num_workers,
        preprocess_mode=preprocess_mode,
        test_set_num=4,
    )

    # Loading loss function, optimizer, and learning rate scheduler
    loss_func = Loss(loss_name).get_loss()
    optimizer = Optimizer(optimizer, model, learning_rate, momentum).get_optim()
    scheduler = LRScheduler(lr_scheduler, optimizer, epochs).get_scheduler()

    # Model training
    lowest_val_loss = start_val_loss
    early_stp_count = 0

    for epoch in range(epochs):
        train_loss = 0
        train_batch_cnt = 0
        val_loss = 0
        val_batch_cnt = 0
        s_time = time.time()

        global_microacc_data_train = None
        global_microacc_data_val = None

        # model training on training dataset
        model.train()
        for image_batch, label_batch in train_dataloader:
            image_batch, label_batch = image_batch.to(
                device, non_blocking=True
            ), label_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(image_batch)
            t_loss = loss_func(outputs, label_batch)
            t_loss.backward()
            optimizer.step()
            train_loss += t_loss.item()

            # Micro-accuracy calculation
            micro_accuracy_train = MicroAccuracyBatch(
                outputs, label_batch, label_info, taxon_hierarchy
            ).batch_accuracy()
            global_microacc_data_train = add_batch_microacc(
                global_microacc_data_train, micro_accuracy_train
            )
            train_batch_cnt += 1
        train_loss = train_loss / train_batch_cnt

        # Update the learning rate
        scheduler.step()

        model.eval()
        # Model evaluation on validation dataset
        for image_batch, label_batch in val_dataloader:
            image_batch, label_batch = image_batch.to(
                device, non_blocking=True
            ), label_batch.to(device, non_blocking=True)

            outputs = model(image_batch)
            v_loss = loss_func(outputs, label_batch)
            val_loss += v_loss.item()

            # Micro-accuracy calculation
            micro_accuracy_val = MicroAccuracyBatch(
                outputs, label_batch, label_info, taxon_hierarchy
            ).batch_accuracy()
            global_microacc_data_val = add_batch_microacc(
                global_microacc_data_val, micro_accuracy_val
            )
            val_batch_cnt += 1
        val_loss = val_loss / val_batch_cnt

        if val_loss < lowest_val_loss:
            save_model(model, optimizer, epoch, default_save_as)

            lowest_val_loss = val_loss
            early_stp_count = 0
        else:
            early_stp_count += 1

        # Log metrics
        final_micro_accuracy_train = final_micro_accuracy(global_microacc_data_train)
        final_micro_accuracy_val = final_micro_accuracy(global_microacc_data_val)
        e_time = (time.time() - s_time) / 60  # time taken in minutes
        wandb.log(
            {
                "epoch": epoch,
                "training loss": train_loss,
                "validation loss": val_loss,
                "learning rate": scheduler.get_last_lr()[0],
                "train_micro_species_top1": final_micro_accuracy_train[
                    "micro_species_top1"
                ],
                "train_micro_genus_top1": final_micro_accuracy_train[
                    "micro_genus_top1"
                ],
                "train_micro_family_top1": final_micro_accuracy_train[
                    "micro_family_top1"
                ],
                "val_micro_species_top1": final_micro_accuracy_val[
                    "micro_species_top1"
                ],
                "val_micro_genus_top1": final_micro_accuracy_val["micro_genus_top1"],
                "val_micro_family_top1": final_micro_accuracy_val["micro_family_top1"],
                "time per epoch": e_time,
            }
        )

        if save_ckpt_on_epochs is not None and epoch in save_ckpt_on_epochs:
            save_as = os.path.join(
                model_save_path, model_name + f"_{wandb.run.id}_epoch_{epoch}.pt"
            )
            save_model(model, optimizer, epoch, save_as)

        if early_stp_count >= early_stop:
            break

    wandb.log_artifact(default_save_as, name=model_name, type="models")
    model.eval()
    global_micro_acc_data = None
    global_macro_acc_data = None

    print(f"Prediction on test data started ...")

    with torch.no_grad():
        for image_batch, label_batch in test_dataloader:
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            predictions = model(image_batch)

            # Micro-accuracy calculation
            micro_accuracy = MicroAccuracyBatch(
                predictions, label_batch, label_info, taxon_hierarchy
            ).batch_accuracy()
            global_micro_acc_data = add_batch_microacc(
                global_micro_acc_data, micro_accuracy
            )

            # Macro-accuracy calculation
            macro_accuracy = MacroAccuracyBatch(
                predictions, label_batch, label_info, taxon_hierarchy
            ).batch_accuracy()
            global_macro_acc_data = add_batch_macroacc(
                global_macro_acc_data, macro_accuracy
            )

    final_micro_acc = final_micro_accuracy(global_micro_acc_data)
    final_macro_acc, taxon_acc = final_macro_accuracy(global_macro_acc_data)
    taxa_accuracy = taxon_accuracy(taxon_acc, label_read)
    species_checklist_w_accuracy = add_taxon_accuracy_to_species_checklist(
        species_checklist, taxa_accuracy
    )

    wandb.log({"final micro accuracy": final_micro_acc})
    wandb.log({"final macro accuracy": final_macro_acc})

    # Convert the DataFrame into a W&B Table and log on W&B
    species_checklist_w_accuracy_table = wandb.Table(
        dataframe=species_checklist_w_accuracy
    )
    species_checklist_w_accuracy_table_artifact = wandb.Artifact(
        "species_accuracy_artifact", type="results"
    )
    species_checklist_w_accuracy_table_artifact.add(
        species_checklist_w_accuracy_table, "species_accuracy_table"
    )
    wandb.log_artifact(species_checklist_w_accuracy_table_artifact)

    # Save DataFrame locally as csv
    species_checklist_w_accuracy.to_csv(
        os.path.join(
            model_save_path, model_name + "_" + dtstr + f"_{wandb.run.id}_checklist_accuracy.csv"
        ),
        index=False,
    )

    wandb.finish()


def save_model(
    model: nn.Module, optimizer: optim.Optimizer, epoch: int, save_path: str
):
    if torch.cuda.device_count() > 1:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            save_path,
        )
    else:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            save_path,
        )


def set_random_seed(random_seed: int):
    """set random seed for reproducibility"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_webdataset_url",
        help="path to webdataset tar files for training",
        required=True,
    )
    parser.add_argument(
        "--val_webdataset_url",
        help="path to webdataset tar files for validation",
        required=True,
    )
    parser.add_argument(
        "--test_webdataset_url",
        help="path to webdataset tar files for testing",
        required=True,
    )
    parser.add_argument(
        "--config_file",
        help="path to configuration file containing training information",
        required=True,
    )
    parser.add_argument(
        "--dataloader_num_workers",
        help="number of cpus available",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--random_seed",
        help="random seed for reproducible experiments",
        default=42,
        type=int,
    )
    args = parser.parse_args()

    set_random_seed(args.random_seed)
    train_model(args)
