#!/usr/bin/env python3
"""
Geo-prior FCNet training with wandb logging.

Wraps the original geo_prior/train_geo_net.py from fagner-lepsAI, adding:
  - wandb run init + per-step/per-epoch logging + finish
  - Saves checkpoint after each epoch (not just final) so tmux interruption
    leaves a usable model behind.

Original script: github.com/mihow/fagner-lepsAI/blob/main/geo_prior/train_geo_net.py
"""
import datetime
import os
import random
import time

import numpy as np
import torch
import wandb
from absl import app, flags
from timm.utils import AverageMeter

from src.geoprior import config

# Geo-prior network (FCNet) modules, from Fagner's lepsAI — see geoprior_fagner/README.md
from src.geoprior.geoprior_fagner import dataloader, losses, models

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "train_data_json",
    default=str(config.DATA_DIR / "train.json"),
    help="Path to JSON file containing training data",
)
flags.DEFINE_integer("batch_size", default=1024, help="Batch size")
flags.DEFINE_string(
    "loc_encode", default="encode_cos_sin", help="Encoding type for location coords"
)
flags.DEFINE_string(
    "date_encode", default="encode_cos_sin", help="Encoding type for date"
)
flags.DEFINE_bool("use_date_feats", default=True, help="Include date features")
flags.DEFINE_bool(
    "use_photographers", default=False, help="Include photographers classifier branch"
)
flags.DEFINE_integer(
    "max_instances_per_class",
    default=100,
    help="Max samples per class per epoch (BalancedSampler)",
)
flags.DEFINE_integer("epochs", default=30, help="Number of training epochs")
flags.DEFINE_integer("embed_dim", default=256, help="FCNet embedding dim")
flags.DEFINE_float("lr", default=0.0005, help="Initial learning rate")
flags.DEFINE_float("lr_decay", default=0.98, help="LR decay per epoch")
flags.DEFINE_integer("log_frequency", default=50, help="Log every N steps")
flags.DEFINE_string(
    "model_save_path",
    default=str(config.MODEL_DIR),
    help="Directory to save checkpoints",
)
flags.DEFINE_integer("dataloader_num_workers", default=4, help="DataLoader workers")
flags.DEFINE_integer("random_seed", default=42, help="Random seed")

# wandb-specific
flags.DEFINE_string("wandb_project", default=config.WANDB_PROJECT, help="W&B project")
flags.DEFINE_string("wandb_entity", default=config.WANDB_ENTITY, help="W&B entity")
flags.DEFINE_string(
    "wandb_run_name", default="geoprior-fcnet-global-12317cls-v1", help="W&B run name"
)
flags.DEFINE_bool("wandb_offline", default=False, help="Run wandb in offline mode")


def build_input_data(data_json, is_training, max_instances_per_class=0):
    loc_dataset = dataloader.LocationDataset(
        data_json,
        loc_encode=FLAGS.loc_encode,
        date_encode=FLAGS.date_encode,
        use_date_feats=FLAGS.use_date_feats,
        use_photographers=(FLAGS.use_photographers if is_training else False),
    )

    if is_training and max_instances_per_class > 0:
        shuffle = False
        sampler = dataloader.BalancedSampler(
            loc_dataset.get_labels().tolist(),
            num_per_class=max_instances_per_class,
            use_replace=False,
            multi_label=False,
        )
    else:
        shuffle = is_training
        sampler = None

    loc_dataloader = torch.utils.data.DataLoader(
        loc_dataset,
        num_workers=FLAGS.dataloader_num_workers,
        batch_size=FLAGS.batch_size,
        shuffle=shuffle,
        sampler=sampler,
    )

    return (
        loc_dataloader,
        loc_dataset.get_num_classes(),
        loc_dataset.get_num_users(),
        loc_dataset.get_num_feats(),
    )


def train_one_epoch(
    model,
    train_data,
    randgen,
    loc_o_loss,
    loc_p_loss,
    p_o_loss,
    optimizer,
    device,
    epoch,
    steps_per_epoch,
):
    batch_time = AverageMeter()
    running_loss = AverageMeter()
    running_obj_loss = AverageMeter()
    running_obj_loss_rand = AverageMeter()

    model.train()
    batch_end = time.time()
    for i, data in enumerate(train_data):
        if FLAGS.use_photographers:
            feats, labels, users = data
            users = users.to(device, non_blocking=True)
        else:
            feats, labels = data
            users = None
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        bs = feats.shape[0]
        rand_samples = randgen.get_rand_samples(bs).to(device, non_blocking=True)
        combined_inputs = torch.cat((feats, rand_samples), 0)
        rand_labels = torch.zeros(labels.shape, device=device)

        optimizer.zero_grad()
        loc_emb = model(combined_inputs, return_feats=True)
        loc_pred = torch.sigmoid(model.class_emb(loc_emb))

        obj_loss = loc_o_loss(labels, loc_pred[:bs])
        obj_loss_rand = loc_o_loss(rand_labels, loc_pred[bs:])
        loss = obj_loss + obj_loss_rand

        if FLAGS.use_photographers:
            user_pred = torch.sigmoid(model.user_emb(loc_emb))
            phot_loss = loc_p_loss(users, user_pred[:bs])
            user_pred_rand = 1 - user_pred[bs:]
            phot_loss_rand = loc_p_loss(users, user_pred_rand)
            p_c_given_u = torch.matmul(users, model.user_emb.weight.transpose(0, 1))
            p_c_given_u = torch.matmul(
                p_c_given_u, model.class_emb.weight.transpose(0, 1)
            )
            p_c_given_u = torch.sigmoid(p_c_given_u)
            phot_obj_loss = p_o_loss(labels, p_c_given_u)
            loss = loss + phot_loss + phot_loss_rand + phot_obj_loss

        loss.backward()
        optimizer.step()

        running_loss.update(loss.item(), labels.size(0))
        running_obj_loss.update(obj_loss.item(), labels.size(0))
        running_obj_loss_rand.update(obj_loss_rand.item(), labels.size(0))
        batch_time.update(time.time() - batch_end)
        batch_end = time.time()

        if i % FLAGS.log_frequency == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            eta = batch_time.avg * (steps_per_epoch - i)
            print(
                f"[{epoch:02d}/{FLAGS.epochs:02d}][{i:05d}/{steps_per_epoch:05d}]"
                f" ETA: {datetime.timedelta(seconds=int(eta))} -"
                f" loss: {running_loss.avg:.4f} -"
                f" obj_loss: {running_obj_loss.avg:.4f} -"
                f" obj_loss_rand: {running_obj_loss_rand.avg:.4f} -"
                f" lr: {current_lr:.8f}",
                flush=True,
            )
            wandb.log(
                {
                    "step/loss": running_loss.avg,
                    "step/obj_loss": running_obj_loss.avg,
                    "step/obj_loss_rand": running_obj_loss_rand.avg,
                    "step/lr": current_lr,
                    "step/batch_ms": batch_time.avg * 1000,
                    "step/eta_seconds": int(eta),
                    "step/epoch": epoch,
                }
            )

    return {
        "epoch/loss": running_loss.avg,
        "epoch/obj_loss": running_obj_loss.avg,
        "epoch/obj_loss_rand": running_obj_loss_rand.avg,
        "epoch/lr": optimizer.param_groups[0]["lr"],
        "epoch/batch_ms": batch_time.avg * 1000,
    }


def save_checkpoint(model, save_dir, epoch, is_final=False):
    os.makedirs(save_dir, exist_ok=True)
    suffix = "final" if is_final else f"epoch{epoch:02d}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"model_{suffix}_{timestamp}.pth")
    state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )
    torch.save(state_dict, path)
    return path


def set_random_seeds():
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    torch.manual_seed(FLAGS.random_seed)
    torch.cuda.manual_seed(FLAGS.random_seed)
    torch.backends.cudnn.deterministic = True


def main(_):
    set_random_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    print("Building training data...", flush=True)
    t0 = time.time()
    train_dataloader, num_classes, num_users, num_feats = build_input_data(
        FLAGS.train_data_json,
        is_training=True,
        max_instances_per_class=FLAGS.max_instances_per_class,
    )
    print(
        f"  num_classes={num_classes}, num_users={num_users}, num_feats={num_feats}",
        flush=True,
    )
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    randgen = dataloader.RandSpatioTemporalGenerator(
        loc_encode=FLAGS.loc_encode,
        date_encode=FLAGS.date_encode,
        use_date_feats=FLAGS.use_date_feats,
    )

    model = models.FCNet(num_feats, num_classes, FLAGS.embed_dim, num_users).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}", flush=True)

    loc_o_loss = losses.weighted_binary_cross_entropy(pos_weight=num_classes)
    loc_p_loss = losses.log_loss()
    p_o_loss = losses.weighted_binary_cross_entropy(pos_weight=num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=FLAGS.lr_decay)

    # wandb init
    wandb.init(
        project=FLAGS.wandb_project,
        entity=FLAGS.wandb_entity,
        name=FLAGS.wandb_run_name,
        mode="offline" if FLAGS.wandb_offline else "online",
        tags=[
            "geoprior",
            "fcnet",
            "global-butterflies",
            f"{num_classes}cls",
            f"embed-dim-{FLAGS.embed_dim}",
        ],
        config={
            "train_data_json": FLAGS.train_data_json,
            "epochs": FLAGS.epochs,
            "batch_size": FLAGS.batch_size,
            "embed_dim": FLAGS.embed_dim,
            "lr": FLAGS.lr,
            "lr_decay": FLAGS.lr_decay,
            "max_instances_per_class": FLAGS.max_instances_per_class,
            "use_date_feats": FLAGS.use_date_feats,
            "use_photographers": FLAGS.use_photographers,
            "num_classes": num_classes,
            "num_users": num_users,
            "num_feats": num_feats,
            "n_params": n_params,
            "random_seed": FLAGS.random_seed,
            "device": device,
        },
    )
    print(f"wandb run: {wandb.run.url}", flush=True)

    steps_per_epoch = len(train_dataloader)
    print(f"Steps per epoch: {steps_per_epoch}", flush=True)

    train_start = time.time()
    for epoch in range(1, FLAGS.epochs + 1):
        print(f"\n=== Starting epoch {epoch}/{FLAGS.epochs} ===", flush=True)
        epoch_start = time.time()
        epoch_metrics = train_one_epoch(
            model,
            train_dataloader,
            randgen,
            loc_o_loss,
            loc_p_loss,
            p_o_loss,
            optimizer,
            device,
            epoch,
            steps_per_epoch,
        )
        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_metrics["epoch/wall_seconds"] = epoch_time
        epoch_metrics["epoch"] = epoch
        wandb.log(epoch_metrics)
        print(
            f"Epoch {epoch}/{FLAGS.epochs} done in {epoch_time:.1f}s — "
            f"loss={epoch_metrics['epoch/loss']:.4f} "
            f"lr={epoch_metrics['epoch/lr']:.8f}",
            flush=True,
        )

        ckpt_path = save_checkpoint(model, FLAGS.model_save_path, epoch, is_final=False)
        print(f"  saved {ckpt_path}", flush=True)

    final_path = save_checkpoint(
        model, FLAGS.model_save_path, FLAGS.epochs, is_final=True
    )
    total_time = time.time() - train_start
    print(
        f"\nTraining complete in {datetime.timedelta(seconds=int(total_time))}",
        flush=True,
    )
    print(f"Final checkpoint: {final_path}", flush=True)

    wandb.run.summary["final_loss"] = epoch_metrics["epoch/loss"]
    wandb.run.summary["total_wall_seconds"] = total_time
    wandb.run.summary["final_checkpoint"] = final_path
    wandb.save(final_path)
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
