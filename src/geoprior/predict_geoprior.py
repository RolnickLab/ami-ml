#!/usr/bin/env python3
"""
Streaming version of geo_prior/predict.py.

The original predict.py accumulates ALL predictions in memory before writing
(~12 GB for our 240K val/test images × 12317-class output). This version
streams: each batch is written to disk immediately, no accumulation.

Output format matches the fusion code in classification/geo_prior.py:
    <results_dir>/preds/<gbif_id>.npy   — float32 array, shape (num_classes,)
    <results_dir>/valid/<gbif_id>.npy   — scalar bool/float
"""
import os
import random
import time

import numpy as np
import torch
from absl import app, flags

# Geo-prior network (FCNet) modules, from Fagner's lepsAI — see geoprior_fagner/README.md
from src.geoprior.geoprior_fagner import dataloader, models

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "test_data_json", default=None, help="Path to JSON file containing test data"
)
flags.DEFINE_integer("batch_size", default=1024, help="Batch size")
flags.DEFINE_string("loc_encode", default="encode_cos_sin", help="Loc encoding")
flags.DEFINE_string("date_encode", default="encode_cos_sin", help="Date encoding")
flags.DEFINE_bool("use_date_feats", default=True, help="Include date feats")
flags.DEFINE_integer("dataloader_num_workers", default=4, help="DataLoader workers")
flags.DEFINE_integer("embed_dim", default=256, help="FCNet embedding dim")
flags.DEFINE_integer("num_classes", default=12317, help="Number of classes")
flags.DEFINE_integer("num_users", default=0, help="Number of photographers")
flags.DEFINE_string("model_path", default=None, help="Path to checkpoint .pth")
flags.DEFINE_integer("log_frequence", default=50, help="Log every N steps")
flags.DEFINE_string("results_dir", default=None, help="Output dir")
flags.DEFINE_integer("random_seed", default=42, help="Random seed")

flags.mark_flag_as_required("test_data_json")
flags.mark_flag_as_required("model_path")
flags.mark_flag_as_required("results_dir")


def build_input_data():
    loc_dataset = dataloader.LocationDataset(
        FLAGS.test_data_json,
        loc_encode=FLAGS.loc_encode,
        date_encode=FLAGS.date_encode,
        use_date_feats=FLAGS.use_date_feats,
        use_photographers=False,
        remove_invalid=False,
        provide_validity_info_output=True,
        num_classes=FLAGS.num_classes,
        return_instance_id=True,
    )

    loc_dataloader = torch.utils.data.DataLoader(
        loc_dataset,
        num_workers=FLAGS.dataloader_num_workers,
        batch_size=FLAGS.batch_size,
        shuffle=False,
    )

    return loc_dataloader, loc_dataset.get_num_feats()


def load_prior_model(num_feats, device):
    model = models.FCNet(num_feats, FLAGS.num_classes, FLAGS.embed_dim, FLAGS.num_users)
    state = torch.load(
        FLAGS.model_path, map_location=torch.device(device), weights_only=True
    )
    model.load_state_dict(state)
    return model.to(device)


def generate_and_stream(prior_model, dataloader_iter, device, preds_dir, valid_dir):
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    prior_model.eval()
    total_written = 0
    t0 = time.time()
    with torch.no_grad():
        for i, data in enumerate(dataloader_iter):
            feats, _, valid, instance_id = data
            feats = feats.to(device, non_blocking=True)
            outputs = prior_model(feats).cpu().numpy().astype(np.float32)
            valids = valid.cpu().numpy().astype(np.float32)
            ids_np = instance_id.cpu().numpy()
            for j in range(len(ids_np)):
                sid = int(ids_np[j])
                np.save(os.path.join(preds_dir, f"{sid}.npy"), outputs[j])
                np.save(os.path.join(valid_dir, f"{sid}.npy"), valids[j])
            total_written += len(ids_np)
            if i % FLAGS.log_frequence == 0:
                elapsed = time.time() - t0
                rate = total_written / max(elapsed, 1e-6)
                print(
                    f"  batch {i:4d}  written={total_written:>7,}  "
                    f"rate={rate:>6.0f} samples/s  elapsed={elapsed:.1f}s",
                    flush=True,
                )
    return total_written, time.time() - t0


def set_random_seeds():
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    torch.manual_seed(FLAGS.random_seed)
    torch.cuda.manual_seed(FLAGS.random_seed)


def main(_):
    set_random_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    print(f"Test JSON: {FLAGS.test_data_json}", flush=True)
    print(f"Checkpoint: {FLAGS.model_path}", flush=True)
    print(f"Results dir: {FLAGS.results_dir}", flush=True)

    t0 = time.time()
    print("Loading test data...", flush=True)
    loader, num_feats = build_input_data()
    print(f"  loaded in {time.time()-t0:.1f}s, num_feats={num_feats}", flush=True)

    print("Loading model...", flush=True)
    model = load_prior_model(num_feats, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}", flush=True)

    preds_dir = os.path.join(FLAGS.results_dir, "preds")
    valid_dir = os.path.join(FLAGS.results_dir, "valid")
    print(f"Streaming predictions to {preds_dir}, {valid_dir}", flush=True)
    n, elapsed = generate_and_stream(model, loader, device, preds_dir, valid_dir)
    print(
        f"\nDone. Wrote {n:,} predictions in {elapsed:.1f}s "
        f"({n/elapsed:.0f} samples/s)",
        flush=True,
    )


if __name__ == "__main__":
    app.run(main)
