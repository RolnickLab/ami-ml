#!/usr/bin/env python
# coding: utf-8

"""
Visualize a webdataset using FiftyOne.

Usage:
    python src/dataset_tools/visualize_webdataset.py \
        --webdataset-pattern "data/vermont_butterflies/webdataset/vermont_train_verbatim-{000000..000051}.tar" \
        --category-map-json "data/vermont_butterflies/webdataset/vermont_category_map_verbatim.json" \
        --num-samples 500 \
        --port 5151

To access on a remote cluster via SSH tunnel:
    ssh -L 5151:NODE_NAME:5151 USERNAME@fir.alliancecan.ca
Then open: http://localhost:5151
"""

import argparse
import io
import json
import tempfile
from pathlib import Path

import fiftyone as fo
import webdataset as wds


def load_category_map(json_path: str) -> dict[int, str]:
    """Load category map JSON and return a reverse mapping from int label -> species name."""
    with open(json_path, "r") as f:
        categories = json.load(f)
    # categories maps "Species name" -> int; reverse it
    return {v: k for k, v in categories.items()}


def build_fiftyone_dataset(
    webdataset_pattern: str,
    category_map_json: str,
    num_samples: int,
    dataset_name: str,
    temp_dir: str,
) -> fo.Dataset:
    """Load samples from a webdataset and create a FiftyOne dataset."""
    int_to_label = load_category_map(category_map_json)

    dataset = fo.Dataset(name=dataset_name, overwrite=True)
    samples = []
    temp_path = Path(temp_dir)

    pipeline = (
        wds.WebDataset(webdataset_pattern)
        .decode("pil")
        .to_tuple("__key__", "jpg", "cls")
    )

    for i, (key, image, cls_label) in enumerate(pipeline):
        if i >= num_samples:
            break

        # Save image to temp dir
        img_path = temp_path / f"{key.replace('/', '_')}.jpg"
        image.save(str(img_path), format="JPEG")

        label = int_to_label.get(int(cls_label), str(cls_label))

        sample = fo.Sample(
            filepath=str(img_path),
            ground_truth=fo.Classification(label=label),
            cls_int=int(cls_label),
            key=key,
        )
        samples.append(sample)

        if (i + 1) % 100 == 0:
            print(f"Loaded {i + 1} samples...", flush=True)

    dataset.add_samples(samples)
    print(f"Dataset built with {len(dataset)} samples.", flush=True)
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a webdataset with FiftyOne."
    )
    parser.add_argument(
        "--webdataset-pattern",
        required=True,
        help="Glob/brace pattern for tar files, e.g. 'path/to/train-{000000..000051}.tar'",
    )
    parser.add_argument(
        "--category-map-json",
        required=True,
        help="Path to category map JSON (maps species name -> int label)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Maximum number of samples to load (default: 500)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5151,
        help="Port for the FiftyOne app (default: 5151)",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Writing images to temp dir: {temp_dir}", flush=True)

        dataset = build_fiftyone_dataset(
            webdataset_pattern=args.webdataset_pattern,
            category_map_json=args.category_map_json,
            num_samples=args.num_samples,
            dataset_name="webdataset_viz",
            temp_dir=temp_dir,
        )

        print(
            f"Launching FiftyOne app on port {args.port}. "
            "Access via SSH tunnel: ssh -L {port}:NODENAME:{port} USER@fir.alliancecan.ca".format(
                port=args.port
            ),
            flush=True,
        )

        session = fo.launch_app(dataset, remote=True, port=args.port)
        session.wait()


if __name__ == "__main__":
    main()
