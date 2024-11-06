# Copyright 2022 Fagner Cunha
# Copyright 2023 Rolnick Lab at Mila Quebec AI Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# *** BORROWED AS-IS WITH SOME CHANGES. *** #


import os

import torch
import webdataset as wds
from preprocessing import get_image_transforms


def get_num_workers() -> int:
    """Gets the optimal number of DatLoader workers to use in the current job."""
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return torch.multiprocessing.cpu_count()


def identity(x):
    return x


def get_label_transform():
    return identity


def build_webdataset_pipeline(
    sharedurl,
    input_size,
    batch_size,
    is_training,
    preprocess_mode,
    shuffle_samples=None,
):
    if shuffle_samples is None:
        shuffle_samples = is_training

    transform = get_image_transforms(input_size, is_training, preprocess_mode)
    label_transform = get_label_transform()

    dataset = wds.WebDataset(sharedurl, shardshuffle=shuffle_samples)

    dataset = (
        dataset.decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(transform, label_transform)
    )

    num_avail_workers = get_num_workers()
    print(f"Number of available workers: {num_avail_workers}")
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=num_avail_workers, batch_size=batch_size
    )

    return loader
