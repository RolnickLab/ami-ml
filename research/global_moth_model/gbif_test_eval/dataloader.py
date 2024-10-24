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

import json
import os
import tarfile

import braceexpand
import preprocessing
import torch
import webdataset as wds
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "dataloader_num_workers",
    default=4,
    help=("How many subprocesses to use for dataloading"),
)

flags.DEFINE_string(
    "label_transform_json",
    default=None,
    help=("JSON file containing a label transformation map"),
)

flags.DEFINE_string(
    "sample_exclude_list",
    default=None,
    help=("JSON file containing a list of id samples to be skipped"),
)


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


def _load_exclude_list():
    with open(FLAGS.sample_exclude_list, "r") as f:
        exclude_list = json.load(f)
    exclude_list = [str(i).lower() for i in exclude_list]
    exclude_list = set(exclude_list)

    return exclude_list


def get_sample_filter_by_key():
    exclude_list = _load_exclude_list()

    def not_in_exclude_list(sample):
        sample_id = sample["__key__"].split("/")[-1]

        return sample_id not in exclude_list

    return not_in_exclude_list


def geo_prior_preprocess(json_data):
    lat = json_data["decimalLatitude"]
    lon = json_data["decimalLongitude"]
    date = json_data["eventDate"]

    feats, valid = preprocessing.preprocess_loc_date(lat, lon, date, validate=True)
    return feats, valid


def build_webdataset_pipeline(
    sharedurl,
    input_size,
    batch_size,
    is_training,
    preprocess_mode,
    return_instance_id=False,
    use_geoprior_data=False,
    shuffle_samples=None,
    multihead_map=None,
):
    if shuffle_samples is None:
        shuffle_samples = is_training

    transform = preprocessing.get_image_transforms(
        input_size, is_training, preprocess_mode
    )
    label_transform = get_label_transform()

    dataset = wds.WebDataset(sharedurl, shardshuffle=shuffle_samples)

    if shuffle_samples:
        dataset = dataset.shuffle(10000)

    if use_geoprior_data:
        dataset = dataset.decode("pil").to_tuple("jpg", "cls", "json", "__key__")
    else:
        dataset = dataset.decode("pil").to_tuple("jpg", "cls", "__key__")

    def map_fn(data):
        sample = []
        if use_geoprior_data:
            image, cls, json_data, data_key = data
            feats, valid = geo_prior_preprocess(json_data)
            sample += [feats, valid]
        else:
            image, cls, data_key = data

        image = transform(image)
        label = label_transform(cls)

        if multihead_map is not None:
            label = tuple([head[label] for head in multihead_map])

        sample = [image, label] + sample

        if return_instance_id:
            sample += [data_key]

        return tuple(sample)

    dataset = dataset.map(map_fn)

    num_avail_workers = get_num_workers()
    print(f"Number of available workers: {num_avail_workers}")
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=num_avail_workers, batch_size=batch_size
    )

    return loader


def _count_files_from_tar(tar_filename, exclude_list=None, ext="jpg"):
    tar = tarfile.open(tar_filename)
    files = [f for f in tar.getmembers() if f.name.endswith(ext)]
    files = [
        f for f in files if f.name.split("/")[-1][: -(len(ext) + 1)] not in exclude_list
    ]
    count_files = len(files)
    tar.close()
    return count_files


def get_webdataset_length(sharedurl):
    if FLAGS.sample_exclude_list is not None:
        exclude_list = _load_exclude_list()
    else:
        exclude_list = {}

    tar_filenames = list(braceexpand.braceexpand(sharedurl))
    counts = [_count_files_from_tar(tar_f, exclude_list) for tar_f in tar_filenames]
    return int(sum(counts))
