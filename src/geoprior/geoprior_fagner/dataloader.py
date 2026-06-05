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

import calendar
import datetime
import json
import math

import torch
import pandas as pd
import numpy as np
from absl import flags
from torch.utils.data.sampler import Sampler

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "datetime_format",
    default="%Y-%m-%d %H:%M:%S+00:00",
    help=("Datetime format used to convert to days to float"),
)


def date2float(date):
    if date is None:
        date = datetime.datetime.today().strftime(FLAGS.datetime_format)
    dt = datetime.datetime.strptime(date, FLAGS.datetime_format).timetuple()
    year_days = 366 if calendar.isleap(dt.tm_year) else 365

    return dt.tm_yday / year_days


def encode_feat(feat, encode, concat_dim=0):
    if encode == "encode_cos_sin":
        return torch.cat(
            (torch.sin(math.pi * feat), torch.cos(math.pi * feat)), concat_dim
        )
    else:
        raise RuntimeError("%s not implemented" % encode)

    return feat


def preprocess_loc_date(
    lat,
    lon,
    date_c,
    valid=True,
    loc_encode="encode_cos_sin",
    date_encode="encode_cos_sin",
    use_date_feats=True,
    decode_date=False,
):
    if valid:
        lat = lat / 90.0
        lon = lon / 180.0
        if decode_date:
            date_c = date2float(date_c)
    else:
        lat = 0.0
        lon = 0.0
        date_c = 0.5

    lat = torch.tensor(lat).unsqueeze(-1)
    lat = encode_feat(lat, loc_encode)
    lon = torch.tensor(lon).unsqueeze(-1)
    lon = encode_feat(lon, loc_encode)
    feats = torch.cat((lat, lon), dim=0)

    if use_date_feats:
        date_c = date_c * 2.0 - 1.0
        date_c = torch.tensor(date_c).unsqueeze(-1)
        date_c = encode_feat(date_c, date_encode)
        feats = torch.cat((feats, date_c), dim=0)

    return feats.float()


class BalancedSampler(Sampler):
    # sample "evenly" from each from class
    def __init__(self, classes, num_per_class, use_replace=False, multi_label=False):
        self.class_dict = {}
        self.num_per_class = num_per_class
        self.use_replace = use_replace
        self.multi_label = multi_label

        if self.multi_label:
            self.class_dict = classes
        else:
            # standard classification
            un_classes = np.unique(classes)
            for cc in un_classes:
                self.class_dict[cc] = []

            for ii, _ in enumerate(classes):
                self.class_dict[classes[ii]].append(ii)

        if self.use_replace:
            self.num_exs = self.num_per_class * len(un_classes)
        else:
            self.num_exs = 0
            for cc in self.class_dict.keys():
                self.num_exs += np.minimum(len(self.class_dict[cc]), self.num_per_class)

    def __iter__(self):
        indices = []
        for cc in self.class_dict:
            if self.use_replace:
                indices.extend(
                    np.random.choice(self.class_dict[cc], self.num_per_class).tolist()
                )
            else:
                indices.extend(
                    np.random.choice(
                        self.class_dict[cc],
                        np.minimum(len(self.class_dict[cc]), self.num_per_class),
                        replace=False,
                    ).tolist()
                )
        # in the multi label setting there will be duplictes at training time
        np.random.shuffle(indices)  # will remain a list
        return iter(indices)

    def __len__(self):
        return self.num_exs


class LocationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_file,
        num_classes=None,
        loc_encode="encode_cos_sin",
        date_encode="encode_cos_sin",
        use_date_feats=True,
        use_photographers=False,
        remove_invalid=True,
        provide_validity_info_output=False,
        transform=None,
        default_label=0,
        return_instance_id=False,
    ):
        self.loc_encode = loc_encode
        self.date_encode = date_encode
        self.use_date_feats = use_date_feats
        self.use_photographers = use_photographers
        self.remove_invalid = remove_invalid
        self.provide_validity_info_output = provide_validity_info_output
        self.default_label = default_label
        self.num_classes = num_classes
        self.transform = transform
        self.return_instance_id = return_instance_id

        with open(json_file) as f:
            json_data = json.load(f)

        metadata = pd.DataFrame(json_data["images"])
        if "annotations" in json_data.keys():
            annotations = pd.DataFrame(json_data["annotations"])
            metadata = pd.merge(
                metadata,
                annotations[["image_id", "category_id"]],
                how="left",
                left_on="id",
                right_on="image_id",
            )
        else:
            metadata["category_id"] = self.default_label

        num_classes = len(json_data["categories"])
        if self.num_classes is None:
            self.num_classes = num_classes

        metadata = self._validate_location_info_from_metadata(metadata)
        if self.remove_invalid:
            metadata = metadata[metadata.valid].copy()
        metadata = metadata[
            ["id", "lat", "lon", "date_c", "valid", "user_id", "category_id"]
        ].copy()

        _, train_users = np.unique(metadata.user_id.to_numpy(), return_inverse=True)
        metadata["user_id"] = train_users
        self.num_users = len(metadata.user_id.unique())
        if self.use_photographers and self.num_users < 2:
            raise RuntimeError(
                "To add photographers branch to the model, data must"
                " have more than one photographer"
            )
        print(f"Number of photographers: {self.num_users}")

        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx].to_numpy()
        if self.transform:
            sample = self.transform(sample)

        instance_id, lat, lon, date_c, valid, user_id, category_id = sample

        feats = preprocess_loc_date(
            lat,
            lon,
            date_c,
            valid,
            self.loc_encode,
            self.date_encode,
            self.use_date_feats,
        )

        instance_id = torch.tensor(instance_id)
        valid = torch.tensor(float(valid))
        user_id = torch.tensor(int(user_id))
        user_id = torch.nn.functional.one_hot(user_id, self.num_users)
        category_id = torch.tensor(int(category_id))
        category_id = torch.nn.functional.one_hot(category_id, self.num_classes)

        if self.use_photographers:
            if self.provide_validity_info_output:
                if self.return_instance_id:
                    return feats, category_id, user_id, valid, instance_id
                else:
                    return feats, category_id, user_id, valid
            else:
                if self.return_instance_id:
                    return feats, category_id, user_id, instance_id
                else:
                    return feats, category_id, user_id
        else:
            if self.provide_validity_info_output:
                if self.return_instance_id:
                    return feats, category_id, valid, instance_id
                else:
                    return feats, category_id, valid
            else:
                if self.return_instance_id:
                    return feats, category_id, instance_id
                else:
                    return feats, category_id

    def _validate_location_info_from_metadata(self, metadata_df):
        metadata = metadata_df.copy()
        if "longitude" not in metadata.columns:
            raise RuntimeError(
                "Logintude info does not exists on dataset_json."
                " Please add to json or specify location_info_json."
            )
        if "latitude" not in metadata.columns:
            raise RuntimeError(
                "Latitude info does not exists on dataset_json."
                " Please add to json or specify location_info_json."
            )
        if "date" not in metadata.columns:
            raise RuntimeError(
                "Date info does not exists on dataset_json."
                " Please add to json or specify location_info_json."
            )

        if "user_id" not in metadata.columns:
            metadata["user_id"] = 0

        if "valid" not in metadata.columns:
            metadata["valid"] = ~metadata.longitude.isna()
        else:
            metadata["valid"] = metadata["valid"].astype("bool")
        metadata["lat"] = metadata["latitude"]
        metadata["lon"] = metadata["longitude"]
        metadata["date_c"] = metadata.apply(lambda row: date2float(row["date"]), axis=1)

        return metadata

    def get_labels(self):
        return self.metadata.category_id.to_numpy()

    def get_num_classes(self):
        return self.num_classes

    def get_num_users(self):
        return self.num_users

    def get_num_feats(self):
        num_feats = 0

        if self.loc_encode == "encode_cos_sin":
            num_feats += 4

        if self.use_date_feats:
            if self.date_encode == "encode_cos_sin":
                num_feats += 2

        return num_feats


class RandSpatioTemporalGenerator:
    def __init__(
        self,
        rand_type="spherical",
        loc_encode="encode_cos_sin",
        date_encode="encode_cos_sin",
        use_date_feats=True,
    ):
        self.rand_type = rand_type
        self.loc_encode = loc_encode
        self.date_encode = date_encode
        self.use_date_feats = use_date_feats

    def get_rand_samples(self, batch_size):
        if self.rand_type == "spherical":
            rand_feats = torch.rand(batch_size, 3)
            theta1 = 2.0 * math.pi * rand_feats[:, 0]
            theta2 = torch.acos(2.0 * rand_feats[:, 1] - 1.0)
            lat = 1.0 - 2.0 * theta2 / math.pi
            lon = (theta1 / math.pi) - 1.0
            time = rand_feats[:, 2] * 2.0 - 1.0

            lon = lon.unsqueeze(1)
            lat = lat.unsqueeze(1)
            time = time.unsqueeze(1)
        else:
            raise RuntimeError("%s rand type not implemented" % self.rand_type)

        lon = encode_feat(lon, self.loc_encode, concat_dim=1)
        lat = encode_feat(lat, self.loc_encode, concat_dim=1)
        time = encode_feat(time, self.date_encode, concat_dim=1)

        if self.use_date_feats:
            return torch.cat([lat, lon, time], 1)
        else:
            return torch.cat([lat, lon], 1)
