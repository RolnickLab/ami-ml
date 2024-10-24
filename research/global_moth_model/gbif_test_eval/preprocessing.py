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

from functools import partial

import numpy as np
from absl import flags
from torchvision import transforms

FLAGS = flags.FLAGS


def random_resize(image, full_size=300):
    random_num = np.random.uniform()
    if random_num <= 0.25:
        transform = transforms.Resize((int(0.5 * full_size), int(0.5 * full_size)))
        image = transform(image)
    elif random_num <= 0.5:
        transform = transforms.Resize((int(0.25 * full_size), int(0.25 * full_size)))
        image = transform(image)

    return image


def pad_to_square(image):  # CHANGE
    """Padding transformation to make the image square"""

    width, height = image.size
    if height < width:
        transform = transforms.Pad(padding=[0, 0, 0, width - height])
    elif height > width:
        transform = transforms.Pad(padding=[0, 0, height - width, 0])
    else:
        transform = transforms.Pad(padding=[0, 0, 0, 0])

    return transform(image)


def get_image_transforms(input_size=224, is_training=True, preprocess_mode="torch"):
    if preprocess_mode == "torch":
        # imagenet preprocessing
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif preprocess_mode == "tf":
        # global butterfly preprocessing (-1 to 1)
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    else:
        # (0 to 1)
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    ops = []
    # CHANGE: Add square padding
    ops += [transforms.Lambda(pad_to_square)]

    if is_training:
        if FLAGS.use_mixres:
            print("Mix res is implemented.")
            f_random_resize = partial(random_resize, full_size=input_size)
            ops += [transforms.Lambda(f_random_resize)]

        if FLAGS.dataaug == "randaug":
            print("Rand augmentation is implemented.")
            ops += [
                transforms.RandomResizedCrop(input_size, scale=(0.3, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
            ]
        elif FLAGS.dataaug == "simple":
            ops += [
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
            ]
    else:
        ops += [transforms.Resize((input_size, input_size))]

    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    return transforms.Compose(ops)
