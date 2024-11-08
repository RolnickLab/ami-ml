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


from torchvision import transforms


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
        print("Only meant for testing.")
    else:
        ops += [transforms.Resize((input_size, input_size))]

    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    return transforms.Compose(ops)
