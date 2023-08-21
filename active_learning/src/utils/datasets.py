"""
Note: the random_resize and identity functions were copy pasted from Aditya's repo
"""
import os
import typing as tp

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def random_resize(image):
    """randomly resize image given a probability distribution"""

    random_num = np.random.uniform()
    if random_num <= 0.25:
        transform = transforms.Resize((150, 150))
        new_image = transform(image)
    elif random_num > 0.25 and random_num <= 0.5:
        transform = transforms.Resize((75, 75))
        new_image = transform(image)
    else:
        new_image = image

    return new_image


def identity(x):
    return x


class InferenceDataset(Dataset):
    def __init__(
        self,
        image_list_csv: str,
        root_dir: str,
        input_size: int,
        sampling_rate: int = 1,
        preprocess_mode: tp.Literal["torch", "tf"] = "torch",
    ):
        """ """

        # Define list of image paths
        df_paths = pd.read_csv(image_list_csv)
        duplicates = df_paths.duplicated().sum()
        if duplicates > 0:
            print(f"Dropping {duplicates} duplicates in the given list of image paths")
            df_paths.drop_duplicates(inplace=True)

        self.imgs = []

        for _, row in df_paths.iterrows():
            self.imgs.append(
                os.path.join(
                    root_dir,
                    row["family"],
                    row["genus"],
                    row["species"],
                    row["filename"],
                )
            )

        self.imgs = self.imgs[::sampling_rate]

        # Define transforms
        if preprocess_mode == "torch":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        elif preprocess_mode == "tf":
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

        self.transforms = transforms.Compose(
            [
                transforms.Lambda(random_resize),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __len__(self):
        # return size of dataset
        return len(self.imgs)

    def __getitem__(self, idx):
        image_name = os.path.basename(self.imgs[idx])
        image = Image.open(self.imgs[idx])
        image = self.transforms(image)

        return image, image_name
