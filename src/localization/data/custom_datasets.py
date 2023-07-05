"""
"""
import glob
import json
import os
import typing as tp

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, Subset

torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T
from torchvision import datapoints


class TrainingDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: tp.Optional[T.Transform] = T.Compose(
            [T.ToImageTensor(), T.ConvertImageDtype()]
        ),
        remove_empty_images=True,
    ):
        """The root_dir is expected to contain:
            - the images, in jpg format
            - one json file, containing the annotations
        The json file is expected to contain, for each image key, a list with:
            - a list of bounding boxes in the [x1, y1, x2, y2] format,
            with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
            - a list of labels
        Images that are not mentioned in the json file are ignored.
        """

        self.root_dir = root_dir
        self.transform = transform
        os.chdir(root_dir)
        json_files = glob.glob("*.json")
        if len(json_files) != 1:
            raise Exception(
                f"root_dir must contain a single json file. It contains {len(json_files)}."
            )
        with open(os.path.join(root_dir, json_files[0])) as f:
            self.annot_data = json.load(f)
        self.imgs = sorted(list(self.annot_data.keys()))

        if remove_empty_images:
            self.remove_empty_images()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        """Boxes which contains at least one side smaller than 10 px are filtered out."""
        image_name = self.imgs[idx]
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path)

        bboxes = torch.FloatTensor(self.annot_data[image_name][0])
        bboxes_idx = torchvision.ops.remove_small_boxes(bboxes, 10)
        labels = torch.LongTensor(self.annot_data[image_name][1])

        target = {}
        target["boxes"] = datapoints.BoundingBox(
            bboxes[bboxes_idx],
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=image.size[::-1],
        )

        target["labels"] = labels[bboxes_idx]

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def remove_empty_images(self):
        """Remove images with no bounding boxes from the dataset"""
        filtered_data = {}
        for key, data in self.annot_data.items():
            if len(data[0][0]) > 0:
                filtered_data[key] = data
        nb_filtered_images = len(self.annot_data) - len(filtered_data)
        if nb_filtered_images > 0:
            print(f"Warning: {nb_filtered_images} images filtered out (no bboxes)")
        self.annot_data = filtered_data
        self.imgs = sorted(list(self.annot_data.keys()))


class InferenceDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: tp.Optional[T.Transform] = T.Compose(
            [T.ToImageTensor(), T.ConvertImageDtype()]
        ),
        sampling_rate: int = 1,
    ):
        """The root_dir is expected to contain the images in jpg format. Any other file
        will be ignored.
        """

        self.root_dir = root_dir
        self.transform = transform
        os.chdir(root_dir)
        self.imgs = sorted(glob.glob("*.jpg"))
        self.imgs = self.imgs[::sampling_rate]

    def __len__(self):
        # return size of dataset
        return len(self.imgs)

    def __getitem__(self, idx):
        image_name = self.imgs[idx]
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, image_name


class SplitDataset(Dataset):
    """This class is meant to be used on the subsets obtained after the random_split()
    call, in order to have different transforms on the training and validation sets.
    These will work on top of any transform that was given to the dataset before the
    random_split() call.
    """

    def __init__(
        self,
        dataset: Subset,
        transform=T.Compose([T.ToImageTensor(), T.ConvertImageDtype()]),
    ):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


class DatasetAsNumpyArrays(Dataset):
    def __init__(
        self,
        root_dir: str,
        sampling_rate: int = 1,
    ):
        """The root_dir is expected to contain the images in jpg format. Any other file
        will be ignored. root_dir can also be the path to a single jpg file.
        """
        if os.path.isdir(root_dir):
            self.root_dir = root_dir
            os.chdir(root_dir)
            self.imgs = sorted(glob.glob("*.jpg"))
            self.imgs = self.imgs[::sampling_rate]
        elif os.path.isfile(root_dir) and os.path.splitext(root_dir)[1].lower() in [
            ".jpg",
            ".jpeg",
        ]:
            self.root_dir = os.path.dirname(root_dir)
            self.imgs = [os.path.basename(root_dir)]
        else:
            raise ValueError("Given string is neither dir nor file")

    def __len__(self):
        # return size of dataset
        return len(self.imgs)

    def __getitem__(self, idx):
        image_name = self.imgs[idx]
        image_path = os.path.join(self.root_dir, image_name)
        image = np.asarray(Image.open(image_path))

        return image, image_name
