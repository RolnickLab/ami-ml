"""
Note: these functions are from Aditya's mothAI/gbif...
"""
from torchvision import transforms
import torch
import webdataset as wds
import numpy as np
import torchvision.models as models
from torch import nn


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


def get_transforms(
    input_size=300, set_type=None, preprocess_mode="torch", test_set_num=None
):
    """transform to be applied to each image"""

    if preprocess_mode == "torch":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif preprocess_mode == "tf":
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    else:
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    low_res1 = int(input_size / 4)
    low_res2 = int(input_size / 2)
    train_resize = transforms.Lambda(random_resize)
    val_resize = transforms.Lambda(random_resize)

    if test_set_num == 1:
        test_resize = transforms.RandomApply(
            torch.nn.ModuleList([transforms.Resize((low_res1, low_res1))]), p=1
        )
    elif test_set_num == 2:
        test_resize = transforms.RandomApply(
            torch.nn.ModuleList([transforms.Resize((low_res2, low_res2))]), p=1
        )
    elif test_set_num == 3:
        # do not apply this transform
        test_resize = transforms.RandomApply(
            torch.nn.ModuleList([transforms.Resize((input_size, input_size))]), p=1
        )
    else:
        test_resize = transforms.Lambda(random_resize)

    if set_type == "train":
        return transforms.Compose(
            [
                train_resize,
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=4),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif set_type == "validation":
        return transforms.Compose(
            [
                val_resize,
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        return transforms.Compose(
            [
                test_resize,
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


def identity(x):
    return x


def build_webdataset_pipeline(
    sharedurl: str,
    input_size: int,
    batch_size: int,
    set_type: str,
    num_workers: int,
    preprocess_mode: str,
    test_set_num=None,
):
    """main dataset building function"""

    transform = get_transforms(input_size, set_type, preprocess_mode, test_set_num)
    if set_type == "train":
        dataset = wds.WebDataset(sharedurl, shardshuffle=True)
    else:
        dataset = wds.WebDataset(sharedurl, shardshuffle=False)

    if set_type == "train":
        dataset = dataset.shuffle(10000)

    dataset = (
        dataset.decode("pil").to_tuple("jpg", "cls").map_tuple(transform, identity)
    )

    loader = torch.utils.data.DataLoader(
        dataset, num_workers=num_workers, batch_size=batch_size
    )

    return loader


class Resnet50(nn.Module):
    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: number of species classes
        """
        super(Resnet50, self).__init__()
        self.num_classes = num_classes
        self.backbone = models.resnet50(pretrained=True)
        out_dim = self.backbone.fc.in_features

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(out_dim, self.num_classes, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
