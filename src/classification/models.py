""" List of available models to train
"""

import timm
import torch
from torchvision import models


def model_list(model_name: str, num_classes: int, pretrained: bool):
    """Main model builder function"""

    if model_name == "efficientnetv2-b3":
        model = timm.create_model(
            "tf_efficientnetv2_b3", pretrained=pretrained, num_classes=num_classes
        )
    elif model_name == "efficientnetv2-s-in21k":
        model = timm.create_model(
            "tf_efficientnetv2_s_in21k", pretrained=pretrained, num_classes=num_classes
        )
    elif model_name == "swin-s":
        model = timm.create_model(
            "swin_small_patch4_window7_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
    elif model_name == "timm_mobilenetv3large":
        model = timm.create_model(
            "mobilenetv3_large_100", pretrained=pretrained, num_classes=num_classes
        )
    elif model_name == "timm_resnet50":
        model = timm.create_model(
            "resnet50", pretrained=pretrained, num_classes=num_classes
        )
    elif model_name == "timm_convnext-t":
        model = timm.create_model(
            "convnext_tiny_in22k", pretrained=pretrained, num_classes=num_classes
        )
    elif model_name == "timm_convnext-b":
        model = timm.create_model(
            "convnext_base_in22k", pretrained=pretrained, num_classes=num_classes
        )
    elif model_name == "timm_vit-b16-128":
        model = timm.create_model(
            "vit_base_patch16_224_in21k",
            pretrained=pretrained,
            img_size=128,
            num_classes=num_classes,
        )
    elif model_name == "timm_vit-b16-224":
        model = timm.create_model(
            "vit_base_patch16_224_in21k",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif model_name == "timm_vit-b16-384":
        model = timm.create_model(
            "vit_base_patch16_384",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    else:
        raise RuntimeError(f"Model {model_name} not implemented")

    return model
