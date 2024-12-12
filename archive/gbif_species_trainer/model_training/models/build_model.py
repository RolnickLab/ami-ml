"""
Author         : Aditya Jain
Last modified  : May 11th, 2023
About          : Main function for building a model
"""
from models.resnet50 import Resnet50
from models.efficientnet import EfficientNet


def build_model(num_classes: int, model_type: str):
    if model_type == "resnet50":
        return Resnet50(num_classes)
    elif model_type == "efficientnetv2-b3":
        return EfficientNet(num_classes).get_model()
    else:
        raise RuntimeError(f"Model {model_type} not implemented.")
