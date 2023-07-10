"""
Author         : Aditya Jain
Last modified  : May 11th, 2022
About          : Description of EfficientNet Model
"""
import timm


class EfficientNet:
    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: number of species classes
        """
        self.num_classes = num_classes

    def get_model(self):
        return timm.create_model(
            "tf_efficientnetv2_b3", pretrained=True, num_classes=self.num_classes
        )
