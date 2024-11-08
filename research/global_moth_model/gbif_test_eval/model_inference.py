import json

import PIL
import timm
import torch
from constants import AVAILABLE_MODELS, VIT_B16_128
from torchvision import transforms


class ModelInference:
    """Model inference class definition"""

    def __init__(
        self,
        model_path: str,
        model_type: str,
        category_map_json: str,
        device: str,
        input_size: int = 128,
        topk: int = 10,
    ):
        self.device = device
        self.topk = topk
        self.input_size = input_size
        self.model_type = model_type
        self.image = None
        self.id2categ = self._load_category_map(category_map_json)
        self.model = self._load_model(model_path, num_classes=len(self.id2categ))
        self.model.eval()

    def _load_category_map(self, category_map_json: str):
        with open(category_map_json, "r") as f:
            categories_map = json.load(f)

        id2categ = {categories_map[categ]: categ for categ in categories_map}
        return id2categ

    def _pad_to_square(self):
        """Padding transformation to make the image square"""
        width, height = self.image.size
        if height < width:
            return transforms.Pad(padding=[0, 0, 0, width - height])
        elif height > width:
            return transforms.Pad(padding=[0, 0, height - width, 0])
        else:
            return transforms.Pad(padding=[0, 0, 0, 0])

    def get_transforms(self):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        return transforms.Compose(
            [
                self._pad_to_square(),
                transforms.ToTensor(),
                transforms.Resize((self.input_size, self.input_size), antialias=True),
                transforms.Normalize(mean, std),
            ]
        )

    def _load_model(self, model_path: str, num_classes: int, pretrained: bool = True):
        """Build and load the model"""

        if self.model_type not in AVAILABLE_MODELS:
            raise RuntimeError(f"Model {self.model_type} not implemented")

        model_arguments = {"pretrained": pretrained, "num_classes": num_classes}

        if self.model_type == VIT_B16_128:
            # There is no off-the-shelf ViT model for 128x128 image size,
            # so we use 224x224 model with a custom input image size
            self.model_type = "vit_base_patch16_224_in21k"
            model_arguments["img_size"] = 128

        model = timm.create_model(self.model_type, **model_arguments)

        # Load model weights
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device(self.device))
        )
        # Parallelize inference if multiple GPUs available
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        model = model.to(self.device)
        return model

    def get_category(self, pred: list[int]):
        """Return categ from indices"""
        pred_results = []

        for idx in pred:
            categ = self.id2categ[idx]
            pred_results.append(categ)

        return pred_results

    def predict(self, image: PIL.Image.Image):
        with torch.no_grad():
            # Process the image for prediction
            self.image = image
            transforms = self.get_transforms()
            image = transforms(image)
            image = image.to(self.device)
            image = image.unsqueeze_(0)

            # Model prediction on the image
            predictions = self.model(image)
            predictions = torch.nn.functional.softmax(predictions, dim=1)
            predictions = predictions.cpu()
            if self.topk == 0 or self.topk > len(
                predictions[0]
            ):  # topk=0 means get all predictions
                predictions = torch.topk(predictions, len(predictions[0]))
            else:
                predictions = torch.topk(predictions, self.topk)

            # Process the results
            values, indices = (
                predictions.values.numpy()[0],
                predictions.indices.numpy()[0],
            )
            pred_results = []

            for i in range(len(indices)):
                idx, value = indices[i], values[i]
                categ = self.id2categ[idx]
                pred_results.append([categ, value])

            return pred_results
