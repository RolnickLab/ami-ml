import numpy as np
import torch
from torchvision import transforms
import json
import PIL
import timm
from models import model_builder

class ModelInference:
    """Model inference class definition"""

    def __init__(
        self,
        model_path: str,   
        model_type: str,     
        category_map_json: str,
        device: str,        
        input_size: int = 128,
        topk: int = 10
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
        height, width = self.image.shape[1], self.image.shape[2]
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
                transforms.Resize((self.input_size, self.input_size)),
                transforms.Normalize(mean, std),
            ]
        )

    def _load_model(self, model_path: str, num_classes: int, pretrained: bool = True):
        if self.model_type == "resnet50":
            model = timm.create_model(
            "resnet50", pretrained=pretrained, num_classes=num_classes
        )

        elif self.model_type == "timm_convnext-t":
            model = timm.create_model(
            "convnext_tiny_in22k", pretrained=pretrained, num_classes=num_classes
        )
            
        elif self.model_type == "timm_convnext-b":
            model = timm.create_model(
            "convnext_base_in22k", pretrained=pretrained, num_classes=num_classes
        )

        elif self.model_type == "efficientnetv2-b3":
            model = timm.create_model(
            "tf_efficientnetv2_b3", pretrained=pretrained, num_classes=num_classes
        )
            
        elif self.model_type == "timm_mobilenetv3large":
            model = timm.create_model(
            "mobilenetv3_large_100", pretrained=pretrained, num_classes=num_classes
        )
            
        elif self.model_type == "timm_vit-b16-128":
            model = timm.create_model(
            "vit_base_patch16_224_in21k",
            pretrained=pretrained,
            img_size=128,
            num_classes=num_classes,
        )

        else:
            raise RuntimeError(f"Model {self.model_type} not implemented")

        # Load model weights
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device(self.device))
        )
        # Parallelize inference if multiple GPUs available
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        model = model.to(self.device)
        return model

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
            if self.topk == 0 or self.topk > len(predictions[0]): # topk=0 means get all predictions
                predictions = torch.topk(predictions, len(predictions[0]))
            else:
                predictions = torch.topk(predictions, self.topk)

            # Process the results
            values, indices = predictions.values.numpy()[0], predictions.indices.numpy()[0]
            pred_results = []
            
            for i in range(len(indices)):
                idx, value = indices[i], values[i]
                categ = self.id2categ[idx]
                pred_results.append([categ, value])

            return pred_results
