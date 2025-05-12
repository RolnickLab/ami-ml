"""Model inference class definition"""

import json
import os
import pathlib

import dotenv
import PIL
import timm
import torch
from PIL import Image
from torch.nn.functional import softmax
from torchvision import transforms

from src.classification.constants import AVAILABLE_MODELS, VIT_B16_128

dotenv.load_dotenv()

CLASSIFICATION_ROOT = pathlib.Path(__file__).resolve().parent
SOURCE_ROOT = CLASSIFICATION_ROOT.parent
PROJECT_ROOT = SOURCE_ROOT.parent
ASSETS_PATH = PROJECT_ROOT / "assets"


class ModelInference:
    """Model inference class definition"""

    def __init__(
        self,
        model_path: str,
        model_type: str,
        taxon_key_to_id_json: str,
        taxon_key_to_name_json: str,
        device: str,
        preprocess_mode: str = "torch",
        input_size: int = 128,
        topk: int = 5,  # set -1 to get all predictions
        checkpoint: bool = False,
        class_masking_list: list | None = None,
    ):
        self.device = device
        self.topk = topk
        self.checkpoint = checkpoint
        self.input_size = input_size
        self.model_type = model_type
        self.preprocess_mode = preprocess_mode
        self.image = None
        self.id_to_taxon_key = self._load_id_to_taxon_key_map(taxon_key_to_id_json)
        self.taxon_key_to_name = self._load_taxon_key_to_name_map(
            taxon_key_to_name_json
        )
        self.name_to_id_map = self._build_name_to_id_map()
        self.model = self._load_model(model_path, num_classes=len(self.id_to_taxon_key))
        self.model.eval()
        self.class_masking_list = class_masking_list

    def _load_id_to_taxon_key_map(self, taxon_key_to_id_json: str):
        """Load the mapping from category id to taxon key"""

        with open(taxon_key_to_id_json, "r", encoding="utf-8") as f:
            categories_map = json.load(f)

        id2categ = {categories_map[categ]: categ for categ in categories_map}
        return id2categ

    def _load_taxon_key_to_name_map(self, taxon_key_to_name_json: str):
        """Load the mapping from category taxon key to species name"""

        with open(taxon_key_to_name_json, "r", encoding="utf-8") as f:
            categ_to_name_map = json.load(f)

        return categ_to_name_map

    def _build_name_to_id_map(self):
        """Build mapping from taxon name to category id"""

        name2key = {self.taxon_key_to_name[key]: key for key in self.taxon_key_to_name}
        key2id = {self.id_to_taxon_key[id]: id for id in self.id_to_taxon_key}
        name2id = {name: key2id[name2key[name]] for name in name2key}

        return name2id

    def _pad_to_square(self):
        """Padding transformation to make the image square"""

        width, height = self.image.size
        if height < width:
            return transforms.Pad(padding=[0, 0, 0, width - height])
        elif height > width:
            return transforms.Pad(padding=[0, 0, height - width, 0])
        else:
            return transforms.Pad(padding=[0, 0, 0, 0])

    def _get_transforms(self):
        """List of transformations to apply to the image"""

        preprocess_params = {
            "torch": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            "tf": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            "default": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        }

        mean, std = preprocess_params.get(
            self.preprocess_mode, preprocess_params["default"]
        )

        return transforms.Compose(
            [
                self._pad_to_square(),
                transforms.Resize((self.input_size, self.input_size), antialias=True),
                transforms.ToTensor(),
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
        model_weights = torch.load(model_path, map_location=torch.device(self.device))
        if self.checkpoint:
            model.load_state_dict(model_weights["model_state_dict"])
        else:
            model.load_state_dict(model_weights)

        # Parallelize inference if multiple GPUs available
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        model = model.to(self.device)
        return model

    def _mask_classes(self, predictions: torch.Tensor):
        """Class mask function to include specific output classes and exclude the rest"""

        # Create a mask for the classes to prune
        mask = torch.zeros(
            predictions.size(1), dtype=torch.bool, device=predictions.device
        )

        # Get species keys that needs to be removed
        for taxon_to_keep in self.class_masking_list:
            id_to_keep = self.name_to_id_map[taxon_to_keep]
            mask[id_to_keep] = True

        # Apply the mask to zero out unwanted nodes
        predictions[:, ~mask] = float("-inf")  # Set to -inf to ignore during softmax
        return predictions

    @torch.no_grad()
    def predict(self, image: PIL.Image.Image):
        """Main function for predicting on image"""

        # Process the image for prediction
        self.image = image
        image_transform = self._get_transforms()
        image = image_transform(image).to(self.device).unsqueeze_(0)

        # Model prediction on the image
        predictions = self.model(image)

        # Mask classes, if requested
        if self.class_masking_list:
            predictions = self._mask_classes(predictions)

        predictions = softmax(predictions, dim=1).cpu()

        # First, get all predictions in decreasing order of confidence
        topk_predictions = torch.topk(predictions, len(predictions[0]))

        # Get only top k predictions
        if self.topk > 0:
            topk_predictions = torch.topk(predictions, self.topk)

        # Process the results
        values, indices = (
            topk_predictions.values.numpy()[0],
            topk_predictions.indices.numpy()[0],
        )
        prediction_results = []

        for _, (idx, confidence) in enumerate(zip(indices, values)):
            taxon_key = self.id_to_taxon_key[idx]
            species_name = self.taxon_key_to_name[taxon_key]
            prediction_results.append([species_name, round(confidence * 100, 2)])

        return prediction_results


if __name__ == "__main__":
    # Load the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the test image
    test_image = Image.open(ASSETS_PATH / "test_luna_moth.jpg")

    # Build the model inference class
    model_path = os.environ.get("GLOBAL_MODEL")
    taxon_key_to_id_map = str(ASSETS_PATH / "example_taxon_key_to_id_map.json")
    taxon_key_to_name_map = str(ASSETS_PATH / "example_taxon_key_to_name_map.json")
    moth_classifier = ModelInference(
        model_path, "resnet50", taxon_key_to_id_map, taxon_key_to_name_map, device
    )

    #  Predict on image
    species_prediction = moth_classifier.predict(test_image)
    print(f"The model prediction results are: {species_prediction}")
