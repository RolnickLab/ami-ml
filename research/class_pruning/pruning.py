import os
import pathlib
import pickle

import dotenv
import torch
from PIL import Image

from src.classification.model_inference import ModelInference

dotenv.load_dotenv()

CLASSIFICATION_ROOT = pathlib.Path(__file__).resolve().parent
SOURCE_ROOT = CLASSIFICATION_ROOT.parent
PROJECT_ROOT = SOURCE_ROOT.parent
ASSETS_PATH = PROJECT_ROOT / "assets"

if __name__ == "__main__":
    # Load the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the test image
    test_image = Image.open(ASSETS_PATH / "Arrhenophanes_perspicilla.jpg")

    # Model-builder related parameters
    model_path = os.getenv("CLASS_PRUNING_PANAMA_MODEL", "panama_model.pth")
    taxon_key_to_id_map = os.getenv(
        "CLASS_PRUNING_KEY_TO_ID_MAP", "taxon_key_to_id_map.json"
    )
    taxon_key_to_name_map = os.getenv(
        "CLASS_PRUNING_KEY_TO_NAME_MAP", "taxon_key_to_name_map.json"
    )
    with open(
        os.getenv("CLASS_PRUNING_SPECIES_OF_INTEREST", "pruning_list.pkl"), "rb"
    ) as f:
        class_pruning_list = pickle.load(f)

    # Predict on image
    pruned_classifier = ModelInference(
        model_path,
        "resnet50",
        taxon_key_to_id_map,
        taxon_key_to_name_map,
        device,
        topk=5,
        class_pruning_list=class_pruning_list,
    )
    species_prediction = pruned_classifier.predict(test_image)
    print(f"The species predictions along with confidences are: {species_prediction}")
