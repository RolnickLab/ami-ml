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
    model_path = os.getenv("CLASS_MASKING_PANAMA_MODEL", "panama_model.pth")
    taxon_key_to_id_map = ASSETS_PATH / "class_masking" / "panama_key_to_id_map.json"

    taxon_key_to_name_map = (
        ASSETS_PATH / "class_masking" / "panama_key_to_taxon_map.json"
    )
    with open(ASSETS_PATH / "class_masking" / "november_bci_species.pkl", "rb") as f:
        class_masking_list = pickle.load(f)

    # Predict on image
    masked_classifier = ModelInference(
        model_path,
        "resnet50",
        taxon_key_to_id_map,
        taxon_key_to_name_map,
        device,
        topk=5,
        class_masking_list=class_masking_list,
    )
    species_prediction = masked_classifier.predict(test_image)
    print(f"The species predictions along with confidences are: {species_prediction}")
