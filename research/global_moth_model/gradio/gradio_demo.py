import os

import gradio as gr
import PIL
import torch
from dotenv import load_dotenv
from model_inference import ModelInference

# Load secrets and config from optional .env file
load_dotenv()
GLOBAL_MODEL = os.getenv("GLOBAL_MODEL", default="global_model.pt")
CATEGORY_MAP = os.getenv("CATEGORY_MAP_JSON", default="category_map.json")
CATEG_TO_NAME_MAP = os.getenv("CATEG_TO_NAME_MAP", default="categ_to_name_map.json")


# Model prediction function
def predict_species(image: PIL.Image.Image) -> dict[str, float]:
    """Moth species prediction"""

    # Build the model class
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fgrained_classifier = ModelInference(
        GLOBAL_MODEL, "timm_resnet50", CATEGORY_MAP, CATEG_TO_NAME_MAP, device, topk=5
    )

    # Predict on image
    sp_pred = fgrained_classifier.predict(image)

    return sp_pred


demo = gr.Interface(
    fn=predict_species,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Mila Global Moth Species Classifier",
)

if __name__ == "__main__":
    demo.launch(share=True)
