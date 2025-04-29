import os
from pathlib import Path

import dotenv
import pandas as pd

from src.dataset_tools.fetch_images import _url_retrieve

dotenv.load_dotenv()

DATA_DIR = os.getenv("CONF_CALIB_GLOBAL_MODEL_DATASET_PATH")
VAL_CSV = os.getenv("CONF_CALIB_GLOBAL_MODEL_VAL_CSV")

data_dir = Path(DATA_DIR)
val_csv = pd.read_csv(VAL_CSV)

# iterate over the rows of the csv file
for index, row in val_csv.iterrows():
    # Create dataset folder if it does not exist
    rel_img_path = Path(row["image_path"])
    dataset_dir = data_dir / rel_img_path.parts[0]
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)

    # Get full image path and url
    img_path = data_dir / row["image_path"]
    img_url = row["identifier"]

    # Check if the image is missing and download the image
    if not os.path.exists(img_path):
        print(f"Downloading {img_url}.", flush=True)

        try:
            _url_retrieve(img_url, img_path, 10)
        except Exception as e:
            print(f"Error downloading {img_url}", flush=True)
            print(e)
