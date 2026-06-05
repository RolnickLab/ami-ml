"""Central configuration for the geo-prior pipeline.

Every path, cloud identifier, and secret is read from environment variables,
loaded from ``src/geoprior/.env`` if present (see ``.env.sample`` for the full
list). Nothing machine-specific or secret is hardcoded in the scripts.

Non-secret identifiers (BigQuery project/dataset, W&B entity/project) carry
defaults that match the current deployment but are overridable via the
environment. Secrets (``WANDB_API_KEY``, ``GOOGLE_APPLICATION_CREDENTIALS``)
have NO defaults — they must come from the environment / ``.env``.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is in requirements.txt; degrade gracefully
    def load_dotenv(*_args, **_kwargs):
        return False

PKG_DIR = Path(__file__).resolve().parent          # .../src/geoprior
REPO_ROOT = PKG_DIR.parents[1]                      # repo root (…/ami-ml)

# Load src/geoprior/.env if present (no-op otherwise). Real env vars take
# precedence over .env values.
load_dotenv(PKG_DIR / ".env")


def _path(var: str, default) -> Path:
    return Path(os.environ.get(var, str(default))).expanduser()


# --- BigQuery (source occurrence data) ---
BQ_PROJECT = os.environ.get("GEOPRIOR_BQ_PROJECT", "leps-ai")
BQ_DATASET = os.environ.get("GEOPRIOR_BQ_DATASET", "global_butterflies_2604")
TBL_OCCURRENCES = f"{BQ_PROJECT}.{BQ_DATASET}.gbif_inat_occurrences"
TBL_LOCATION = f"{BQ_PROJECT}.{BQ_DATASET}.gbif_occurrence_location"

# --- Filesystem ---
# Frozen category map — defaults to the committed in-repo artifact.
CATEG_MAP_PATH = _path("GEOPRIOR_CATEG_MAP", PKG_DIR / "geoprior_categ_map.json")
# Working dir for generated train/val/test.json + sibling artifacts.
DATA_DIR = _path("GEOPRIOR_DATA_DIR", REPO_ROOT / "data" / "geoprior")
# Vision split CSVs providing the val/test gbif_id hold-out lists.
SPLITS_DIR = _path("GEOPRIOR_SPLITS_DIR", REPO_ROOT / "data" / "splits")
# Where trained checkpoints are written.
MODEL_DIR = _path("GEOPRIOR_MODEL_DIR", REPO_ROOT / "models" / "geoprior")

# --- Weights & Biases (secret WANDB_API_KEY is read from the environment) ---
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "Global-Butterfly")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "moth-ai")

# --- Fusion eval (downstream) ---
CLF_VAL_PREDS = _path("GEOPRIOR_CLF_VAL_PREDS", DATA_DIR / "clf_val_predictions.csv")
CLF_TEST_PREDS = _path("GEOPRIOR_CLF_TEST_PREDS", DATA_DIR / "clf_test_predictions.csv")
GEOPRIOR_VAL_PREDS = _path("GEOPRIOR_VAL_PREDS", DATA_DIR / "preds" / "val")
GEOPRIOR_TEST_PREDS = _path("GEOPRIOR_TEST_PREDS", DATA_DIR / "preds" / "test")
