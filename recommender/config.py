"""Central configuration. Every value can be overridden with an environment
variable so the same code runs locally, in tests and inside Docker."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("RECO_DATA_DIR", ROOT_DIR / "data"))
LOG_DIR = Path(os.getenv("RECO_LOG_DIR", ROOT_DIR / "logs"))

INTERACTIONS_PATH = DATA_DIR / "interactions.csv"          # training data
NEW_INTERACTIONS_PATH = DATA_DIR / "new_interactions.csv"  # logged via the API since last retrain
RESTAURANTS_PATH = DATA_DIR / "restaurants.csv"
USER_METADATA_PATH = DATA_DIR / "user_metadata.csv"
ALS_MODEL_PATH = DATA_DIR / "als_model.pkl"
CONTENT_MODEL_PATH = DATA_DIR / "content_model.pkl"
TRAINING_STATE_PATH = DATA_DIR / "training_state.json"

# --- ALS hyperparameters (chosen by grid search in evaluate_and_tune.py) ---
ALS_FACTORS = 20
ALS_REGULARIZATION = 0.001
ALS_ITERATIONS = 50
ALS_ALPHA = 1.0
RANDOM_SEED = 42

# --- Hybrid blend (chosen by weight search in evaluate_and_tune.py) ---
ALS_WEIGHT = 0.61
CONTENT_WEIGHT = 0.39
CANDIDATES_PER_MODEL = 20

# --- Implicit-feedback strength per interaction type ---
INTERACTION_STRENGTH = {"view": 1, "cart": 2, "order": 3}

# --- Retraining trigger: retrain after this many new interactions ---
RETRAIN_THRESHOLD = int(os.getenv("RECO_RETRAIN_THRESHOLD", "500"))
AUTO_RETRAIN = os.getenv("RECO_AUTO_RETRAIN", "true").lower() == "true"
