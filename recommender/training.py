"""Model training: ALS (collaborative filtering) and the cosine-similarity
content model. Used by train_models.py, the API retraining trigger and tests.

Note on a past bug: an earlier ALS model had user_factors of shape (100, 20)
and item_factors of shape (1000, 20) -- swapped with the real counts
(1000 users, 100 restaurants) because the matrix passed to .fit() was
(items x users). The matrix is now built explicitly as (users x items) and
the factor shapes are asserted before anything is saved.
"""

import json
import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from implicit.cpu.als import AlternatingLeastSquares
from sklearn.metrics.pairwise import cosine_similarity

from . import config


def build_user_item_matrix(interactions: pd.DataFrame, user_mapper: dict, item_mapper: dict):
    """Sparse (users x items) matrix of summed interaction strengths."""
    rows = interactions["user_id"].map(user_mapper).values
    cols = interactions["restaurant_id"].map(item_mapper).values
    vals = interactions["strength"].values.astype(np.float32)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(user_mapper), len(item_mapper)))


def train_als(interactions: pd.DataFrame, items: list | None = None, **params) -> dict:
    unique_users = interactions["user_id"].unique()
    unique_items = interactions["restaurant_id"].unique() if items is None else np.asarray(items)

    user_mapper = {uid: idx for idx, uid in enumerate(unique_users)}
    item_mapper = {iid: idx for idx, iid in enumerate(unique_items)}
    interactions = interactions[interactions["restaurant_id"].isin(item_mapper)]
    matrix = build_user_item_matrix(interactions, user_mapper, item_mapper)

    model = AlternatingLeastSquares(
        factors=params.get("factors", config.ALS_FACTORS),
        regularization=params.get("regularization", config.ALS_REGULARIZATION),
        iterations=params.get("iterations", config.ALS_ITERATIONS),
        alpha=params.get("alpha", config.ALS_ALPHA),
        random_state=params.get("random_state", config.RANDOM_SEED),
    )
    model.fit(matrix, show_progress=False)

    # Guard against the swapped-dimension bug before anything is saved.
    assert model.user_factors.shape[0] == len(user_mapper), (
        f"user_factors has {model.user_factors.shape[0]} rows, expected {len(user_mapper)}"
    )
    assert model.item_factors.shape[0] == len(item_mapper), (
        f"item_factors has {model.item_factors.shape[0]} rows, expected {len(item_mapper)}"
    )

    return {
        "model": model,
        "user_mapper": user_mapper,
        "item_mapper": item_mapper,
        "user_inv_mapper": {i: u for u, i in user_mapper.items()},
        "item_inv_mapper": {i: r for r, i in item_mapper.items()},
    }


def train_content_model(restaurants: pd.DataFrame) -> dict:
    restaurants = restaurants.reset_index(drop=True)
    cuisine_dummies = pd.get_dummies(restaurants["cuisines"]).astype(float)
    price = restaurants["avg_cost_for_two"]
    price_norm = (price - price.min()) / (price.max() - price.min())
    rating_norm = restaurants["aggregate_rating"] / 5.0
    features = pd.concat(
        [cuisine_dummies, price_norm.rename("price_norm"), rating_norm.rename("rating_norm")], axis=1
    ).values
    return {
        "similarity_matrix": cosine_similarity(features),
        "id_to_index": {rid: i for i, rid in enumerate(restaurants["restaurant_id"])},
        "index_to_id": {i: rid for i, rid in enumerate(restaurants["restaurant_id"])},
    }


def load_all_interactions() -> pd.DataFrame:
    """Original training data plus anything logged through the API."""
    frames = [pd.read_csv(config.INTERACTIONS_PATH)]
    if config.NEW_INTERACTIONS_PATH.exists():
        new = pd.read_csv(config.NEW_INTERACTIONS_PATH)
        if not new.empty:
            frames.append(new[["user_id", "restaurant_id", "interaction_type", "strength"]])
    return pd.concat(frames, ignore_index=True)


def _atomic_pickle(obj, path: Path):
    """Write to a temp file then rename, so a running API never reads half a file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def retrain_and_save(merge_new_interactions: bool = True) -> dict:
    """Retrain both models on all interactions and save them.

    If merge_new_interactions is True, API-logged interactions are folded into
    interactions.csv and the new-interactions log is cleared.
    """
    interactions = load_all_interactions()
    restaurants = pd.read_csv(config.RESTAURANTS_PATH)

    als_data = train_als(interactions)
    content_data = train_content_model(restaurants)
    _atomic_pickle(als_data, config.ALS_MODEL_PATH)
    _atomic_pickle(content_data, config.CONTENT_MODEL_PATH)

    if merge_new_interactions and config.NEW_INTERACTIONS_PATH.exists():
        interactions.to_csv(config.INTERACTIONS_PATH, index=False)
        config.NEW_INTERACTIONS_PATH.unlink()

    state = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_interactions": int(len(interactions)),
        "n_users": int(interactions["user_id"].nunique()),
        "n_items": int(len(content_data["id_to_index"])),
        "user_factors_shape": list(als_data["model"].user_factors.shape),
        "item_factors_shape": list(als_data["model"].item_factors.shape),
    }
    config.TRAINING_STATE_PATH.write_text(json.dumps(state, indent=2))
    return state
