"""
train_models.py

Trains both models used by app.py:
  1. ALS (Collaborative Filtering) -> data/als_model.pkl
  2. Cosine-similarity content model -> data/content_model.pkl

Run with:
    python train_models.py

Fix vs. the original supplied als_model.pkl:
  The original model's user_factors had shape (100, 20) and item_factors
  had shape (1000, 20) -- i.e. swapped with the real counts (1000 users,
  100 restaurants). That happens when the sparse matrix passed into
  .fit() is (items x users) instead of (users x items).
  This script explicitly builds the matrix as (users x items) and
  verifies the resulting factor shapes before saving, so the bug can't
  silently reappear.
"""

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"
INTERACTIONS_PATH = f"{DATA_DIR}/interactions.csv"
RESTAURANTS_PATH = f"{DATA_DIR}/restaurants.csv"
ALS_OUT_PATH = f"{DATA_DIR}/als_model.pkl"
CONTENT_OUT_PATH = f"{DATA_DIR}/content_model.pkl"


def train_als(interactions: pd.DataFrame) -> dict:
    # --- Build ID <-> index mappings ---
    unique_users = interactions["user_id"].unique()
    unique_items = interactions["restaurant_id"].unique()

    user_mapper = {uid: idx for idx, uid in enumerate(unique_users)}
    item_mapper = {iid: idx for idx, iid in enumerate(unique_items)}
    user_inv_mapper = {idx: uid for uid, idx in user_mapper.items()}
    item_inv_mapper = {idx: iid for iid, idx in item_mapper.items()}

    # --- Build the sparse matrix: rows = users, cols = items ---
    user_indices = interactions["user_id"].map(user_mapper).values
    item_indices = interactions["restaurant_id"].map(item_mapper).values
    confidence = interactions["strength"].values.astype(np.float32)

    sparse_user_item = sparse.csr_matrix(
        (confidence, (user_indices, item_indices)),
        shape=(len(unique_users), len(unique_items)),
    )

    # --- Fit ALS on the (users x items) matrix ---
    # regularization=0.001 chosen via grid search on held-out validation
    # users (factors=20, iterations=50, alpha=1.0 confirmed near-optimal;
    # see evaluate_and_tune.py for the full search and Hit Rate@10 results)
    model = AlternatingLeastSquares(
        factors=20,
        regularization=0.001,
        iterations=50,
        alpha=1.0,
        random_state=42,
    )
    model.fit(sparse_user_item)

    # --- Sanity check: catch the swapped-dimension bug before saving ---
    n_users, n_items = len(unique_users), len(unique_items)
    assert model.user_factors.shape[0] == n_users, (
        f"user_factors has {model.user_factors.shape[0]} rows, "
        f"expected {n_users} (one per user). Matrix orientation is wrong."
    )
    assert model.item_factors.shape[0] == n_items, (
        f"item_factors has {model.item_factors.shape[0]} rows, "
        f"expected {n_items} (one per restaurant). Matrix orientation is wrong."
    )
    print(f"[ALS] OK: user_factors {model.user_factors.shape}, "
          f"item_factors {model.item_factors.shape}")

    return {
        "model": model,
        "user_mapper": user_mapper,
        "item_mapper": item_mapper,
        "user_inv_mapper": user_inv_mapper,
        "item_inv_mapper": item_inv_mapper,
    }


def train_content_model(restaurants: pd.DataFrame) -> dict:
    restaurants = restaurants.reset_index(drop=True)  # lock in row order

    # One-hot encode cuisines
    cuisine_dummies = pd.get_dummies(restaurants["cuisines"])

    # Normalize price and rating to comparable 0-1 ranges
    price = restaurants["avg_cost_for_two"]
    price_norm = (price - price.min()) / (price.max() - price.min())

    rating_norm = restaurants["aggregate_rating"] / 5.0

    feature_matrix = pd.concat(
        [cuisine_dummies, price_norm.rename("price_norm"),
         rating_norm.rename("rating_norm")],
        axis=1,
    ).values

    similarity_matrix = cosine_similarity(feature_matrix)

    id_to_index = {
        rid: idx for idx, rid in enumerate(restaurants["restaurant_id"])
    }

    print(f"[Content] OK: similarity_matrix {similarity_matrix.shape}")

    return {
        "similarity_matrix": similarity_matrix,
        "id_to_index": id_to_index,
        "df_meta": restaurants,
    }


def main():
    interactions = pd.read_csv(INTERACTIONS_PATH)
    restaurants = pd.read_csv(RESTAURANTS_PATH)

    als_data = train_als(interactions)
    with open(ALS_OUT_PATH, "wb") as f:
        pickle.dump(als_data, f)
    print(f"Saved {ALS_OUT_PATH}")

    content_data = train_content_model(restaurants)
    with open(CONTENT_OUT_PATH, "wb") as f:
        pickle.dump(content_data, f)
    print(f"Saved {CONTENT_OUT_PATH}")


if __name__ == "__main__":
    main()