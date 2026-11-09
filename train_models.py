"""
Train both models used by the API and the Streamlit app:
  1. ALS (collaborative filtering)    -> data/als_model.pkl
  2. Cosine-similarity content model  -> data/content_model.pkl

Run:  python train_models.py

Any interactions logged through the API (data/new_interactions.csv) are
merged into data/interactions.csv before training.
"""

from recommender.training import retrain_and_save

if __name__ == "__main__":
    state = retrain_and_save(merge_new_interactions=True)
    print(f"[ALS] user_factors {tuple(state['user_factors_shape'])}, "
          f"item_factors {tuple(state['item_factors_shape'])}")
    print(f"Trained on {state['n_interactions']} interactions "
          f"({state['n_users']} users, {state['n_items']} restaurants)")
    print("Saved data/als_model.pkl and data/content_model.pkl")
