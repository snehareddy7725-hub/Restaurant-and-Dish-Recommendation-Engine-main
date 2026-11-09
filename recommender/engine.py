"""Serving logic with explicit cold-start detection.

Every request is routed to exactly one mode:

  hybrid       user is known to the ALS model AND has a last interaction the
               content model knows   -> 0.61 * ALS + 0.39 * content
  als_only     user is known to ALS but has no usable content seed
  content_only user is NOT in the trained ALS model (e.g. a new user who has
               interacted since the last retrain) but has a usable seed item
  popularity   no usable history at all -> most popular restaurants, optionally
               filtered by a preferred cuisine
"""

import pickle
import threading
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sparse

from . import config
from .training import load_all_interactions

MODES = ("hybrid", "als_only", "content_only", "popularity")


@dataclass
class Recommendation:
    user_id: int
    mode: str
    items: list = field(default_factory=list)


class Recommender:
    def __init__(self):
        self._lock = threading.RLock()
        self.reload()

    # ------------------------------------------------------------------ loading
    def reload(self):
        """(Re)load models and data. Safe to call while serving requests."""
        with open(config.ALS_MODEL_PATH, "rb") as f:
            als = pickle.load(f)
        with open(config.CONTENT_MODEL_PATH, "rb") as f:
            content = pickle.load(f)
        restaurants = pd.read_csv(config.RESTAURANTS_PATH)
        interactions = load_all_interactions()

        with self._lock:
            self.als_model = als["model"]
            self.user_mapper = als["user_mapper"]
            self.item_mapper = als["item_mapper"]
            self.item_inv_mapper = als["item_inv_mapper"]
            self.sim = content["similarity_matrix"]
            self.id_to_index = content["id_to_index"]
            self.index_to_id = content.get("index_to_id") or {i: r for r, i in self.id_to_index.items()}
            self.restaurants = restaurants.set_index("restaurant_id")
            self.interactions = interactions
            self._rebuild_history()

    def _rebuild_history(self):
        df = self.interactions
        self.history = {uid: g["restaurant_id"].tolist() for uid, g in df.groupby("user_id", sort=False)}
        # Popularity = total interaction strength per restaurant.
        self.popularity = df.groupby("restaurant_id")["strength"].sum().sort_values(ascending=False)

    def add_interaction(self, user_id: int, restaurant_id: int, strength: int):
        """Make a freshly logged interaction visible immediately (before retraining)."""
        with self._lock:
            self.history.setdefault(user_id, []).append(restaurant_id)
            self.popularity.loc[restaurant_id] = self.popularity.get(restaurant_id, 0) + strength
            self.popularity = self.popularity.sort_values(ascending=False)

    # ------------------------------------------------------------- mode choice
    def detect_mode(self, user_id: int) -> str:
        hist = self.history.get(user_id, [])
        known_to_als = user_id in self.user_mapper
        has_seed = bool(hist) and hist[-1] in self.id_to_index
        if known_to_als and has_seed:
            return "hybrid"
        if known_to_als:
            return "als_only"
        if has_seed:
            return "content_only"
        return "popularity"

    # ------------------------------------------------------------ components
    def _user_row(self, user_id):
        row = np.zeros(len(self.item_mapper), dtype=np.float32)
        for rid in self.history.get(user_id, []):
            if rid in self.item_mapper:
                row[self.item_mapper[rid]] += 1
        return sparse.csr_matrix(row)

    def als_scores(self, user_id: int, n: int) -> dict:
        """ALS scores scaled to 0-1 by dividing by the top score."""
        uidx = self.user_mapper[user_id]
        ids, scores = self.als_model.recommend(
            uidx, self._user_row(user_id), N=n, filter_already_liked_items=True
        )
        raw = {self.item_inv_mapper[i]: float(s) for i, s in zip(ids, scores) if i in self.item_inv_mapper}
        top = max(raw.values()) if raw else 1.0
        top = top if top > 0 else 1.0
        return {rid: s / top for rid, s in raw.items()}

    def content_scores(self, seed_id: int, n: int) -> dict:
        """Cosine similarity to the seed restaurant (seed itself excluded)."""
        idx = self.id_to_index[seed_id]
        ranked = sorted(enumerate(self.sim[idx]), key=lambda x: -x[1])
        out = {}
        for i, s in ranked:
            rid = self.index_to_id[i]
            if rid == seed_id:
                continue
            out[rid] = float(s)
            if len(out) == n:
                break
        return out

    def popularity_items(self, user_id: int, n: int, cuisine: str | None = None) -> list:
        seen = set(self.history.get(user_id, []))
        ranked = [r for r in self.popularity.index if r not in seen and r in self.restaurants.index]
        if cuisine:
            match = [r for r in ranked if cuisine.lower() in str(self.restaurants.loc[r, "cuisines"]).lower()]
            ranked = match + [r for r in ranked if r not in match]  # preferred cuisine first
        top = ranked[:n]
        max_pop = float(self.popularity.iloc[0]) or 1.0
        return [(rid, float(self.popularity[rid]) / max_pop) for rid in top]

    # ------------------------------------------------------------- public API
    def recommend(self, user_id: int, n: int = 10, cuisine: str | None = None) -> Recommendation:
        with self._lock:
            mode = self.detect_mode(user_id)
            k = max(config.CANDIDATES_PER_MODEL, n)
            hist = self.history.get(user_id, [])

            if mode == "hybrid":
                als = self.als_scores(user_id, k)
                cont = self.content_scores(hist[-1], k)
                scored = {
                    rid: config.ALS_WEIGHT * als.get(rid, 0.0) + config.CONTENT_WEIGHT * cont.get(rid, 0.0)
                    for rid in set(als) | set(cont)
                }
                ranked = sorted(scored.items(), key=lambda x: -x[1])[:n]
                items = [self._item(rid, s, als.get(rid, 0.0), cont.get(rid, 0.0)) for rid, s in ranked]
            elif mode == "als_only":
                ranked = sorted(self.als_scores(user_id, k).items(), key=lambda x: -x[1])[:n]
                items = [self._item(rid, s, s, 0.0) for rid, s in ranked]
            elif mode == "content_only":
                ranked = sorted(self.content_scores(hist[-1], k).items(), key=lambda x: -x[1])[:n]
                items = [self._item(rid, s, 0.0, s) for rid, s in ranked]
            else:
                items = [self._item(rid, s, 0.0, 0.0) for rid, s in self.popularity_items(user_id, n, cuisine)]

            return Recommendation(user_id=user_id, mode=mode, items=items)

    def _item(self, rid, score, als_part, content_part):
        info = self.restaurants.loc[rid]
        return {
            "restaurant_id": int(rid),
            "name": str(info["name"]),
            "cuisines": str(info["cuisines"]),
            "avg_cost_for_two": int(info["avg_cost_for_two"]),
            "rating": float(info["aggregate_rating"]),
            "score": round(float(score), 4),
            "als_score": round(float(als_part), 4),
            "content_score": round(float(content_part), 4),
        }
