import pandas as pd
import pytest

from recommender import config
from recommender.engine import Recommender
from recommender.training import train_als


@pytest.fixture(scope="module")
def reco():
    return Recommender()


def test_als_factor_shapes_not_swapped():
    df = pd.read_csv(config.INTERACTIONS_PATH)
    als = train_als(df, iterations=2)
    assert als["model"].user_factors.shape[0] == df["user_id"].nunique()
    assert als["model"].item_factors.shape[0] == df["restaurant_id"].nunique()


def test_known_user_gets_hybrid(reco):
    rec = reco.recommend(0, n=10)
    assert rec.mode == "hybrid"
    assert len(rec.items) == 10
    ids = [i["restaurant_id"] for i in rec.items]
    assert len(ids) == len(set(ids)), "duplicate recommendations"
    scores = [i["score"] for i in rec.items]
    assert scores == sorted(scores, reverse=True)


def test_unknown_user_gets_popularity(reco):
    rec = reco.recommend(10_000_001, n=5)
    assert rec.mode == "popularity"
    assert len(rec.items) == 5


def test_popularity_respects_cuisine(reco):
    rec = reco.recommend(10_000_002, n=3, cuisine="Italian")
    assert all("Italian" in i["cuisines"] for i in rec.items)


def test_new_user_with_interaction_gets_content_only(reco):
    reco.add_interaction(10_000_003, 24, 3)
    rec = reco.recommend(10_000_003, n=5)
    assert rec.mode == "content_only"
    assert 24 not in [i["restaurant_id"] for i in rec.items]


def test_known_user_without_content_seed_gets_als_only(reco):
    uid = 0
    saved = reco.history[uid]
    reco.history[uid] = saved + [999_999]  # last item unknown to content model
    try:
        assert reco.recommend(uid).mode == "als_only"
    finally:
        reco.history[uid] = saved
