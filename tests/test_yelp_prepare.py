"""Checks the Yelp converter on a tiny file in the real Yelp JSON format."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _write_fake_yelp(d: Path):
    businesses = [
        {"business_id": f"b{i}", "name": f"Place {i}", "city": "Philadelphia", "stars": 4.0,
         "review_count": 50, "categories": ("Italian, Restaurants" if i % 2 else "Mexican, Restaurants"),
         "attributes": {"RestaurantsPriceRange2": str(1 + i % 4)}}
        for i in range(6)
    ] + [{"business_id": "salon", "name": "Salon", "city": "Philadelphia", "stars": 5.0,
          "review_count": 99, "categories": "Hair Salons", "attributes": None}]
    (d / "yelp_academic_dataset_business.json").write_text("\n".join(json.dumps(b) for b in businesses))

    reviews = []
    for u in range(8):
        for j, b in enumerate([0, 1, 2, 3, 4, 5, "salon"]):
            bid = b if b == "salon" else f"b{b}"
            # dates deliberately written out of order
            reviews.append({"user_id": f"u{u}", "business_id": bid, "stars": 1 + (u + j) % 5,
                            "date": f"2020-01-{28 - j:02d} 10:00:00"})
    (d / "yelp_academic_dataset_review.json").write_text("\n".join(json.dumps(r) for r in reviews))


def test_prepare_yelp(tmp_path):
    _write_fake_yelp(tmp_path)
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(ROOT / "experiments/yelp/prepare_yelp.py"),
                    "--yelp-dir", str(tmp_path), "--out", str(out),
                    "--min-user", "3", "--min-rest", "3"], check=True, capture_output=True)

    inter = pd.read_csv(out / "interactions.csv")
    rest = pd.read_csv(out / "restaurants.csv")

    assert len(rest) == 6                                  # salon removed
    assert set(inter["strength"]) <= {1, 2, 3}
    assert set(inter["restaurant_id"]) <= set(rest["restaurant_id"])
    for _, g in inter.groupby("user_id"):                  # time-ordered within each user
        assert list(g["date"]) == sorted(g["date"])
    assert set(rest["cuisines"]) == {"Italian", "Mexican"}
