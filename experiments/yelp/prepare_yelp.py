"""
Convert the Yelp Open Dataset into this project's data format.

Input  (from https://www.yelp.com/dataset):
    yelp_academic_dataset_business.json
    yelp_academic_dataset_review.json

Output (same columns as data/, so the same pipeline can run on it):
    <out>/interactions.csv   user_id, restaurant_id, interaction_type, strength, date
    <out>/restaurants.csv    restaurant_id, name, cuisines, avg_cost_for_two,
                             aggregate_rating, price_bucket
    <out>/summary.json       what was kept

How Yelp is mapped onto the project:
  * Only businesses whose categories include "Restaurants", in one city
    (default: the city with the most restaurant reviews).
  * Each review = one implicit interaction ("the user went there").
    Strength from the review's stars:  4-5 stars -> 3,  3 stars -> 2,
    1-2 stars -> 1. So a visit the user loved counts like an "order",
    and a bad visit only like a "view".
  * Reviews are sorted by date, so "last interaction" really is the most
    recent one (a time-based leave-one-out split, unlike the synthetic data).
  * Iterative k-core filter: keep users and restaurants with at least
    --min-user / --min-rest interactions, so every user has some history.
  * Content features: primary cuisine, price level (RestaurantsPriceRange2,
    1-4) and the restaurant's average star rating. All categories and the
    latitude/longitude are also saved for the improved model (v2).

Run:
    python experiments/yelp/prepare_yelp.py --yelp-dir ~/Downloads/yelp --out data_yelp
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

BUSINESS_FILE = "yelp_academic_dataset_business.json"
REVIEW_FILE = "yelp_academic_dataset_review.json"

# Checked in this order; the first match becomes the restaurant's cuisine.
CUISINES = [
    "Indian", "Chinese", "Japanese", "Sushi Bars", "Thai", "Vietnamese", "Korean",
    "Mexican", "Italian", "Pizza", "Greek", "Mediterranean", "Middle Eastern",
    "French", "Caribbean", "Seafood", "Steakhouses", "Barbeque", "Vegan", "Vegetarian",
    "Burgers", "Chicken Wings", "Fast Food", "Sandwiches", "Delis", "Breakfast & Brunch",
    "Cafes", "Bakeries", "Desserts", "American (New)", "American (Traditional)",
]
SUSHI_TO = {"Sushi Bars": "Japanese"}
GENERIC = {"Restaurants", "Food"}  # on every restaurant, so they carry no information


def primary_cuisine(categories: str) -> str:
    cats = {c.strip() for c in (categories or "").split(",")}
    for c in CUISINES:
        if c in cats:
            return SUSHI_TO.get(c, c)
    return "Other"


def price_level(attributes) -> int:
    try:
        lvl = int(str((attributes or {}).get("RestaurantsPriceRange2", "")).strip("'\""))
        return lvl if 1 <= lvl <= 4 else 2
    except ValueError:
        return 2  # missing -> assume mid-range


def stars_to_strength(stars: float) -> int:
    return 3 if stars >= 4 else (2 if stars >= 3 else 1)


def load_restaurants(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            if "Restaurants" not in (b.get("categories") or ""):
                continue
            rows.append({
                "business_id": b["business_id"],
                "name": b.get("name", ""),
                "city": (b.get("city") or "").strip(),
                "state": b.get("state", ""),
                "cuisines": primary_cuisine(b.get("categories")),
                "price_level": price_level(b.get("attributes")),
                "aggregate_rating": float(b.get("stars", 0)),
                "review_count": int(b.get("review_count", 0)),
                "categories": "|".join(sorted(
                    c.strip() for c in (b.get("categories") or "").split(",")
                    if c.strip() and c.strip() not in GENERIC)),
                "latitude": b.get("latitude"),
                "longitude": b.get("longitude"),
            })
    return pd.DataFrame(rows)


def load_reviews(path: Path, keep_ids: set) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            r = json.loads(line)
            if r["business_id"] in keep_ids:
                rows.append((r["user_id"], r["business_id"], float(r["stars"]), r["date"]))
            if i % 1_000_000 == 0:
                print(f"  read {i:,} reviews, kept {len(rows):,}")
    return pd.DataFrame(rows, columns=["yelp_user", "business_id", "stars", "date"])


def k_core(df: pd.DataFrame, min_user: int, min_rest: int) -> pd.DataFrame:
    while True:
        u = df["yelp_user"].map(df["yelp_user"].value_counts())
        b = df["business_id"].map(df["business_id"].value_counts())
        keep = (u >= min_user) & (b >= min_rest)
        if keep.all():
            return df
        df = df[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yelp-dir", required=True, help="folder with the Yelp JSON files")
    ap.add_argument("--out", default="data_yelp")
    ap.add_argument("--city", default="auto", help='city name, or "auto" for the busiest one')
    ap.add_argument("--min-user", type=int, default=5, help="min interactions per user")
    ap.add_argument("--min-rest", type=int, default=10, help="min interactions per restaurant")
    args = ap.parse_args()

    ydir = Path(args.yelp_dir).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading restaurants ...")
    rest = load_restaurants(ydir / BUSINESS_FILE)
    city = args.city
    if city == "auto":
        city = rest.groupby("city")["review_count"].sum().idxmax()
    rest = rest[rest["city"].str.lower() == city.lower()]
    if rest.empty:
        raise SystemExit(f"No restaurants found in city '{city}'")
    print(f"City: {city} ({len(rest):,} restaurants)")

    print("Loading reviews (the review file is large; this takes a few minutes) ...")
    rev = load_reviews(ydir / REVIEW_FILE, set(rest["business_id"]))
    n_raw = len(rev)

    rev = k_core(rev, args.min_user, args.min_rest)
    rev = rev.sort_values(["yelp_user", "date"], kind="stable")  # time order within each user

    # Integer IDs, like the original data (users from 0, restaurants from 1)
    users = {u: i for i, u in enumerate(rev["yelp_user"].unique())}
    kept_rest = rest[rest["business_id"].isin(set(rev["business_id"]))].reset_index(drop=True)
    rest_ids = {b: i + 1 for i, b in enumerate(kept_rest["business_id"])}

    interactions = pd.DataFrame({
        "user_id": rev["yelp_user"].map(users).values,
        "restaurant_id": rev["business_id"].map(rest_ids).values,
        "interaction_type": "review",
        "strength": rev["stars"].map(stars_to_strength).values,
        "date": rev["date"].values,
    })

    restaurants = pd.DataFrame({
        "restaurant_id": kept_rest["business_id"].map(rest_ids),
        "name": kept_rest["name"],
        "cuisines": kept_rest["cuisines"],
        "avg_cost_for_two": kept_rest["price_level"],  # Yelp price level 1-4 (the model min-max scales it)
        "aggregate_rating": kept_rest["aggregate_rating"],
        "price_bucket": kept_rest["price_level"].map({1: "Low", 2: "Mid", 3: "High", 4: "High"}),
        # extra columns used by experiments/evaluate_hybrid_v2.py
        "categories": kept_rest["categories"],
        "latitude": kept_rest["latitude"],
        "longitude": kept_rest["longitude"],
    })

    interactions.to_csv(out / "interactions.csv", index=False)
    restaurants.to_csv(out / "restaurants.csv", index=False)

    per_user = interactions.groupby("user_id").size()
    summary = {
        "city": city,
        "raw_reviews_in_city": int(n_raw),
        "interactions": int(len(interactions)),
        "users": int(len(users)),
        "restaurants": int(len(restaurants)),
        "interactions_per_user_median": float(per_user.median()),
        "density_percent": round(100 * len(interactions) / (len(users) * len(restaurants)), 4),
        "strength_counts": {int(k): int(v) for k, v in Counter(interactions["strength"]).items()},
        "top_cuisines": restaurants["cuisines"].value_counts().head(8).to_dict(),
        "min_user": args.min_user, "min_rest": args.min_rest,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved {out}/interactions.csv and {out}/restaurants.csv")


if __name__ == "__main__":
    main()
