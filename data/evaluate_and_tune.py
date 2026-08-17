"""
Comprehensive improvement testing for the hybrid recommender.

Tests, in order, all tuned on VALIDATION users only:
  Stage A: ALS hyperparameters (factors, regularization)
  Stage B: Blend weight (fine-grained, using best ALS from Stage A)
  Stage C: Multi-item content seeding (last 1 vs last 3 interactions)
  Stage D: Reciprocal Rank Fusion (RRF) vs weighted score averaging

Final number is reported ONLY on the held-out TEST users, using
whatever combination of choices won each stage — this is the honest,
leakage-free number.
"""

import pickle
import itertools
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares

DATA_DIR = "data"
K = 10
RANDOM_SEED = 42

interactions = pd.read_csv(f"{DATA_DIR}/interactions.csv")
restaurants = pd.read_csv(f"{DATA_DIR}/restaurants.csv")
with open(f"{DATA_DIR}/content_model.pkl", "rb") as f:
    content_data = pickle.load(f)
content_sim = content_data["similarity_matrix"]
content_id_to_index = content_data["id_to_index"]

unique_users = interactions["user_id"].unique()
unique_items = interactions["restaurant_id"].unique()
user_mapper = {uid: idx for idx, uid in enumerate(unique_users)}
item_mapper = {iid: idx for idx, iid in enumerate(unique_items)}
item_inv_mapper = {idx: iid for iid, idx in item_mapper.items()}

rng = np.random.default_rng(RANDOM_SEED)
shuffled_users = rng.permutation(unique_users)
split_point = len(shuffled_users) // 2
validation_users = set(shuffled_users[:split_point])
test_users = set(shuffled_users[split_point:])

interactions_sorted = interactions.reset_index(drop=True)
test_rows_all = interactions_sorted.groupby("user_id").tail(1)
train_rows = interactions_sorted.drop(test_rows_all.index)

validation_rows = test_rows_all[test_rows_all["user_id"].isin(validation_users)]
test_rows = test_rows_all[test_rows_all["user_id"].isin(test_users)]

# Last-3 (for multi-item seeding) and last-1 (original) TRAIN interactions per user
train_sorted = train_rows.sort_values(["user_id"])
last3_per_user = train_sorted.groupby("user_id").tail(3).groupby("user_id")["restaurant_id"].apply(list)
last1_per_user = train_sorted.groupby("user_id").tail(1).set_index("user_id")["restaurant_id"]


def build_matrix(df):
    ui = df["user_id"].map(user_mapper).values
    ii = df["restaurant_id"].map(item_mapper).values
    conf = df["strength"].values.astype(np.float32)
    return sparse.csr_matrix((conf, (ui, ii)), shape=(len(user_mapper), len(item_mapper)))


train_matrix = build_matrix(train_rows)


# ============================================================
# STAGE A: ALS hyperparameter tuning (factors, regularization)
# Scored using ALS-ONLY hit rate on validation users
# ============================================================
print("=" * 60)
print("STAGE A: ALS hyperparameter tuning")
print("=" * 60)

def als_only_hit_rate(model, rows):
    hits, n = 0, 0
    for _, row in rows.iterrows():
        uid, held = row["user_id"], row["restaurant_id"]
        n += 1
        if uid not in user_mapper:
            continue
        uidx = user_mapper[uid]
        ids, _ = model.recommend(uidx, train_matrix[uidx], N=K, filter_already_liked_items=True)
        recs = {item_inv_mapper[i] for i in ids if i in item_inv_mapper}
        hits += held in recs
    return hits / n

factor_options = [10, 20, 30, 50]
reg_options = [0.001, 0.01, 0.1]

best_als_hr, best_factors, best_reg, best_als_model = -1, None, None, None
for factors, reg in itertools.product(factor_options, reg_options):
    model = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=50, alpha=1.0, random_state=RANDOM_SEED)
    model.fit(train_matrix)
    hr = als_only_hit_rate(model, validation_rows)
    marker = ""
    if hr > best_als_hr:
        best_als_hr, best_factors, best_reg, best_als_model = hr, factors, reg, model
        marker = "  <-- best so far"
    print(f"  factors={factors:3d}  reg={reg:.3f}  ALS-only Hit Rate@{K} = {hr*100:.2f}%{marker}")

print(f"\nBest ALS config: factors={best_factors}, regularization={best_reg} "
      f"(validation ALS-only Hit Rate@{K} = {best_als_hr*100:.2f}%)")

als_model = best_als_model  # use best model going forward


# ============================================================
# Helper: build score dicts (ALS + content, both variants)
# ============================================================
def get_als_scores(uid, n=20):
    if uid not in user_mapper:
        return {}
    uidx = user_mapper[uid]
    ids, scores = als_model.recommend(uidx, train_matrix[uidx], N=n, filter_already_liked_items=True)
    raw = {item_inv_mapper[i]: float(s) for i, s in zip(ids, scores) if i in item_inv_mapper}
    max_s = max(raw.values()) if raw else 1
    return {rid: s / max_s for rid, s in raw.items()}

def get_content_scores_single(uid, n=20):
    """Original approach: seed from only the last interaction."""
    seed = last1_per_user.get(uid)
    if seed is None or seed not in content_id_to_index:
        return {}
    idx = content_id_to_index[seed]
    sims = sorted(enumerate(content_sim[idx]), key=lambda x: x[1], reverse=True)[1:n + 1]
    return {restaurants.iloc[i]["restaurant_id"]: float(s) for i, s in sims}

def get_content_scores_multi(uid, n=20):
    """New approach: average similarity across the last 3 interactions."""
    seeds = last3_per_user.get(uid, [])
    seeds = [s for s in seeds if s in content_id_to_index]
    if not seeds:
        return {}
    acc = {}
    for seed in seeds:
        idx = content_id_to_index[seed]
        for i, score in enumerate(content_sim[idx]):
            rid = restaurants.iloc[i]["restaurant_id"]
            if rid in seeds:  # don't recommend an item already interacted with
                continue
            acc[rid] = acc.get(rid, 0.0) + float(score)
    for rid in acc:
        acc[rid] /= len(seeds)
    top = sorted(acc.items(), key=lambda x: -x[1])[:n]
    return dict(top)


# ============================================================
# STAGE B: fine-grained blend weight search (weighted average),
# using single-item content seeding (original) for now
# ============================================================
print("\n" + "=" * 60)
print("STAGE B: Fine-grained blend weight search")
print("=" * 60)

def weighted_hit_rate(w_als, rows, content_fn):
    w_content = 1 - w_als
    hits, n = 0, 0
    for _, row in rows.iterrows():
        uid, held = row["user_id"], row["restaurant_id"]
        n += 1
        als_s = get_als_scores(uid)
        cont_s = content_fn(uid)
        all_ids = set(als_s) | set(cont_s)
        if not all_ids:
            continue
        scored = [(rid, w_als * als_s.get(rid, 0) + w_content * cont_s.get(rid, 0)) for rid in all_ids]
        scored.sort(key=lambda x: -x[1])
        top_k = {rid for rid, _ in scored[:K]}
        hits += held in top_k
    return hits / n

best_w, best_w_hr = None, -1
for w in np.arange(0.45, 0.81, 0.02):
    hr = weighted_hit_rate(w, validation_rows, get_content_scores_single)
    if hr > best_w_hr:
        best_w_hr, best_w = hr, w
print(f"Best weight (fine search): ALS={best_w:.2f}, Content={1-best_w:.2f} "
      f"(validation Hit Rate@{K} = {best_w_hr*100:.2f}%)")


# ============================================================
# STAGE C: multi-item content seeding vs single-item
# (both evaluated at the best weight found in Stage B)
# ============================================================
print("\n" + "=" * 60)
print("STAGE C: Multi-item content seeding vs single-item")
print("=" * 60)

hr_single = weighted_hit_rate(best_w, validation_rows, get_content_scores_single)
hr_multi = weighted_hit_rate(best_w, validation_rows, get_content_scores_multi)
print(f"Single-item seed: validation Hit Rate@{K} = {hr_single*100:.2f}%")
print(f"Multi-item seed (last 3): validation Hit Rate@{K} = {hr_multi*100:.2f}%")

best_content_fn = get_content_scores_multi if hr_multi > hr_single else get_content_scores_single
best_content_label = "multi-item" if hr_multi > hr_single else "single-item"
print(f"Winner: {best_content_label} seeding")


# ============================================================
# STAGE D: Reciprocal Rank Fusion (RRF) vs weighted averaging
# ============================================================
print("\n" + "=" * 60)
print("STAGE D: Reciprocal Rank Fusion (RRF) vs weighted averaging")
print("=" * 60)

def rrf_hit_rate(rows, content_fn, rrf_k=60):
    hits, n = 0, 0
    for _, row in rows.iterrows():
        uid, held = row["user_id"], row["restaurant_id"]
        n += 1
        als_s = get_als_scores(uid)
        cont_s = content_fn(uid)
        als_rank = {rid: rank for rank, rid in enumerate(sorted(als_s, key=lambda r: -als_s[r]), start=1)}
        cont_rank = {rid: rank for rank, rid in enumerate(sorted(cont_s, key=lambda r: -cont_s[r]), start=1)}
        all_ids = set(als_rank) | set(cont_rank)
        if not all_ids:
            continue
        rrf_scores = []
        for rid in all_ids:
            score = 0.0
            if rid in als_rank:
                score += 1 / (rrf_k + als_rank[rid])
            if rid in cont_rank:
                score += 1 / (rrf_k + cont_rank[rid])
            rrf_scores.append((rid, score))
        rrf_scores.sort(key=lambda x: -x[1])
        top_k = {rid for rid, _ in rrf_scores[:K]}
        hits += held in top_k
    return hits / n

hr_weighted_final = weighted_hit_rate(best_w, validation_rows, best_content_fn)
hr_rrf = rrf_hit_rate(validation_rows, best_content_fn)
print(f"Weighted average (w={best_w:.2f}): validation Hit Rate@{K} = {hr_weighted_final*100:.2f}%")
print(f"RRF fusion: validation Hit Rate@{K} = {hr_rrf*100:.2f}%")

use_rrf = hr_rrf > hr_weighted_final
print(f"Winner: {'RRF' if use_rrf else 'Weighted average'}")


# ============================================================
# FINAL: evaluate the winning configuration on TEST users
# (never touched during any tuning stage above)
# ============================================================
print("\n" + "=" * 60)
print("FINAL RESULT — evaluated on held-out TEST users")
print("=" * 60)
print(f"Config: ALS(factors={best_factors}, reg={best_reg}), "
      f"content seeding={best_content_label}, "
      f"combination={'RRF' if use_rrf else f'weighted avg w={best_w:.2f}'}")

if use_rrf:
    final_test_hr = rrf_hit_rate(test_rows, best_content_fn)
else:
    final_test_hr = weighted_hit_rate(best_w, test_rows, best_content_fn)

baseline_test_hr = weighted_hit_rate(0.5, test_rows, get_content_scores_single)

print(f"\nOriginal baseline (default ALS, 50/50 weight, single-item seed): {baseline_test_hr*100:.2f}%")
print(f"Fully tuned pipeline:                                             {final_test_hr*100:.2f}%")
print(f"Improvement: {(final_test_hr - baseline_test_hr)*100:+.2f} percentage points")
print("=" * 60)
