"""
What happens to the metrics if you hide the last 1, 2 or 3 interactions per user?

Shows why Hit Rate numbers are only comparable under the same test setup:
with more hidden items, Hit Rate rises (more chances to hit) while Recall,
the share of hidden restaurants actually found, stays about the same.
Uses the tuned hybrid (ALS 20 factors, reg 0.001, weight 0.61) and the same
500 test users as evaluate_and_tune.py.

Run:  python experiments/leave_k_out.py
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.cpu.als import AlternatingLeastSquares

warnings.filterwarnings("ignore")

d = pd.read_csv("data/interactions.csv").reset_index(drop=True)
R = pd.read_csv("data/restaurants.csv")
content = pickle.load(open("data/content_model.pkl", "rb"))
S, c2i = content["similarity_matrix"], content["id_to_index"]

U = d.user_id.unique()
I = d.restaurant_id.unique()
um = {u: i for i, u in enumerate(U)}
im = {x: i for i, x in enumerate(I)}
inv = {i: x for x, i in im.items()}
test_users = set(np.random.default_rng(42).permutation(U)[500:])

print(f"{'hidden/user':>11} {'HitRate@10':>11} {'Recall@10':>10} {'Precision@10':>13} {'NDCG@10':>8}")
for k in (1, 2, 3):
    held = d.groupby("user_id").tail(k)
    tr = d.drop(held.index)
    last = tr.groupby("user_id").tail(1).set_index("user_id").restaurant_id
    M = sp.csr_matrix((tr.strength.values.astype(np.float32),
                       (tr.user_id.map(um), tr.restaurant_id.map(im))), shape=(len(U), len(I)))
    m = AlternatingLeastSquares(factors=20, regularization=0.001, iterations=50, alpha=1.0, random_state=42)
    m.fit(M, show_progress=False)

    hr, rec, prec, nd = [], [], [], []
    for uid, g in held[held.user_id.isin(test_users)].groupby("user_id"):
        u = um[uid]
        ids, s = m.recommend(u, M[u], N=20, filter_already_liked_items=True)
        a = {inv[i]: x / max(s) for i, x in zip(ids, s)}
        idx = c2i[last[uid]]
        cs = {R.iloc[i].restaurant_id: x for i, x in sorted(enumerate(S[idx]), key=lambda t: -t[1])[1:21]}
        top = sorted(set(a) | set(cs), key=lambda r: -(0.61 * a.get(r, 0) + 0.39 * cs.get(r, 0)))[:10]
        rel = set(g.restaurant_id)
        h = [t in rel for t in top]
        hr.append(any(h))
        rec.append(sum(h) / len(rel))
        prec.append(sum(h) / 10)
        dcg = sum(1 / np.log2(i + 2) for i, x in enumerate(h) if x)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), 10)))
        nd.append(dcg / idcg)
    print(f"{k:>11} {np.mean(hr)*100:>10.1f}% {np.mean(rec)*100:>9.1f}% {np.mean(prec)*100:>12.1f}% {np.mean(nd):>8.3f}")
