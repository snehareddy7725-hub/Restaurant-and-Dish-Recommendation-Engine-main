"""
Improved hybrid (v2): what changes when the content side is made richer?

Same protocol and the same validation/test users as evaluate_hybrid.py
(leave-last-out, users split 50/50 with seed 42, tuning on validation only,
test scored once). Changes to the model:

  1. ALS       - also tunes alpha (confidence scaling) and tries BM25
                 weighting of the user-item matrix, a standard fix for sparse
                 real-world data.
  2. Content   - all categories (multi-hot) instead of one cuisine label, and a
                 taste profile built from the user's whole history (recent and
                 strong interactions weigh more) instead of only the last item.
  3. Location  - if restaurants have latitude/longitude: how close a restaurant
                 is to the places the user usually goes (a kernel density over
                 their history, scale tuned). Food delivery only shows nearby
                 restaurants, so location matters.
  4. Popularity as a fourth signal.
  5. The four scores are min-max scaled per user and blended; the blend
     weights, and whether to exclude already-visited restaurants, are tuned on
     validation users.

Run:
    python experiments/evaluate_hybrid_v2.py --data-dir data_yelp \
        --md experiments/yelp/RESULTS_v2.md --title "Yelp results: improved hybrid (v2)"
"""

import argparse
import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from implicit.cpu.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

warnings.filterwarnings("ignore")
K = 10
SEED = 42


# ------------------------------------------------------------------ data
def load(data_dir: Path, max_eval_users):
    inter = pd.read_csv(data_dir / "interactions.csv").reset_index(drop=True)
    rest = pd.read_csv(data_dir / "restaurants.csv")

    held = inter.groupby("user_id").tail(1)
    train = inter.drop(held.index)
    users = inter["user_id"].unique()
    items = inter["restaurant_id"].unique()
    umap = {u: i for i, u in enumerate(users)}
    imap = {r: i for i, r in enumerate(items)}
    rows = train["user_id"].map(umap).values
    cols = train["restaurant_id"].map(imap).values
    shape = (len(users), len(items))

    M = sparse.csr_matrix((train["strength"].values.astype(np.float32), (rows, cols)), shape=shape)
    # position from the end of each user's history (0 = most recent), for recency weighting
    pos_from_end = train.groupby("user_id").cumcount(ascending=False).values
    hist = (rows, cols, train["strength"].values.astype(np.float32), pos_from_end)

    rest = rest.set_index("restaurant_id").loc[items].reset_index()

    # item content features: categories (multi-hot) or primary cuisine, price, rating
    if "categories" in rest.columns:
        cats = rest["categories"].fillna("").str.get_dummies(sep="|")
        cats = cats.loc[:, cats.sum() >= 5]                 # drop very rare categories
    else:
        cats = pd.get_dummies(rest["cuisines"])
    price = rest["avg_cost_for_two"].astype(float)
    pr = ((price - price.min()) / ((price.max() - price.min()) or 1)).values[:, None]
    rating = (rest["aggregate_rating"].astype(float) / 5.0).values[:, None]
    F = np.hstack([cats.astype(np.float32).values, pr, rating]).astype(np.float32)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12

    coords = None
    if {"latitude", "longitude"} <= set(rest.columns) and rest[["latitude", "longitude"]].notna().all().all():
        lat = rest["latitude"].astype(float).values
        lon = rest["longitude"].astype(float).values
        x = lon * np.cos(np.radians(lat.mean())) * 111.32   # km, local flat approximation
        y = lat * 110.57
        coords = np.stack([x, y], axis=1).astype(np.float32)

    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(users)
    half = len(shuffled) // 2
    splits = {"validation": shuffled[:half], "test": shuffled[half:]}
    if max_eval_users:
        splits = {k: v[:max_eval_users] for k, v in splits.items()}
    held_i = held.set_index("user_id")["restaurant_id"]
    S = {k: {"uidx": np.array([umap[u] for u in v]),
             "target": held_i.loc[v].map(imap).values} for k, v in splits.items()}

    info = {"interactions": len(inter), "users": len(users), "restaurants": len(items),
            "content_features": int(F.shape[1]), "has_location": coords is not None,
            "eval_users_per_split": len(S["validation"]["uidx"])}
    return M, hist, F, coords, S, info, shape


# ----------------------------------------------------------- components
def minmax_rows(X):
    lo = X.min(axis=1, keepdims=True)
    hi = X.max(axis=1, keepdims=True)
    return (X - lo) / np.maximum(hi - lo, 1e-9)


def history_weights(hist, shape, decay):
    """Sparse users x items weights: strength * decay^(position from end)."""
    rows, cols, strength, pos = hist
    w = strength * (decay ** pos if decay > 0 else (pos == 0).astype(np.float32))
    W = sparse.csr_matrix((w.astype(np.float32), (rows, cols)), shape=shape)
    sums = np.asarray(W.sum(axis=1)).ravel()
    return sparse.diags(1.0 / np.maximum(sums, 1e-9)) @ W


def content_scores(Wn, F, uidx):
    P = Wn[uidx] @ F
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    return P @ F.T                                          # cosine to the taste profile


def location_kernel(coords, scale_km):
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    return np.exp(-d / scale_km).astype(np.float32)


def location_scores(Wn, Kmat, uidx):
    return np.asarray(Wn[uidx] @ Kmat)                      # density near the user's usual places


def fit_als(M, factors, reg, alpha, bm25):
    X = bm25_weight(M, K1=100, B=0.8).tocsr().astype(np.float32) if bm25 else M
    m = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=50,
                                alpha=alpha, random_state=SEED)
    m.fit(X, show_progress=False)
    return m


def als_scores(model, uidx):
    return np.asarray(model.user_factors[uidx] @ model.item_factors.T)


def top_k(scores, M, uidx, exclude_seen):
    s = scores.copy()
    if exclude_seen:
        for r, u in enumerate(uidx):
            s[r, M.indices[M.indptr[u]:M.indptr[u + 1]]] = -np.inf
    part = np.argpartition(-s, K, axis=1)[:, :K]
    order = np.argsort(-np.take_along_axis(s, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def metrics(top, target):
    hit = top == target[:, None]
    pos = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    ndcg = np.where(pos >= 0, 1.0 / np.log2(pos + 2), 0.0)
    return {"hit_rate@10": round(100 * float((pos >= 0).mean()), 2), "ndcg@10": round(float(ndcg.mean()), 4)}


def weight_grid(n, step=0.1):
    steps = int(round(1 / step))
    for combo in itertools.product(range(steps + 1), repeat=n):
        if sum(combo) == steps:
            yield np.array(combo) / steps


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_yelp")
    ap.add_argument("--factors", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--regs", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 10.0, 40.0])
    ap.add_argument("--max-eval-users", type=int, default=5000)
    ap.add_argument("--md", default=None)
    ap.add_argument("--title", default="Improved hybrid (v2)")
    args = ap.parse_args()
    t0 = time.time()

    data_dir = Path(args.data_dir)
    M, hist, F, coords, S, info, shape = load(data_dir, args.max_eval_users or None)
    val, test = S["validation"], S["test"]
    print("Dataset:", json.dumps(info))

    def ev(scores, split, excl=True):
        return metrics(top_k(scores, M, split["uidx"], excl), split["target"])

    # Stage A: ALS (factors x regularization x alpha), then BM25 on the best
    print("\nStage A - ALS grid (validation, ALS only)")
    best = None
    for f, reg, a in itertools.product(args.factors, args.regs, args.alphas):
        m = fit_als(M, f, reg, a, bm25=False)
        hr = ev(als_scores(m, val["uidx"]), val)["hit_rate@10"]
        print(f"  factors={f:<4} reg={reg:<5} alpha={a:<5} HR@10={hr:.2f}%")
        if best is None or hr > best["hr"]:
            best = {"hr": hr, "factors": f, "reg": reg, "alpha": a, "bm25": False, "model": m}
    m = fit_als(M, best["factors"], best["reg"], best["alpha"], bm25=True)
    hr = ev(als_scores(m, val["uidx"]), val)["hit_rate@10"]
    print(f"  same + BM25 weighting          HR@10={hr:.2f}%")
    if hr > best["hr"]:
        best.update(hr=hr, bm25=True, model=m)
    als = best["model"]
    print(f"  -> ALS: factors={best['factors']}, reg={best['reg']}, alpha={best['alpha']}, bm25={best['bm25']}")

    # Stage B: content taste profile - how fast older history fades (0 = last item only)
    print("\nStage B - content profile recency (validation, content only)")
    best_decay, best_c = None, -1
    for decay in (0.0, 0.5, 0.8, 1.0):
        hr = ev(content_scores(history_weights(hist, shape, decay), F, val["uidx"]), val)["hit_rate@10"]
        print(f"  decay={decay:<4} HR@10={hr:.2f}%")
        if hr > best_c:
            best_decay, best_c = decay, hr
    Wn = history_weights(hist, shape, best_decay)

    # Stage C: location scale (validation, location only)
    best_scale, Kmat = None, None
    if coords is not None:
        print("\nStage C - location scale (validation, location only)")
        best_l = -1
        for scale in (0.5, 1.0, 2.0, 5.0):
            Kc = location_kernel(coords, scale)
            hr = ev(location_scores(Wn, Kc, val["uidx"]), val)["hit_rate@10"]
            print(f"  scale={scale:<4} km HR@10={hr:.2f}%")
            if hr > best_l:
                best_scale, best_l, Kmat = scale, hr, Kc

    pop = np.log1p(np.asarray(M.sum(axis=0)).ravel()).astype(np.float32)
    pop = (pop - pop.min()) / max(pop.max() - pop.min(), 1e-9)

    def components(split):
        u = split["uidx"]
        comps = [minmax_rows(als_scores(als, u)), minmax_rows(content_scores(Wn, F, u))]
        if Kmat is not None:
            comps.append(minmax_rows(location_scores(Wn, Kmat, u)))
        comps.append(np.broadcast_to(pop, (len(u), len(pop))))
        return comps

    names = ["ALS", "content"] + (["location"] if Kmat is not None else []) + ["popularity"]

    # Stage D: blend weights + exclude-seen, tuned on validation
    print("\nStage D - blend weights (validation)")
    cv = components(val)
    best_blend = None
    for excl in (True, False):
        for w in weight_grid(len(cv)):
            sc = sum(wi * c for wi, c in zip(w, cv) if wi > 0)
            hr = ev(sc, val, excl)["hit_rate@10"]
            if best_blend is None or hr > best_blend[0]:
                best_blend = (hr, w, excl)
    hr_b, w_best, excl_best = best_blend
    print("  -> weights: " + ", ".join(f"{n}={x:.1f}" for n, x in zip(names, w_best))
          + f"; exclude already-visited={excl_best} (HR@10={hr_b:.2f}%)")

    # Final: TEST users, once
    ct = components(test)
    results = {"Popularity baseline": ev(ct[-1], test)}
    results[f"ALS only (tuned{', BM25' if best['bm25'] else ''})"] = ev(ct[0], test)
    results["Content only (all categories, history profile)"] = ev(ct[1], test)
    if Kmat is not None:
        results[f"Location only (scale {best_scale} km)"] = ev(ct[2], test)
    v1_path = data_dir / "results.json"
    if v1_path.exists():
        v1 = json.loads(v1_path.read_text())["test_results"]
        key = next((k for k in v1 if k.startswith("Hybrid after tuning (") and "excluding" not in k), None)
        if key:
            results["Original hybrid, tuned (v1: ALS + cuisine, last item)"] = v1[key]
    results["Improved hybrid (v2)"] = ev(sum(wi * c for wi, c in zip(w_best, ct) if wi > 0), test, excl_best)

    print("\nRESULTS on held-out TEST users")
    print(f"  {'Model':<58} {'HR@10':>8} {'NDCG@10':>9}")
    for n, r in results.items():
        print(f"  {n:<58} {r['hit_rate@10']:>7.2f}% {r['ndcg@10']:>9.4f}")
    print(f"({time.time() - t0:.0f}s)")

    chosen = {"als": {k: best[k] for k in ("factors", "reg", "alpha", "bm25")},
              "content_decay": best_decay, "location_scale_km": best_scale,
              "weights": dict(zip(names, map(float, w_best))), "exclude_seen": excl_best}
    (data_dir / "results_v2.json").write_text(json.dumps(
        {"dataset": info, "chosen": chosen, "test_results": results}, indent=2))

    if args.md:
        lines = [f"# {args.title}", "",
                 f"{info['users']:,} users, {info['restaurants']:,} restaurants, "
                 f"{info['interactions']:,} interactions; {info['eval_users_per_split']:,} test users "
                 "(same users and split as the v1 run).", "",
                 "| Model (held-out test users) | Hit Rate@10 | NDCG@10 |", "|---|---|---|"]
        lines += [f"| {n} | {r['hit_rate@10']:.2f}% | {r['ndcg@10']:.4f} |" for n, r in results.items()]
        lines += ["", "Chosen on validation users: ALS factors = {factors}, regularization = {reg}, alpha = {alpha}, "
                  "BM25 = {bm25}; ".format(**chosen["als"])
                  + f"content recency decay = {best_decay}; "
                  + (f"location scale = {best_scale} km; " if best_scale else "")
                  + "blend weights: " + ", ".join(f"{n} {x:.1f}" for n, x in zip(names, w_best))
                  + f"; exclude already-visited = {excl_best}.", ""]
        Path(args.md).write_text("\n".join(lines))
        print(f"Saved {args.md}")


if __name__ == "__main__":
    main()
