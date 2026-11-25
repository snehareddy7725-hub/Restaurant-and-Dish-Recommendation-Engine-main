"""
Evaluate the hybrid recommender on any dataset in the project's format
(interactions.csv + restaurants.csv). Same method as evaluate_and_tune.py,
but vectorised so it also runs on large real datasets such as Yelp, and it
reports NDCG@10 next to Hit Rate@10.

Protocol (identical to the main project):
  * leave-one-out: each user's LAST interaction (file order) is held out,
    everything else is training data
  * users are shuffled (seed 42) and split 50/50 into validation and test
  * all tuning uses validation users only; test users are scored once
  * ALS candidates exclude restaurants the user has already interacted with;
    content candidates are cosine-similar to the user's last training item

Run:
    python experiments/evaluate_hybrid.py --data-dir data_yelp
    python experiments/evaluate_hybrid.py --data-dir data --factors 10 20 30 50 --regs 0.001 0.01 0.1
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

warnings.filterwarnings("ignore")
K = 10
N_CAND = 20
SEED = 42


# ---------------------------------------------------------------- data
def load(data_dir: Path, max_eval_users: int | None):
    inter = pd.read_csv(data_dir / "interactions.csv").reset_index(drop=True)
    rest = pd.read_csv(data_dir / "restaurants.csv")

    held = inter.groupby("user_id").tail(1)
    train = inter.drop(held.index)

    users = inter["user_id"].unique()
    items = inter["restaurant_id"].unique()
    umap = {u: i for i, u in enumerate(users)}
    imap = {r: i for i, r in enumerate(items)}

    M = sparse.csr_matrix(
        (train["strength"].values.astype(np.float32),
         (train["user_id"].map(umap).values, train["restaurant_id"].map(imap).values)),
        shape=(len(users), len(items)),
    )

    # content features in ALS item order: one-hot cuisine, min-max price, rating/5
    rest = rest.set_index("restaurant_id").loc[items].reset_index()
    cuis = pd.get_dummies(rest["cuisines"]).astype(np.float32).values
    price = rest["avg_cost_for_two"].astype(float)
    pr = ((price - price.min()) / ((price.max() - price.min()) or 1)).values[:, None]
    rating = (rest["aggregate_rating"].astype(float) / 5.0).values[:, None]
    F = np.hstack([cuis, pr, rating]).astype(np.float32)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12

    seed_item = train.groupby("user_id")["restaurant_id"].last()  # last training interaction

    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(users)
    half = len(shuffled) // 2
    splits = {"validation": shuffled[:half], "test": shuffled[half:]}
    if max_eval_users:
        splits = {k: v[:max_eval_users] for k, v in splits.items()}

    def build(split_users):
        h = held.set_index("user_id").loc[split_users]
        return {
            "uidx": np.array([umap[u] for u in split_users]),
            "target": h["restaurant_id"].map(imap).values,
            "seed": np.array([imap[seed_item[u]] for u in split_users]),
        }

    info = {
        "interactions": len(inter), "users": len(users), "restaurants": len(items),
        "train_rows": len(train), "eval_users_per_split": len(splits["validation"]),
        "density_percent": round(100 * len(inter) / (len(users) * len(items)), 4),
        "held_out_already_seen_percent": round(100 * float(np.mean([
            M[umap[u], imap[r]] > 0 for u, r in zip(held["user_id"], held["restaurant_id"])])), 1),
    }
    return M, F, {k: build(v) for k, v in splits.items()}, info


# ---------------------------------------------------------- candidates
def fit_als(M, factors, reg):
    m = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=50,
                                alpha=1.0, random_state=SEED)
    m.fit(M, show_progress=False)
    return m


def als_candidates(model, M, uidx, n=N_CAND):
    ids, scores = model.recommend(uidx, M[uidx], N=n, filter_already_liked_items=True)
    top = scores.max(axis=1, keepdims=True)
    top[top <= 0] = 1.0
    return ids, scores / top


def content_candidates(F, M, seeds, uidx, n=N_CAND, exclude_seen=False):
    sims = F[seeds] @ F.T
    sims[np.arange(len(seeds)), seeds] = -np.inf          # never recommend the seed itself
    if exclude_seen:
        for r, u in enumerate(uidx):
            sims[r, M.indices[M.indptr[u]:M.indptr[u + 1]]] = -np.inf
    ids = np.argsort(-sims, axis=1, kind="stable")[:, :n]
    return ids, np.take_along_axis(sims, ids, axis=1)


def blend(als, cont, w):
    """Weighted average over the union of both candidate lists -> top-K item ids."""
    a_ids, a_s = als
    c_ids, c_s = cont
    dup = c_ids[:, :, None] == a_ids[:, None, :]            # content item also in ALS list?
    c_into_a = np.where(dup, c_s[:, :, None], 0).sum(axis=1)  # content score for ALS entries
    a_part = w * a_s + (1 - w) * c_into_a
    c_part = np.where(dup.any(axis=2), -np.inf, (1 - w) * c_s)
    ids = np.hstack([a_ids, c_ids])
    sc = np.hstack([a_part, c_part])
    order = np.argsort(-sc, axis=1, kind="stable")[:, :K]
    return np.take_along_axis(ids, order, axis=1)


def rrf(als, cont, k=60):
    a_ids, _ = als
    c_ids, _ = cont
    ranks = 1.0 / (k + np.arange(1, a_ids.shape[1] + 1))
    dup = c_ids[:, :, None] == a_ids[:, None, :]
    a_part = ranks[None, :] + np.where(dup, ranks[None, :, None], 0).sum(axis=1)
    c_part = np.where(dup.any(axis=2), -np.inf, ranks[None, :])
    ids = np.hstack([a_ids, c_ids])
    sc = np.hstack([a_part, c_part])
    order = np.argsort(-sc, axis=1, kind="stable")[:, :K]
    return np.take_along_axis(ids, order, axis=1)


def popularity_top(M, uidx):
    pop = np.asarray(M.sum(axis=0)).ravel()
    order = np.argsort(-pop, kind="stable")
    out = []
    for u in uidx:
        seen = set(M.indices[M.indptr[u]:M.indptr[u + 1]])
        out.append([i for i in order[:K + len(seen)] if i not in seen][:K])
    return np.array(out)


def metrics(top, target):
    hit = top == target[:, None]
    pos = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    ndcg = np.where(pos >= 0, 1.0 / np.log2(pos + 2), 0.0)
    return {"hit_rate@10": round(100 * float((pos >= 0).mean()), 2), "ndcg@10": round(float(ndcg.mean()), 4)}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_yelp")
    ap.add_argument("--factors", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--regs", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0])
    ap.add_argument("--max-eval-users", type=int, default=5000,
                    help="cap users per split for speed (0 = all)")
    ap.add_argument("--out", default=None, help="where to save results JSON")
    ap.add_argument("--md", default=None, help="also write a Markdown results table here")
    ap.add_argument("--title", default="Results", help="heading for the Markdown file")
    args = ap.parse_args()
    t0 = time.time()

    data_dir = Path(args.data_dir)
    M, F, S, info = load(data_dir, args.max_eval_users or None)
    val, test = S["validation"], S["test"]
    print("Dataset:", json.dumps(info))

    def cands(model, split, exclude_seen=False):
        return (als_candidates(model, M, split["uidx"]),
                content_candidates(F, M, split["seed"], split["uidx"], exclude_seen=exclude_seen))

    # Stage A: ALS grid search (ALS-only Hit Rate on validation users)
    print("\nStage A - ALS grid search (validation, ALS only)")
    best = None
    for f, reg in itertools.product(args.factors, args.regs):
        m = fit_als(M, f, reg)
        ids, _ = als_candidates(m, M, val["uidx"], n=K)
        hr = metrics(ids, val["target"])["hit_rate@10"]
        print(f"  factors={f:<4} reg={reg:<6} HR@10={hr:.2f}%")
        if best is None or hr > best[0]:
            best = (hr, f, reg, m)
    _, best_f, best_reg, model = best
    print(f"  -> best: factors={best_f}, regularization={best_reg}")

    # Stage B: blend weight search (validation)
    weights = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    va = cands(model, val)
    hrs = [metrics(blend(*va, w), val["target"])["hit_rate@10"] for w in weights]
    best_w = float(weights[int(np.argmax(hrs))])
    print(f"\nStage B - best ALS weight on validation: {best_w:.2f} (HR@10={max(hrs):.2f}%)")

    # Stage C: weighted average vs RRF (validation)
    hr_rrf = metrics(rrf(*va), val["target"])["hit_rate@10"]
    use_rrf = hr_rrf > max(hrs)
    print(f"Stage C - RRF on validation: {hr_rrf:.2f}% -> using {'RRF' if use_rrf else 'weighted average'}")

    # Extra variant: also exclude already-seen restaurants from content candidates
    va_x = cands(model, val, exclude_seen=True)
    hrs_x = [metrics(blend(*va_x, w), val["target"])["hit_rate@10"] for w in weights]
    best_wx = float(weights[int(np.argmax(hrs_x))])

    # Final: score once on TEST users
    te = cands(model, test)
    te_x = cands(model, test, exclude_seen=True)
    untuned = fit_als(M, 20, 0.01)
    te_untuned = cands(untuned, test)

    results = {
        "Popularity baseline": metrics(popularity_top(M, test["uidx"]), test["target"]),
        "ALS only (tuned)": metrics(te[0][0][:, :K], test["target"]),
        "Content only": metrics(blend(*te, 0.0), test["target"]),
        "Hybrid before tuning (ALS reg 0.01, 50/50)": metrics(blend(*te_untuned, 0.5), test["target"]),
        f"Hybrid after tuning ({'RRF' if use_rrf else f'w={best_w:.2f}'})":
            metrics(rrf(*te) if use_rrf else blend(*te, best_w), test["target"]),
        f"Hybrid after tuning, excluding already-seen (w={best_wx:.2f})": metrics(blend(*te_x, best_wx), test["target"]),
    }

    print("\nRESULTS on held-out TEST users")
    print(f"  {'Model':<62} {'HR@10':>8} {'NDCG@10':>9}")
    for name, r in results.items():
        print(f"  {name:<62} {r['hit_rate@10']:>7.2f}% {r['ndcg@10']:>9.4f}")
    print(f"\nChosen: ALS factors={best_f}, regularization={best_reg}, weight={best_w:.2f}  "
          f"({time.time() - t0:.0f}s)")

    out = Path(args.out) if args.out else data_dir / "results.json"
    out.write_text(json.dumps({
        "dataset": info, "best_factors": best_f, "best_regularization": best_reg,
        "best_als_weight": best_w, "best_als_weight_excluding_seen": best_wx,
        "test_results": results}, indent=2))
    print(f"Saved {out}")

    if args.md:
        summ = data_dir / "summary.json"
        extra = json.loads(summ.read_text()) if summ.exists() else {}
        lines = [f"# {args.title}", ""]
        if extra:
            lines += [f"**Data:** {extra.get('city', '')}: {info['users']:,} users, {info['restaurants']:,} restaurants, "
                      f"{info['interactions']:,} interactions (density {info['density_percent']}%, "
                      f"median {extra.get('interactions_per_user_median', '?')} per user)."]
        else:
            lines += [f"**Data:** {info['users']:,} users, {info['restaurants']:,} restaurants, "
                      f"{info['interactions']:,} interactions (density {info['density_percent']}%)."]
        lines += [f"Held-out restaurant already in the user's history: {info['held_out_already_seen_percent']}%. "
                  f"Evaluated on {info['eval_users_per_split']:,} validation and test users each.", "",
                  "| Model (held-out test users) | Hit Rate@10 | NDCG@10 |", "|---|---|---|"]
        lines += [f"| {n} | {r['hit_rate@10']:.2f}% | {r['ndcg@10']:.4f} |" for n, r in results.items()]
        lines += ["", f"Tuned on validation users: ALS factors = {best_f}, regularization = {best_reg}, "
                  f"ALS weight = {best_w:.2f}.", ""]
        Path(args.md).write_text("\n".join(lines))
        print(f"Saved {args.md}")


if __name__ == "__main__":
    main()
