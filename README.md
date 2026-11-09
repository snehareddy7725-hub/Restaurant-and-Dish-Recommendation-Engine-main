# 🍔 Hybrid Restaurant Recommender (ALS + Content-Based)

![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-service-009688)
![Docker](https://img.shields.io/badge/Docker-containerized-2496ED)
![Streamlit](https://img.shields.io/badge/Streamlit-demo-red)

A Swiggy/Zomato-style restaurant recommender that combines **collaborative filtering (ALS matrix factorization)** with **content-based filtering**, served as a **containerized FastAPI service** with explicit cold-start handling, request logging and an interaction-based retraining trigger.

---

## Results (Hit Rate@10, held-out test users)

| Model | Hit Rate@10 |
|---|---|
| Popularity baseline | 3.8% |
| ALS only (tuned) | 23.2% |
| Content only | 31.6% |
| Hybrid before tuning (ALS reg = 0.01, 50/50 blend) | 31.8% |
| **Hybrid after tuning (ALS reg = 0.001, 0.61 / 0.39 blend)** | **35.8%** |

Reproduce with `python evaluate_and_tune.py`.

**Protocol:** leave-one-out (each user's last interaction is hidden). Users are split 50/50 into **validation** users (all tuning) and **test** users (reported once at the end), so no tuning decision sees the test users.

**Tuning stages (validation users):**
1. ALS grid search: factors ∈ {10, 20, 30, 50} × regularization ∈ {0.001, 0.01, 0.1} → factors = 20, reg = 0.001
2. Blend weight search, ALS weight 0.45 → 0.80 in steps of 0.02 → **0.61 ALS / 0.39 content**
3. Content seed: last 1 item (35.8%) vs average of last 3 (31.0%) → last 1
4. Combination: weighted average (35.8%) vs Reciprocal Rank Fusion (32.2%) → weighted average

---

## How it works

**Data (synthetic):** 1,000 users, 100 restaurants, ~20k implicit interactions. Strength: view = 1, cart = 2, order = 3.

1. **Collaborative filtering:** implicit-feedback ALS (`implicit` library) on a sparse users × restaurants matrix. It learns 20 latent factors per user and restaurant.
2. **Content-based:** each restaurant becomes a vector of one-hot cuisine, min-max price and rating/5; cosine similarity to the user's last interaction.
3. **Hybrid:** top 20 candidates from each model; ALS scores scaled to 0–1;
   `score = 0.61 × ALS + 0.39 × content`.

### Cold-start detection
Every request is routed to exactly one mode, and the mode is returned in the response:

| Mode | When |
|---|---|
| `hybrid` | user is in the trained ALS model **and** has a last interaction known to the content model |
| `als_only` | user is in the ALS model but has no usable content seed |
| `content_only` | user is **not** in the ALS model yet (new since the last retrain) but has interacted |
| `popularity` | no history at all → most popular restaurants (optional `cuisine` preference) |

### Retraining trigger
`POST /interactions` appends to `data/new_interactions.csv`, and the interaction is used straight away (a new user moves from `popularity` to `content_only`). After `RECO_RETRAIN_THRESHOLD` new interactions (default 500), both models retrain **in a background thread**, are saved atomically (temp file + rename) and are hot-reloaded without restarting the service. If training fails, the old model keeps serving.

### Request logging
Every request writes one JSON line to `logs/requests.jsonl` (and stdout) with request id, method, path, status, latency, and, for recommendations, `user_id` and the chosen `mode`. `/health` also reports how often each mode was served.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/recommend/{user_id}?n=10&cuisine=Italian` | Recommendations + chosen mode |
| POST | `/interactions` | Log `{user_id, restaurant_id, interaction_type: view/cart/order}` |
| GET | `/health` | Model info, pending interactions, mode counts |
| POST | `/admin/retrain` | Force a retrain |
| GET | `/admin/retrain/status` | Last retrain result |

Interactive docs: `http://localhost:8000/docs`

```json
GET /recommend/0?n=2
{
  "user_id": 0,
  "mode": "hybrid",
  "recommendations": [
    {"restaurant_id": 62, "name": "Resto_62", "cuisines": "North Indian", "score": 0.9967,
     "als_score": 1.0, "content_score": 0.9917, ...},
    ...
  ]
}
```

---

## Run it

**Docker**
```bash
docker compose up --build        # API on http://localhost:8000
```
The image trains the models at build time, so the pickles always match the installed library versions. A named volume keeps logged interactions and retrained models between restarts. The service runs as a single worker, because the retrain counter lives in memory.

**Locally**
```bash
pip install -r requirements-dev.txt
python train_models.py            # trains + saves data/*.pkl
uvicorn api.main:app --reload     # API
streamlit run app.py              # demo UI (shows mode + score breakdown)
python evaluate_and_tune.py       # full tuning + results table
pytest                            # tests
```

---

## Project structure
```
├── api/main.py              FastAPI service, request-logging middleware
├── recommender/
│   ├── config.py            hyperparameters, paths, env overrides
│   ├── training.py          ALS + content model training, atomic save, shape checks
│   ├── engine.py            cold-start mode detection + hybrid scoring
│   └── retraining.py        interaction log + background retraining trigger
├── app.py                   Streamlit demo
├── train_models.py          offline training entry point
├── evaluate_and_tune.py     tuning + Hit Rate@10 evaluation
├── tests/                   engine + API end-to-end tests
├── Dockerfile, docker-compose.yml
└── data/                    interactions, restaurants, user metadata, trained models
```

## Notable bug fixed
An earlier ALS model had `user_factors` of shape (100, 20) and `item_factors` of shape (1000, 20), swapped with the real counts, because the matrix passed to `.fit()` was items × users. Training now builds the matrix explicitly as users × items, and asserts both factor shapes before saving (see `recommender/training.py` and `tests/test_engine.py`).

## Limitations and next steps
- The data is synthetic (no timestamps). Next: a real dataset and a time-based split.
- 64% of held-out items are restaurants the user had already interacted with. ALS filters seen items, the content model does not, so part of the hybrid's gain comes from repeat orders.
- Next: NDCG/MRR, a learning-to-rank stage, and ANN search (FAISS) for large catalogues.
