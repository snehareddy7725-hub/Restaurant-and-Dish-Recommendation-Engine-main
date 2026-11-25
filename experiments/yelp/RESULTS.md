# Yelp results (real data)

**Data:** Philadelphia: 26,251 users, 4,145 restaurants, 403,786 interactions (density 0.3711%, median 8.0 per user).
Held-out restaurant already in the user's history: 5.7%. Evaluated on 5,000 validation and test users each.

| Model (held-out test users) | Hit Rate@10 | NDCG@10 |
|---|---|---|
| Popularity baseline | 4.46% | 0.0243 |
| ALS only (tuned) | 6.52% | 0.0338 |
| Content only | 0.80% | 0.0042 |
| Hybrid before tuning (ALS reg 0.01, 50/50) | 1.94% | 0.0144 |
| Hybrid after tuning (w=0.96) | 6.52% | 0.0338 |
| Hybrid after tuning, excluding already-seen (w=0.96) | 6.52% | 0.0338 |

Tuned on validation users: ALS factors = 20, regularization = 1.0, ALS weight = 0.96.
