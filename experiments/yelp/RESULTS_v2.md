# Yelp results: improved hybrid (v2)

26,251 users, 4,145 restaurants, 403,786 interactions; 5,000 test users (same users and split as the v1 run).

| Model (held-out test users) | Hit Rate@10 | NDCG@10 |
|---|---|---|
| Popularity baseline | 4.46% | 0.0243 |
| ALS only (tuned) | 6.66% | 0.0345 |
| Content only (all categories, history profile) | 0.82% | 0.0041 |
| Location only (scale 0.5 km) | 2.02% | 0.0098 |
| Original hybrid, tuned (v1: ALS + cuisine, last item) | 6.52% | 0.0338 |
| Improved hybrid (v2) | 7.76% | 0.0394 |

Chosen on validation users: ALS factors = 50, regularization = 1.0, alpha = 10.0, BM25 = False; content recency decay = 0.5; location scale = 0.5 km; blend weights: ALS 0.6, content 0.1, location 0.2, popularity 0.1; exclude already-visited = False.
