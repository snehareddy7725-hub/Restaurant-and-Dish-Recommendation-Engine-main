# 🍔 Swiggy-Style Hybrid Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/ML-Hybrid%20Engine-green)

A personalized restaurant recommendation system that mimics the logic of apps like Swiggy and Zomato. It uses a **Hybrid Filtering Strategy** combining **Collaborative Filtering (ALS)** and **Content-Based Filtering** to deliver relevant dining suggestions.

---

## 🚀 Features

* **Hybrid Recommendation Engine:** Weighted combination of user behavior (clicks/orders) and item similarity (cuisines/price).
* **Collaborative Filtering (ALS):** Uses Matrix Factorization to discover latent user preferences based on interaction history.
* **Content-Based Filtering:** Recommends items similar to a user's last order using Cosine Similarity on restaurant metadata.
* **Interactive Dashboard:** A clean Streamlit UI to visualize recommendations, inspect user history, and debug model scores.
* **Synthetic Data Generator:** Includes a script to simulate realistic user-restaurant interaction logs (Views, Carts, Orders).

---

## 🧠 System Architecture

The system operates on three tiers of logic:

1.  **The "Discovery" Layer (Collaborative Filtering):**
    * **Algorithm:** Alternating Least Squares (ALS) from the `implicit` library.
    * **Goal:** Find restaurants that similar users liked. "People who ordered from Domino's also ordered from Pizza Hut."
    * **Input:** Sparse Matrix of `(User, Item, Confidence)`.

2.  **The "Relevance" Layer (Content-Based):**
    * **Algorithm:** Cosine Similarity via `scikit-learn`.
    * **Goal:** Find restaurants with similar attributes to your last liked item. "You just ordered Biryani? Here are 5 other top-rated Biryani places."
    * **Features:** One-Hot Encoded Cuisines, Normalized Price, and Rating.

3.  **The Hybrid Engine:**
    * Combines the scores to balance discovery and relevance.
    * $$Score_{Hybrid} = 0.5 \times Score_{ALS} + 0.5 \times Score_{Content}$$

---

## 📂 Project Structure

```bash
swiggy-recommender/
│
├── app.py                   # The Streamlit Web Application (Serving Layer)
├── generate_data.py         # Script to simulate Users, Interactions, and Metadata
├── requirements.txt         # Python dependencies
├── README.md                # Project Documentation
│
└── data/                    # Data Storage (GitIgnore recommended for large files)
    ├── interactions.csv     # Simulated User-Item interactions
    ├── restaurants.csv      # Restaurant metadata (Zomato/Mock)
    ├── als_model.pkl        # Trained Collaborative Filtering Model
    └── content_model.pkl    # Trained Content-Based Model