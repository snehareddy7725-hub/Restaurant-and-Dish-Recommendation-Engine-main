"""Streamlit demo UI. Uses the same Recommender as the FastAPI service.

Run:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

from recommender import config
from recommender.engine import Recommender

st.set_page_config(page_title="Hybrid Restaurant Recommender", layout="wide")


@st.cache_resource
def load_recommender():
    return Recommender()


reco = load_recommender()

MODE_LABELS = {
    "hybrid": "🚀 Hybrid (ALS + content)",
    "als_only": "🔹 ALS only (no usable content seed)",
    "content_only": "🔸 Content only (user not yet in ALS model)",
    "popularity": "⭐ Popularity fallback (no history)",
}


def name_of(rid):
    return reco.restaurants.loc[rid, "name"] if rid in reco.restaurants.index else f"#{rid}"


st.title("🍔 Restaurant and Dish Recommendation Engine")
st.markdown("### Hybrid system: Collaborative Filtering (ALS) + Content-Based, with cold-start fallback")

# ---------------- sidebar
user_ids = sorted(reco.history.keys())
choice = st.sidebar.radio("User", ["Existing user", "Brand-new user (cold start)"])
if choice == "Existing user":
    user_id = st.sidebar.selectbox("Select user ID (simulated)", user_ids)
    cuisine = None
else:
    user_id = st.sidebar.number_input("New user ID", value=10_000_000, step=1)
    cuisines = sorted(reco.restaurants["cuisines"].unique())
    cuisine = st.sidebar.selectbox("Preferred cuisine (optional)", ["(none)"] + cuisines)
    cuisine = None if cuisine == "(none)" else cuisine

st.sidebar.subheader("History (last 3)")
hist = reco.history.get(user_id, [])
if hist:
    for rid in hist[-3:]:
        st.sidebar.text(f"• {name_of(rid)}")
else:
    st.sidebar.warning("No history.")

# ---------------- recommendations
rec = reco.recommend(user_id, n=10, cuisine=cuisine)
st.info(f"**Mode selected:** {MODE_LABELS[rec.mode]}")

if rec.mode == "hybrid":
    st.caption(f"Score = {config.ALS_WEIGHT} × ALS + {config.CONTENT_WEIGHT} × content "
               f"(seed: last interaction `{name_of(hist[-1])}`)")
    chart = pd.DataFrame(
        [{"Restaurant": i["name"],
          "Behavioural match (ALS)": config.ALS_WEIGHT * i["als_score"],
          "Content match (similarity)": config.CONTENT_WEIGHT * i["content_score"]}
         for i in rec.items[:5]]
    ).set_index("Restaurant")
    st.markdown("#### 📊 Why these were recommended (top 5)")
    st.bar_chart(chart, height=300)

st.markdown("#### Recommendations")
for i in rec.items:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"**{i['name']}**")
        st.text(f"{i['cuisines']} | ₹{i['avg_cost_for_two']} for two | ★ {i['rating']}")
    with c2:
        st.metric("Score", f"{i['score']:.2f}")
    st.divider()
