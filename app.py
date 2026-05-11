"""
Streamlit App - Hybrid Movie Recommendation System
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --surface2: #1c1c27;
    --accent: #e8b86d;
    --accent2: #c084fc;
    --text: #f0f0f5;
    --muted: #8888aa;
    --border: #2a2a3a;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, #0a0a0f 0%, #1a0a2e 50%, #0a0a0f 100%);
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}
.hero p {
    color: var(--muted);
    font-size: 1rem;
    margin-top: 0.4rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}

/* Cards */
.movie-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.movie-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(var(--accent), var(--accent2));
    border-radius: 4px 0 0 4px;
}
.movie-card:hover {
    border-color: var(--accent);
    transform: translateX(3px);
}
.movie-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.3rem 0;
}
.movie-meta {
    font-size: 0.78rem;
    color: var(--muted);
    margin: 0;
}
.movie-score {
    position: absolute;
    top: 1rem; right: 1.1rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 1.1rem;
}

/* Genre badges */
.badge {
    display: inline-block;
    background: var(--surface2);
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.7rem;
    margin: 2px;
    font-weight: 500;
}

/* Metric cards */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-label {
    color: var(--muted);
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 900;
    color: var(--accent);
    margin: 0.2rem 0 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: var(--text) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1.5rem !important;
    letter-spacing: 0.05em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Inputs */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stSlider {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models ───────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


@st.cache_resource(show_spinner=False)
import os
import pickle
import streamlit as st
import pandas as pd
# Import your classes (ensure these match your file structure)
from models.collaborative import CollaborativeFilter
from models.content_based import ContentBasedFilter
from models.hybrid import HybridRecommender

@st.cache_resource
def load_models():
    # 1. Check if we already have the finished models
    if os.path.exists(os.path.join(MODELS_DIR, "hybrid_model.pkl")):
        try:
            with open(os.path.join(MODELS_DIR, "hybrid_model.pkl"), "rb") as f:
                return pickle.load(f) # ... and so on for other files
        except:
            pass # If loading fails, move to training

    # 2. FALLBACK: Train them live if files are missing/too big for GitHub
    st.info("Models not found on server. Training them now from CSV data...")
    
    # Load raw data (The CSVs should be < 25MB)
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")

    # Fit Content-Based
    cb = ContentBasedFilter()
    cb.fit(movies)

    # Fit Collaborative
    cf = CollaborativeFilter()
    cf.fit(ratings, movies)

    # Create Hybrid
    hybrid = HybridRecommender(cb, cf)

    st.success("Training complete! App is ready.")
    
    # Return everything the app expects
    # (Adjust this return statement to match how your app.py uses the variables)
    return hybrid, cb, cf, movies, ratings, {}

def render_movie_card(row, rank=None):
    # Safe extraction from pandas Series
    def get_val(r, key, default=""):
        try:
            val = r[key]
            return val if not pd.isna(val) else default
        except:
            return default

    title  = str(get_val(row, "title", "Unknown"))
    genres = str(get_val(row, "genres", ""))
    genre_list = [g.strip() for g in genres.split("|") if g.strip()]
    badges = "".join([f'<span class="badge">{g}</span>' for g in genre_list[:4]])

    score_val = 0
    for col in ["hybrid_score", "cf_score", "cb_score", "match_score"]:
        try:
            v = row[col]
            if v and not pd.isna(v):
                score_val = float(v)
                break
        except:
            continue

    score_pct = f"{score_val*100:.0f}%" if score_val else ""
    prefix = f"<span style='color:#555;font-size:0.85rem;margin-right:4px;'>#{rank}</span>" if rank else ""

    st.markdown(f"""
    <div class="movie-card">
        <p class="movie-title">{prefix}{title}</p>
        <div class="movie-score">{score_pct}</div>
        <p class="movie-meta">{badges}</p>
    </div>
    """, unsafe_allow_html=True)


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎬 CineMatch</h1>
    <p>Hybrid AI · Collaborative + Content-Based Filtering · MovieLens 100K</p>
</div>
""", unsafe_allow_html=True)

# ─── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading models..."):
    hybrid, cb, cf, movies, ratings, metrics = load_models()

if hybrid is None:
    st.error("⚠️ Models not found. Please run `python train.py` first to train and save the models.")
    st.code("cd movie_recommender\npython train.py", language="bash")
    st.stop()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    mode = st.radio(
        "Recommendation Mode",
        ["By User ID", "By Movie Title", "By Genre"],
        index=0
    )
    st.markdown("---")
    top_n = st.slider("Number of Recommendations", 5, 20, 10)
    st.markdown("---")

    cb_weight = st.slider("Content-Based Weight", 0.0, 1.0, 0.4, 0.05)
    cf_weight = round(1.0 - cb_weight, 2)
    st.caption(f"Collaborative Filtering Weight: **{cf_weight}**")
    hybrid.cb_weight = cb_weight
    hybrid.cf_weight = cf_weight

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.caption(f"Movies: **{len(movies):,}**")
    st.caption(f"Ratings: **{len(ratings):,}**")
    st.caption(f"Users: **{ratings['userId'].nunique():,}**")

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "📈 Evaluation Metrics", "🗂 Dataset Explorer"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 - RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    all_genres = sorted(set(
        g.strip()
        for genres in movies["genres"].dropna()
        for g in genres.split("|")
        if g.strip()
    ))
    all_titles = sorted(movies["title"].dropna().unique().tolist())
    all_user_ids = sorted(ratings["userId"].unique().tolist())

    col_left, col_right = st.columns([1, 1.3], gap="large")

    with col_left:
        st.markdown("#### 🔍 Your Input")

        if mode == "By User ID":
            user_id = st.selectbox("Select User ID", all_user_ids)
            user_ratings = ratings[ratings["userId"] == user_id]
            liked = user_ratings[user_ratings["rating"] >= 4]["movieId"].tolist()
            liked_movies = movies[movies["movieId"].isin(liked)][["title"]].head(5)

            st.markdown(f"**Previously liked movies ({len(liked)} total):**")
            if not liked_movies.empty:
                for t in liked_movies["title"]:
                    st.markdown(f"- {t}")
            else:
                st.caption("No highly-rated movies found.")

            if st.button("🎬 Get Recommendations"):
                with st.spinner("Generating hybrid recommendations..."):
                    recs = hybrid.recommend(user_id, user_ratings["movieId"].tolist(), top_n=top_n)
                st.session_state["recs"] = recs

        elif mode == "By Movie Title":
            selected_title = st.selectbox("Pick a movie you like", all_titles)
            movie_row = movies[movies["title"] == selected_title]
            if not movie_row.empty:
                movie_id = int(movie_row.iloc[0]["movieId"])
                genres = movie_row.iloc[0]["genres"]
                st.caption(f"Genres: **{genres}**")

            if st.button("🎬 Find Similar Movies"):
                with st.spinner("Finding similar movies..."):
                    recs = cb.get_similar_movies(movie_id, top_n=top_n)
                    recs = recs.rename(columns={"cb_score": "hybrid_score"})
                st.session_state["recs"] = recs

        elif mode == "By Genre":
            selected_genres = st.multiselect("Select Genres", all_genres, default=["Action", "Drama"])
            if st.button("🎬 Get Genre Recommendations"):
                with st.spinner("Finding movies..."):
                    recs = hybrid.recommend_by_genre(selected_genres, top_n=top_n)
                st.session_state["recs"] = recs

    with col_right:
        st.markdown("#### 🏆 Top Recommendations")
        if "recs" in st.session_state and st.session_state["recs"] is not None:
            recs = st.session_state["recs"]
            if recs.empty:
                st.warning("No recommendations found. Try different inputs.")
            else:
                for i, (_, row) in enumerate(recs.iterrows()):
                    render_movie_card(row, rank=i + 1)
        else:
            st.markdown("""
            <div style="
                background: #13131a;
                border: 1px dashed #2a2a3a;
                border-radius: 12px;
                padding: 3rem;
                text-align: center;
                color: #555;
            ">
                <div style="font-size:2.5rem;">🎞️</div>
                <p style="margin-top:1rem;font-size:0.9rem;">
                    Select your preferences and click <strong style="color:#8888aa">Get Recommendations</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 - EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 📊 Model Performance")
    if metrics:
        cf_r = metrics.get("CF_Rating", {})
        cf_rank = metrics.get("CF_Ranking", {})
        hybrid_rank = metrics.get("Hybrid_Ranking", {})

        st.markdown("**Rating Prediction (Collaborative Filtering)**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RMSE</div>
                <div class="metric-value">{cf_r.get('RMSE', 'N/A')}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MAE</div>
                <div class="metric-value">{cf_r.get('MAE', 'N/A')}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>**Ranking Metrics**", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("*Collaborative Filtering*")
            for k, v in cf_rank.items():
                if k != "Users Evaluated":
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:0.5rem">
                        <div class="metric-label">{k}</div>
                        <div class="metric-value">{v}</div>
                    </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("*Hybrid Model*")
            for k, v in hybrid_rank.items():
                if k != "Users Evaluated":
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:0.5rem">
                        <div class="metric-label">{k}</div>
                        <div class="metric-value">{v}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.caption(f"Evaluated on **{cf_rank.get('Users Evaluated', 'N/A')}** users with 80/20 train-test split.")
    else:
        st.info("Metrics will appear here after running `python train.py`.")

    st.markdown("---")
    st.markdown("#### 📋 Evaluation Summary")
    st.markdown("""
    | Metric | Description |
    |--------|-------------|
    | **RMSE** | Root Mean Square Error — lower is better |
    | **MAE** | Mean Absolute Error — lower is better |
    | **Precision@N** | Fraction of recommended movies that are relevant |
    | **Recall@N** | Fraction of relevant movies that were recommended |
    | **F1@N** | Harmonic mean of Precision and Recall |
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 - DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 🗂 Dataset Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Movies</div>
            <div class="metric-value">{len(movies):,}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Ratings</div>
            <div class="metric-value">{len(ratings):,}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Unique Users</div>
            <div class="metric-value">{ratings['userId'].nunique():,}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Sample Movies**")
        sample = movies[["movieId", "title", "genres"]].head(20)
        st.dataframe(sample, use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**Sample Ratings**")
        sample_r = ratings[["userId", "movieId", "rating"]].head(20)
        st.dataframe(sample_r, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Rating Distribution**")
    dist = ratings["rating"].value_counts().sort_index()
    st.bar_chart(dist)
