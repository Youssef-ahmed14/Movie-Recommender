"""
Hybrid Recommendation Engine
Combines Content-Based and Collaborative Filtering using weighted average
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class HybridRecommender:
    def __init__(self, cb_model, cf_model, cb_weight: float = 0.4, cf_weight: float = 0.6):
        """
        cb_model: fitted ContentBasedFilter
        cf_model: fitted CollaborativeFilter
        cb_weight: weight for content-based scores (0-1)
        cf_weight: weight for collaborative scores (0-1)
        Weights should sum to 1.
        """
        assert abs(cb_weight + cf_weight - 1.0) < 1e-5, "Weights must sum to 1"
        self.cb = cb_model
        self.cf = cf_model
        self.cb_weight = cb_weight
        self.cf_weight = cf_weight

    def recommend(self, user_id: int, liked_movie_ids: list, top_n: int = 10) -> pd.DataFrame:
        """
        Generate hybrid recommendations for a user.
        user_id: int
        liked_movie_ids: list of movieIds the user has rated/liked
        top_n: number of recommendations
        """
        # Get more candidates than needed so merging keeps enough
        candidate_n = max(top_n * 5, 50)

        # Content-based recs based on liked movies
        cb_recs = self.cb.get_user_recommendations(liked_movie_ids, top_n=candidate_n)

        # Collaborative filtering recs
        cf_recs = self.cf.get_user_recommendations(user_id, liked_movie_ids, top_n=candidate_n)

        if cb_recs.empty and cf_recs.empty:
            return pd.DataFrame()

        # Normalize scores to [0, 1]
        scaler = MinMaxScaler()

        if not cb_recs.empty and len(cb_recs) > 1:
            cb_recs["cb_score_norm"] = scaler.fit_transform(cb_recs[["cb_score"]])
        elif not cb_recs.empty:
            cb_recs["cb_score_norm"] = 1.0

        if not cf_recs.empty and len(cf_recs) > 1:
            cf_recs["cf_score_norm"] = scaler.fit_transform(cf_recs[["cf_score"]])
        elif not cf_recs.empty:
            cf_recs["cf_score_norm"] = 1.0

        # Merge on movieId
        if cb_recs.empty:
            merged = cf_recs.copy()
            merged["cb_score_norm"] = 0.0
        elif cf_recs.empty:
            merged = cb_recs.copy()
            merged["cf_score_norm"] = 0.0
        else:
            merged = pd.merge(
                cb_recs[["movieId", "cb_score_norm"]],
                cf_recs[["movieId", "cf_score_norm"]],
                on="movieId",
                how="outer"
            ).fillna(0)
            # Re-attach title and genres from movies DataFrame
            movies_info = self.cf.movies[["movieId", "title", "genres"]]
            merged = pd.merge(merged, movies_info, on="movieId", how="left")

        # Weighted hybrid score
        merged["hybrid_score"] = (
            self.cb_weight * merged["cb_score_norm"] +
            self.cf_weight * merged["cf_score_norm"]
        )

        # Sort and return top_n
        result = merged.sort_values("hybrid_score", ascending=False).head(top_n)
        return result[["movieId", "title", "genres", "cb_score_norm", "cf_score_norm", "hybrid_score"]].reset_index(drop=True)

    def recommend_by_genre(self, genres: list, user_id: int = None, top_n: int = 10) -> pd.DataFrame:
        """
        Recommend movies by genre for new/anonymous users.
        Falls back to content-based only.
        """
        movies = self.cf.movies
        # Filter movies that contain at least one of the given genres
        mask = movies["genres"].apply(
            lambda g: any(genre.lower() in g.lower() for genre in genres)
        )
        filtered = movies[mask].copy()
        if filtered.empty:
            return pd.DataFrame()

        # Score by number of matching genres
        filtered["match_score"] = filtered["genres"].apply(
            lambda g: sum(genre.lower() in g.lower() for genre in genres)
        )
        result = filtered.sort_values("match_score", ascending=False).head(top_n)
        result["hybrid_score"] = result["match_score"] / max(result["match_score"])
        return result[["movieId", "title", "genres", "hybrid_score"]].reset_index(drop=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.preprocess import load_data, preprocess
    from models.content_based import ContentBasedFilter
    from models.collaborative import CollaborativeFilter

    ratings, movies = load_data()
    df, ratings, movies = preprocess(ratings, movies)

    cb = ContentBasedFilter()
    cb.fit(movies)

    cf = CollaborativeFilter()
    cf.fit(ratings, movies)

    hybrid = HybridRecommender(cb, cf, cb_weight=0.4, cf_weight=0.6)

    user_id = 1
    liked = ratings[ratings["userId"] == user_id]["movieId"].tolist()
    print(f"Hybrid recs for user {user_id}:")
    print(hybrid.recommend(user_id, liked, top_n=10))
