"""
Content-Based Filtering Module
Uses TF-IDF on movie genres + cosine similarity
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedFilter:
    def __init__(self):
        self.tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
        self.tfidf_matrix = None
        self.movies = None
        self.similarity_matrix = None
        self.movie_index = None  

    def fit(self, movies: pd.DataFrame):
        """
        Fit TF-IDF on movie genres.
        movies: DataFrame with columns [movieId, title, genres]
        genres are pipe-separated, e.g. "Action|Comedy|Drama"
        """
        self.movies = movies.reset_index(drop=True)
        genres = self.movies["genres"].fillna("Unknown")
        # Replace | with space so TF-IDF treats each genre as a token
        genres_clean = genres.str.replace("|", " ", regex=False)
        self.tfidf_matrix = self.tfidf.fit_transform(genres_clean)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        self.movie_index = {mid: idx for idx, mid in enumerate(self.movies["movieId"])}
        print(f"ContentBasedFilter fitted on {len(self.movies)} movies.")

    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        """Return top_n movies most similar to movie_id."""
        if movie_id not in self.movie_index:
            return pd.DataFrame()
        idx = self.movie_index[movie_id]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Exclude the movie itself
        sim_scores = [(i, s) for i, s in sim_scores if i != idx][:top_n]
        indices = [i for i, _ in sim_scores]
        scores = [s for _, s in sim_scores]
        result = self.movies.iloc[indices][["movieId", "title", "genres"]].copy()
        result["cb_score"] = scores
        return result.reset_index(drop=True)

    def get_user_recommendations(self, liked_movie_ids: list, top_n: int = 10) -> pd.DataFrame:
        """
        Given a list of liked movie IDs, aggregate similarity scores
        and return top_n recommendations (excluding already liked movies).
        """
        score_map = {}
        for movie_id in liked_movie_ids:
            if movie_id not in self.movie_index:
                continue
            idx = self.movie_index[movie_id]
            sim_scores = self.similarity_matrix[idx]
            for i, score in enumerate(sim_scores):
                mid = self.movies.iloc[i]["movieId"]
                if mid not in liked_movie_ids:
                    score_map[mid] = score_map.get(mid, 0) + score

        if not score_map:
            return pd.DataFrame()

        sorted_movies = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_n]
        movie_ids = [m for m, _ in sorted_movies]
        scores = [s for _, s in sorted_movies]

        result = self.movies[self.movies["movieId"].isin(movie_ids)].copy()
        score_df = pd.DataFrame({"movieId": movie_ids, "cb_score": scores})
        result = result.merge(score_df, on="movieId")
        result = result.sort_values("cb_score", ascending=False).reset_index(drop=True)
        return result[["movieId", "title", "genres", "cb_score"]]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.preprocess import load_data, preprocess

    ratings, movies = load_data()
    df, ratings, movies = preprocess(ratings, movies)

    cb = ContentBasedFilter()
    cb.fit(movies)

    # Test: similar to movie 1 (Toy Story)
    print("Movies similar to Toy Story (id=1):")
    print(cb.get_similar_movies(1, top_n=5))
