import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
import pickle

class CollaborativeFilter:
    def __init__(self, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42):
        self.model = SVD(
            n_factors=n_factors, 
            n_epochs=n_epochs, 
            lr_all=lr_all, 
            reg_all=reg_all, 
            random_state=random_state
        )
        self.movies = None
        self.trainset = None

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.movies = movies
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)

    def predict_rating(self, user_id, movie_id) -> float:
        pred = self.model.predict(uid=user_id, iid=movie_id)
        return float(pred.est)

    def get_user_recommendations(self, user_id, rated_movie_ids: list, top_n: int = 10) -> pd.DataFrame:
        all_movie_ids = self.movies["movieId"].unique()
        unseen = [mid for mid in all_movie_ids if mid not in rated_movie_ids]
        
        predictions = [(mid, self.predict_rating(user_id, mid)) for mid in unseen]
        predictions.sort(key=lambda x: x[1], reverse=True)
        top = predictions[:top_n]
        
        movie_ids = [m for m, _ in top]
        cf_scores = [s for _, s in top]
        
        result = self.movies[self.movies["movieId"].isin(movie_ids)].copy()
        score_df = pd.DataFrame({"movieId": movie_ids, "cf_score": cf_scores})
        
        result = result.merge(score_df, on="movieId")
        return result.sort_values("cf_score", ascending=False).reset_index(drop=True)

    def evaluate(self, test_ratings: pd.DataFrame) -> dict:
        test_data = list(test_ratings[['userId', 'movieId', 'rating']].itertuples(index=False, name=None))
        predictions = self.model.test(test_data)
        
        rmse = round(accuracy.rmse(predictions, verbose=False), 4)
        mae = round(accuracy.mae(predictions, verbose=False), 4)
        
        return {"RMSE": rmse, "MAE": mae}

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)