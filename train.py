import sys
import os
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data.preprocess import load_data, preprocess, train_test_split_data
from models.content_based import ContentBasedFilter
from models.collaborative import CollaborativeFilter
from models.hybrid import HybridRecommender

def train_and_save():
    print("=" * 50)
    print("HYBRID MOVIE RECOMMENDER - TRAINING PIPELINE")
    print("=" * 50)

    print("\n[Step 1] Loading and preprocessing data...")
    ratings, movies, tags = load_data()
    ratings, movies = preprocess(ratings, movies, tags)
    train_ratings, test_ratings = train_test_split_data(ratings, test_size=0.2)

    print("\n[Step 2] Training Content-Based Filter...")
    cb = ContentBasedFilter()
    cb.fit(movies)

    print("\n[Step 3] Training Collaborative Filtering (SVD)...")
    cf = CollaborativeFilter(n_factors=50, n_epochs=20)
    cf.fit(train_ratings, movies)

    print("\n[Step 4] Building Hybrid Recommender...")
    hybrid = HybridRecommender(cb, cf, cb_weight=0.4, cf_weight=0.6)

    print("\n[Step 5] Running Evaluation...")
    sample_test = test_ratings.sample(n=min(500, len(test_ratings)), random_state=42)
    cf_metrics = cf.evaluate(sample_test)
    print(f"  CF RMSE: {cf_metrics['RMSE']}  MAE: {cf_metrics['MAE']}")

    metrics = {"CF_Rating": cf_metrics}

    print("\n[Step 6] Saving models...")
    models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(models_dir, "cb_model.pkl"), "wb") as f: pickle.dump(cb, f)
    with open(os.path.join(models_dir, "cf_model.pkl"), "wb") as f: pickle.dump(cf, f)
    with open(os.path.join(models_dir, "hybrid_model.pkl"), "wb") as f: pickle.dump(hybrid, f)
    with open(os.path.join(models_dir, "movies.pkl"), "wb") as f: pickle.dump(movies, f)
    with open(os.path.join(models_dir, "ratings.pkl"), "wb") as f: pickle.dump(ratings, f)
    with open(os.path.join(models_dir, "metrics.pkl"), "wb") as f: pickle.dump(metrics, f)

    print("\nAll models saved to saved_models/")
    print("\nDone! Now run: streamlit run app.py")
    return hybrid, metrics
if __name__ == "__main__":
    train_and_save()