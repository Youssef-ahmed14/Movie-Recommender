"""
Evaluation Module
Computes RMSE, MAE, Precision, Recall, F1-Score
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_rmse(y_true, y_pred):
    return round(np.sqrt(mean_squared_error(y_true, y_pred)), 4)


def compute_mae(y_true, y_pred):
    return round(mean_absolute_error(y_true, y_pred), 4)


def compute_precision_recall_f1(recommended_ids: list, relevant_ids: list):
    """
    Compute Precision, Recall, F1 for a single user.
    recommended_ids: list of recommended movie IDs
    relevant_ids: list of actually liked/highly rated movie IDs
    """
    if not recommended_ids or not relevant_ids:
        return 0.0, 0.0, 0.0

    rec_set = set(recommended_ids)
    rel_set = set(relevant_ids)

    tp = len(rec_set & rel_set)
    precision = tp / len(rec_set) if rec_set else 0.0
    recall = tp / len(rel_set) if rel_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return round(precision, 4), round(recall, 4), round(f1, 4)


def evaluate_cf_model(cf_model, test_ratings: pd.DataFrame) -> dict:
    """
    Evaluate collaborative filter on test ratings.
    Returns RMSE and MAE.
    """
    y_true, y_pred = [], []
    for _, row in test_ratings.iterrows():
        pred = cf_model.predict_rating(int(row["userId"]), int(row["movieId"]))
        y_true.append(row["rating"])
        y_pred.append(pred)

    return {
        "RMSE": compute_rmse(y_true, y_pred),
        "MAE": compute_mae(y_true, y_pred)
    }


def evaluate_ranking(model_fn, ratings: pd.DataFrame, threshold: float = 4.0,
                     top_n: int = 10, sample_users: int = 50) -> dict:
    """
    Evaluate recommendation ranking (Precision@N, Recall@N, F1@N).
    model_fn: function(user_id, liked_ids) -> DataFrame with movieId column
    ratings: full ratings DataFrame
    threshold: min rating to consider a movie 'relevant'
    sample_users: number of users to evaluate on
    """
    users = ratings["userId"].unique()
    np.random.seed(42)
    sample = np.random.choice(users, size=min(sample_users, len(users)), replace=False)

    precisions, recalls, f1s = [], [], []

    for user_id in sample:
        user_ratings = ratings[ratings["userId"] == user_id]
        # Split: use 80% as known, 20% as ground truth
        known = user_ratings.sample(frac=0.8, random_state=42)
        held_out = user_ratings.drop(known.index)

        liked_ids = known["movieId"].tolist()
        relevant_ids = held_out[held_out["rating"] >= threshold]["movieId"].tolist()

        if not relevant_ids:
            continue

        try:
            recs = model_fn(user_id, liked_ids)
            if recs is None or recs.empty:
                continue
            rec_ids = recs["movieId"].tolist()[:top_n]
        except Exception:
            continue

        p, r, f = compute_precision_recall_f1(rec_ids, relevant_ids)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        f"Precision@{top_n}": round(np.mean(precisions), 4) if precisions else 0.0,
        f"Recall@{top_n}": round(np.mean(recalls), 4) if recalls else 0.0,
        f"F1@{top_n}": round(np.mean(f1s), 4) if f1s else 0.0,
        "Users Evaluated": len(precisions)
    }


def full_evaluation_report(cf_model, hybrid_model, train_ratings, test_ratings, top_n=10):
    """
    Generate a complete evaluation report for both CF and Hybrid models.
    """
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)

    # CF rating prediction metrics
    print("\n[1] Collaborative Filtering - Rating Prediction")
    cf_metrics = evaluate_cf_model(cf_model, test_ratings)
    for k, v in cf_metrics.items():
        print(f"  {k}: {v}")

    # CF ranking metrics
    print("\n[2] Collaborative Filtering - Ranking Metrics")
    cf_rank = evaluate_ranking(
        lambda uid, lids: cf_model.get_user_recommendations(uid, lids, top_n=top_n),
        train_ratings, top_n=top_n
    )
    for k, v in cf_rank.items():
        print(f"  {k}: {v}")

    # Hybrid ranking metrics
    print("\n[3] Hybrid Model - Ranking Metrics")
    hybrid_rank = evaluate_ranking(
        lambda uid, lids: hybrid_model.recommend(uid, lids, top_n=top_n),
        train_ratings, top_n=top_n
    )
    for k, v in hybrid_rank.items():
        print(f"  {k}: {v}")

    print("\n" + "="*50)

    return {
        "CF_Rating": cf_metrics,
        "CF_Ranking": cf_rank,
        "Hybrid_Ranking": hybrid_rank
    }
