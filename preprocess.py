import pandas as pd
import numpy as np
import os

def load_data():
    data_dir = os.path.dirname(__file__)
    ratings_path = os.path.join(data_dir, "ratings.csv")
    movies_path  = os.path.join(data_dir, "movies.csv")
    tags_path    = os.path.join(data_dir, "tags.csv")

    if not all(os.path.exists(p) for p in [ratings_path, movies_path, tags_path]):
        raise FileNotFoundError("Required CSV files (ratings, movies, tags) are missing.")

    ratings = pd.read_csv(ratings_path)
    movies  = pd.read_csv(movies_path)
    tags    = pd.read_csv(tags_path)
    
    return ratings, movies, tags

def preprocess(ratings, movies, tags):
    movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    
    movies = movies.merge(movie_tags, on='movieId', how='left')
    
    movies['tag'] = movies['tag'].fillna('')
    
    movies['metadata'] = movies['genres'] + " " + movies['tag']
    
    return ratings, movies

def train_test_split_data(ratings, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)
    return train, test

if __name__ == "__main__":
    ratings, movies, tags = load_data()
    ratings, movies = preprocess(ratings, movies, tags)
    train, test = train_test_split_data(ratings)
    
    print(f"Loaded {len(ratings):,} ratings and {len(movies):,} movies.")
    print(f"Preprocessed features created in 'metadata' column.")
    print(movies[['title', 'metadata']].head())