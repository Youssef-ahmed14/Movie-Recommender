# 🎬 CineMatch — Hybrid Movie Recommendation System

A production-ready hybrid recommendation system combining **Collaborative Filtering (SVD)** and **Content-Based Filtering (TF-IDF + Cosine Similarity)** on the MovieLens 100K dataset.

---

## 📁 Project Structure

```
movie_recommender/
│
├── data/
│   └── preprocess.py          # Data loading, cleaning, and preprocessing
│
├── models/
│   ├── content_based.py       # TF-IDF + Cosine Similarity (Content-Based)
│   ├── collaborative.py       # SVD Matrix Factorization (Collaborative)
│   └── hybrid.py              # Hybrid engine (Weighted Average)
│
├── utils/
│   └── evaluation.py          # RMSE, MAE, Precision, Recall, F1
│
├── saved_models/              # Auto-created after training
│
├── train.py                   # Main training pipeline
├── app.py                     # Streamlit UI
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all models
```bash
python train.py
```
This will:
- Automatically download the MovieLens 100K dataset
- Train Content-Based and Collaborative Filtering models
- Build the Hybrid Recommender
- Run full evaluation and print metrics
- Save all models to `saved_models/`

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 🧠 System Architecture

### Content-Based Filtering
- Extracts movie genre features using **TF-IDF Vectorization**
- Computes **Cosine Similarity** between all movie genre vectors
- Recommends movies similar to a user's liked history

### Collaborative Filtering
- Uses **Singular Value Decomposition (SVD)** via the `scikit-surprise` library
- Learns latent user and item factors from historical ratings
- Predicts unseen movie ratings for each user

### Hybrid Engine
- Normalizes scores from both models to [0, 1] using Min-Max scaling
- Combines with a **Weighted Average**:
  ```
  hybrid_score = 0.4 × cb_score + 0.6 × cf_score
  ```
- Weights are tunable in real-time via the Streamlit sidebar

---

## 📊 Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **RMSE** | Rating prediction error (lower = better) |
| **MAE** | Rating prediction error (lower = better) |
| **Precision@N** | % of recommended items that are relevant |
| **Recall@N** | % of relevant items that were recommended |
| **F1@N** | Balance between Precision and Recall |

Evaluation uses an 80/20 train-test split on the rating data.

---

## 🎛 Features

- **3 Recommendation Modes**:
  1. By User ID — personalized recommendations using full hybrid model
  2. By Movie Title — content-based similar movie finder
  3. By Genre — genre-based discovery for new users

- **Adjustable Weights** — tune CB vs CF balance from the sidebar
- **Interactive UI** — built with Streamlit, dark cinema theme
- **Full Evaluation Report** — RMSE, MAE, Precision, Recall, F1

---

## 📦 Dataset

- **MovieLens 100K** — automatically downloaded from GroupLens
- 100,000 ratings from 943 users on 1,682 movies
- Ratings scale: 1–5

---

## 🔧 Dependencies

```
pandas, numpy, scikit-learn, scikit-surprise, streamlit, requests
```
