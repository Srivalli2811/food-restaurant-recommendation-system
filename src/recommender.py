from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(data_dir="data"):
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "restaurants.csv"), pd.read_csv(data_dir / "user_interactions.csv")

def build_content_model(restaurants):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(restaurants["tags"].fillna(""))
    return vectorizer, matrix, cosine_similarity(matrix)

def popularity_scores(interactions):
    return interactions.groupby("restaurant_id")["weight"].sum().sort_values(ascending=False)

def content_recommend(restaurants, similarity, restaurant_id, top_k=10):
    ids = restaurants["restaurant_id"].tolist()
    idx = ids.index(restaurant_id)
    scores = similarity[idx]
    ranked = [i for i in scores.argsort()[::-1] if i != idx][:top_k]
    out = restaurants.iloc[ranked].copy()
    out["similarity_score"] = scores[ranked]
    return out

def user_profile_recommend(restaurants, interactions, similarity, user_id, top_k=10):
    ids = restaurants["restaurant_id"].tolist()
    hist = interactions[interactions.user_id == user_id]
    if hist.empty:
        pop = popularity_scores(interactions).head(top_k)
        return restaurants[restaurants.restaurant_id.isin(pop.index)].copy()
    score = pd.Series(0.0, index=range(len(restaurants)))
    for _, row in hist.iterrows():
        idx = ids.index(int(row.restaurant_id))
        score += similarity[idx] * float(row.weight)
    seen = set(hist.restaurant_id.astype(int))
    for i, rid in enumerate(ids):
        if rid in seen:
            score.iloc[i] = -1
    top_idx = score.sort_values(ascending=False).head(top_k).index
    out = restaurants.iloc[top_idx].copy()
    out["recommendation_score"] = score.iloc[top_idx].values
    return out
