"""
Content-Based Recommender
"""


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_content_matrix(items: pd.DataFrame):
    items["combined"] = items["category"] + " " + items["brand"]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(items["combined"])
    return tfidf, items

def get_similar_items(item_id, items, tfidf, top_n=3):
    idx = items[items["item_id"] == item_id].index[0]
    sims = cosine_similarity(tfidf[idx], tfidf).flatten()
    sim_scores = list(enumerate(sims))
    sim_scores = sorted(sim_scores, key=lambda x: -x[1])
    top_indices = [i for i, _ in sim_scores[1:top_n+1]]
    return items.iloc[top_indices]
