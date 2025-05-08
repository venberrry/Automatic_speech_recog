'''
Collaborative Filtering (Item-Based)
'''

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def build_user_item_matrix(interactions: pd.DataFrame):
    matrix = interactions.pivot_table(index='user_id', columns='item_id',
                                      aggfunc=lambda x: 1, fill_value=0)
    return matrix


def item_based_recommendations(user_id, interactions, top_n=3):
    matrix = build_user_item_matrix(interactions)
    item_sim = cosine_similarity(matrix.T)
    item_sim_df = pd.DataFrame(item_sim, index=matrix.columns, columns=matrix.columns)

    user_row = matrix.loc[user_id]
    purchased_items = user_row[user_row > 0].index.tolist()

    scores = item_sim_df[purchased_items].sum(axis=1)
    scores = scores.drop(labels=purchased_items)
    top_items = scores.sort_values(ascending=False).head(top_n).index.tolist()
    return top_items
