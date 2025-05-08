'''
Эвристически афигистическиt
'''

import pandas as pd

# Рекомендации по популярности
def top_popular(interactions: pd.DataFrame, top_n=3):
    popular = interactions[interactions["interaction"] == "purchase"]
    return popular["item_id"].value_counts().head(top_n).index.tolist()

# С этим товаром также покупают
def also_bought(interactions: pd.DataFrame, item_id, top_n=3):
    co_purchase = interactions[interactions["interaction"] == "purchase"]
    co_purchase_users = co_purchase[co_purchase["item_id"] == item_id]["user_id"].unique()
    related = co_purchase[co_purchase["user_id"].isin(co_purchase_users)]
    related_items = related[related["item_id"] != item_id]["item_id"]
    return related_items.value_counts().head(top_n).index.tolist()
