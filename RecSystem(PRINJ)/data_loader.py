import os
import pandas as pd

def load_data():
    base_path = os.path.join(os.path.dirname(__file__), "datas")

    users_path = os.path.join(base_path, "users.csv")
    items_path = os.path.join(base_path, "items.csv")
    interactions_path = os.path.join(base_path, "interactions.csv")

    users = pd.read_csv(users_path) if os.path.exists(users_path) else None
    items = pd.read_csv(items_path) if os.path.exists(items_path) else None
    interactions = pd.read_csv(interactions_path) if os.path.exists(interactions_path) else None

    return users, items, interactions
