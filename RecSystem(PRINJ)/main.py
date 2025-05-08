from data_loader import load_data
from content_based import build_content_matrix, get_similar_items
from collaborative import item_based_recommendations
from evristika import top_popular, also_bought


def main():
    users, items, interactions = load_data()

    print("\n Рекоментации для первого итема:")
    tfidf, items = build_content_matrix(items)
    recs = get_similar_items(1, items, tfidf)
    print(recs[["item_id", "name"]])

    print("\n Item-Based Collaborative Filtering для первого юзера :")
    recs = item_based_recommendations(1, interactions)
    print(recs)

    print("\n Популярные товары:")
    print(top_popular(interactions))

    print("\n Так же покупают с первым товаром(Shoes, Nike):")
    print(also_bought(interactions, 1))


if __name__ == "__main__":
    main()
