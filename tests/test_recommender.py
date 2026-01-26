import pandas as pd
from src.features import build_feature_matrix
from src.model import train_knn
from src.recommender import recommend_titles

def test_recommend_basic():
    # Petit échantillon factice
    data = {
        "movie_title": ["A", "B", "C"],
        "genre_1": ["Action", "Action", "Drama"],
        "country": ["USA", "USA", "USA"],
        "language": ["EN", "EN", "EN"],
        "content_rating": ["PG", "PG", "PG"],
        "title_year": [2000, 2001, 2002],
        "num_voted_users": [100, 200, 300]
    }
    df = pd.DataFrame(data)

    features = build_feature_matrix(df)
    model = train_knn(features)

    # On prend la 1re ligne => on doit au moins obtenir une liste de titres
    recs = recommend_titles("A", model, features, n_recs=2)
    assert isinstance(recs, list)
    assert len(recs) == 2
