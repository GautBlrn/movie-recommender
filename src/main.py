from pathlib import Path
import pandas as pd

from src.data_loader import load_raw_data, initial_cleaning
from src.preprocessing import split_genres, select_final_columns
from src.features import build_feature_matrix
from src.model import train_knn
from src.recommender import recommend_titles

DATA_PATH = Path("../data/raw/movie_metadata.csv")

def run_recommender():
    df = load_raw_data(DATA_PATH)
    df = initial_cleaning(df)
    df = split_genres(df)
    df = select_final_columns(df)

    feature_matrix = build_feature_matrix(df)
    model = train_knn(feature_matrix)

    movie = input("🎬 Dernier film vu ? ")
    try:
        recos = recommend_titles(movie, model, feature_matrix)
        print("\n🎯 Recommandations :")
        for i, title in enumerate(recos, 1):
            print(f"{i}. {title}")
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    run_recommender()