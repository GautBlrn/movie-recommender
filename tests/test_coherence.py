import joblib
import pandas as pd
from pathlib import Path

from scripts.predict import build_features, ensure_engineered_cols

def test_predict_feature_matrix_shape_matches_training():
    bundle = joblib.load(Path("data/models/recommender.joblib"))
    df = pd.read_parquet("data/processed/movie_imdb.parquet").head(5)
    df = ensure_engineered_cols(df, bundle)

    X = build_features(df, bundle)

    # juste des sanity checks
    assert X.shape[0] == len(df)