import numpy as np

def recommend_titles(movie_title, model, feature_matrix, n_recs=5):
    """
    Return top-n recommendations for a given movie.
    """
    movie_title = movie_title.strip()
    if movie_title not in feature_matrix.index:
        raise ValueError(f"Movie not found: '{movie_title}'")

    distances, indices = model.kneighbors(
        feature_matrix.loc[[movie_title]],
        n_neighbors=n_recs+1
    )

    # Skip first (movie itself)
    return feature_matrix.iloc[indices[0][1:]].index.tolist()