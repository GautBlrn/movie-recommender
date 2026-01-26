import pandas as pd

def split_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split the 'genres' column into multiple genre columns.
    """
    genres_df = df["genres"].str.split("|", expand=True)
    genres_df.columns = [f"genre_{i+1}" for i in range(genres_df.shape[1])]
    df = pd.concat([df, genres_df], axis=1, sort=False)
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Categorical
    df["content_rating"] = df["content_rating"].fillna("Not Rated")
    df["language"] = df["language"].fillna("Unknown")
    df["country"] = df["country"].fillna("Unknown")
    df["color"] = df["color"].fillna("Color")
    df["director_name"] = df["director_name"].fillna("Unknown")

    for c in ["actor_1_name", "actor_2_name", "actor_3_name"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")

    # Numeric
    if "title_year" in df.columns:
        df["title_year"] = df["title_year"].fillna(df["title_year"].median())

    if "imdb_score" in df.columns:
        df["imdb_score"] = df["imdb_score"].fillna(df["imdb_score"].median())

    if "num_voted_users" in df.columns:
        df["num_voted_users"] = df["num_voted_users"].fillna(0)

    return df


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns relevant for modeling.
    """
    df2 = df.copy()
    df2 = df2.drop(['movie_imdb_link', 'plot_keywords'], axis=1, errors="ignore")
    df2 = df2[
        [
            'title_year','movie_title','genre_1','genre_2','genre_3','genre_4',
            'genre_5','genre_6','genre_7','genre_8','content_rating','imdb_score',
            'language','country','color','director_name','actor_1_name',
            'actor_2_name','actor_3_name','num_voted_users'
        ]
    ]
    return df2
