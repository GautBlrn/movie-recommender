import pandas as pd

def split_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split the 'genres' column into multiple genre columns.
    """
    genres_df = df["genres"].str.split("|", expand=True)
    genres_df.columns = [f"genre_{i+1}" for i in range(genres_df.shape[1])]
    df = pd.concat([df, genres_df], axis=1, sort=False)
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
