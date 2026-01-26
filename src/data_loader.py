import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw dataset from local path.
    """
    df = pd.read_csv(path)
    return df


def initial_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop irrelevant columns and return cleaned dataframe.
    """
    cols_to_drop = [
        'budget','gross','movie_facebook_likes','cast_total_facebook_likes',
        'actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes',
        'director_facebook_likes','num_critic_for_reviews','num_user_for_reviews',
        'duration','Unnamed: 0'
    ]
    df = df.drop(cols_to_drop, axis=1, errors='ignore')
    return df