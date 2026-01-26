import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def encode_years(df: pd.DataFrame) -> pd.DataFrame:
    enc = OneHotEncoder()
    pe = pd.DataFrame(
        enc.fit_transform(pd.cut(df.title_year, 10).astype(str).values.reshape(-1,1)).toarray(),
        columns=enc.get_feature_names_out()
    )
    pe.columns = [c.replace('x0_', 'Year_') for c in pe.columns]
    return pd.concat([df[['movie_title']], pe], axis=1)


def encode_categorical(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    enc = OneHotEncoder()
    mat = pd.DataFrame(
        enc.fit_transform(df[col].values.reshape(-1,1)).toarray(),
        columns=enc.get_feature_names_out()
    )
    mat.columns = [c.replace('x0_', prefix) for c in mat.columns]
    return pd.concat([df[['movie_title']], mat], axis=1)


def scale_votes(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df[['num_voted_users']]),
        columns=['num_voted_users']
    )
    return pd.concat([df[['movie_title']], scaled], axis=1)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final numeric feature matrix for modeling.
    """
    year = encode_years(df)
    genre = encode_categorical(df, 'genre_1', 'Genre1_')
    country = encode_categorical(df, 'country', 'Country_')
    language = encode_categorical(df, 'language', 'Lang_')
    content_rating = encode_categorical(df.fillna({'content_rating':'Not Rated'}), 'content_rating', 'CR_')
    votes = scale_votes(df)

    # Merge progressively
    merged = df.merge(genre, on='movie_title')
    merged = merged.merge(country, on='movie_title')
    merged = merged.merge(language, on='movie_title')
    merged = merged.merge(content_rating, on='movie_title')
    merged = merged.merge(year, on='movie_title')
    merged = merged.merge(votes, on='movie_title')

    merged.set_index('movie_title', inplace=True)
    merged.index = merged.index.str.strip()

    # Drop non-numeric and index columns
    obj_cols = merged.select_dtypes(include='object').columns
    return merged.drop(columns=obj_cols)