from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path
from itertools import combinations

import pandas as pd
from tqdm import tqdm


NA = r"\N"
BAD_GENRES = {"News", "Reality-TV", "Talk-Show"}


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def clean_genres_str(genres: pd.Series) -> pd.Series:
    g = genres.fillna("").astype(str).str.replace(",", " ", regex=False)
    bad_pattern = r"\b(?:News|Reality-TV|Talk-Show)\b"
    cleaned = g.str.replace(bad_pattern, "", regex=True)
    return cleaned.str.replace(r"\s+", " ", regex=True).str.strip()


def estimate_total_chunks(path: Path, approx_mb_per_chunk: int = 60) -> int:
    size_bytes = os.path.getsize(path)
    chunk_bytes = approx_mb_per_chunk * 1024 * 1024
    return max(1, (size_bytes + chunk_bytes - 1) // chunk_bytes)


# -------------------------
# BASICS
# -------------------------
def read_movies_basics(basics_path: Path) -> pd.DataFrame:
    usecols = [
        "tconst",
        "titleType",
        "primaryTitle",
        "originalTitle",
        "isAdult",
        "startYear",
        "runtimeMinutes",
        "genres",
    ]

    total = estimate_total_chunks(basics_path, approx_mb_per_chunk=60)
    chunks = []
    for chunk in tqdm(
        pd.read_csv(
            basics_path,
            sep="\t",
            usecols=usecols,
            dtype=str,
            na_values=NA,
            keep_default_na=False,
            chunksize=800_000,
            low_memory=False,
            on_bad_lines="skip",
        ),
        desc="Loading basics",
        total=total,
        unit="chunk",
    ):
        chunk["isAdult"] = clean_numeric(chunk["isAdult"]).fillna(0).astype("Int64")
        chunk["startYear"] = clean_numeric(chunk["startYear"]).astype("Int64")
        chunk["runtimeMinutes"] = clean_numeric(chunk["runtimeMinutes"]).astype("Int64")

        chunk = chunk[(chunk["titleType"] == "movie") & (chunk["isAdult"].fillna(0) == 0)]
        chunk = chunk.drop(columns=["titleType"])
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True, copy=False)


# -------------------------
# RATINGS
# -------------------------
def read_ratings(ratings_path: Path) -> pd.DataFrame:
    usecols = ["tconst", "averageRating", "numVotes"]
    dtypes = {"tconst": "string", "averageRating": "float32", "numVotes": "Int64"}
    return pd.read_csv(
        ratings_path,
        sep="\t",
        usecols=usecols,
        dtype=dtypes,
        na_values=NA,
        keep_default_na=False,
        low_memory=False,
    )


# -------------------------
# PRINCIPALS (actors/directors/writers separated)
# -------------------------
def collect_people_by_role(
    principals_path: Path,
    movie_ids: set[str],
    top_actors: int = 3,
    top_writers: int = 2,
    include_director: bool = True,
) -> tuple[dict[str, dict[str, list[str]]], set[str]]:
    """
    Read title.principals.tsv in chunks and collect people by role.
    Returns:
      - mapping: tconst -> {"actors":[nconst...], "directors":[...], "writers":[...]}
      - needed_nconst: set of all nconst to resolve in name.basics.tsv
    """
    people_by_title: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"actors": [], "directors": [], "writers": []}
    )
    needed_nconst: set[str] = set()

    usecols = ["tconst", "ordering", "nconst", "category"]
    dtypes = {"tconst": "string", "ordering": "Int64", "nconst": "string", "category": "string"}

    actors_seen_count: dict[str, int] = defaultdict(int)
    writers_seen_count: dict[str, int] = defaultdict(int)
    director_seen: set[str] = set()

    total = estimate_total_chunks(principals_path, approx_mb_per_chunk=120)
    for chunk in tqdm(
        pd.read_csv(
            principals_path,
            sep="\t",
            usecols=usecols,
            dtype=dtypes,
            na_values=NA,
            keep_default_na=False,
            chunksize=800_000,
            low_memory=False,
        ),
        desc="Reading principals",
        total=total,
        unit="chunk",
    ):
        chunk = chunk[chunk["tconst"].isin(movie_ids)]
        if chunk.empty:
            continue

        chunk = chunk.sort_values(["tconst", "ordering"], kind="mergesort")

        for row in chunk.itertuples(index=False):
            tconst = row.tconst
            cat = row.category
            nconst = row.nconst

            if cat in ("actor", "actress"):
                if actors_seen_count[tconst] < top_actors:
                    people_by_title[tconst]["actors"].append(nconst)
                    needed_nconst.add(nconst)
                    actors_seen_count[tconst] += 1

            elif include_director and cat == "director":
                if tconst not in director_seen:
                    people_by_title[tconst]["directors"].append(nconst)
                    needed_nconst.add(nconst)
                    director_seen.add(tconst)

            elif cat == "writer":
                if writers_seen_count[tconst] < top_writers:
                    people_by_title[tconst]["writers"].append(nconst)
                    needed_nconst.add(nconst)
                    writers_seen_count[tconst] += 1

    return dict(people_by_title), needed_nconst


# -------------------------
# NAMES
# -------------------------
def load_names(name_basics_path: Path, needed_nconst: set[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}

    usecols = ["nconst", "primaryName"]
    dtypes = {"nconst": "string", "primaryName": "string"}

    total = estimate_total_chunks(name_basics_path, approx_mb_per_chunk=120)
    for chunk in tqdm(
        pd.read_csv(
            name_basics_path,
            sep="\t",
            usecols=usecols,
            dtype=dtypes,
            na_values=NA,
            keep_default_na=False,
            chunksize=800_000,
            low_memory=False,
        ),
        desc="Reading names",
        total=total,
        unit="chunk",
    ):
        chunk = chunk[chunk["nconst"].isin(needed_nconst)]
        if not chunk.empty:
            mapping.update(zip(chunk["nconst"], chunk["primaryName"]))

    return mapping


# -------------------------
# FEATURES
# -------------------------
def _names_to_tokens(nconst_list: list[str], nconst_to_name: dict[str, str]) -> str:
    # Keep spaces safe for tokenization
    out = []
    for nconst in nconst_list:
        name = nconst_to_name.get(nconst, "")
        if name:
            out.append(name.replace(" ", "_"))
    return " ".join(out)


def build_people_columns(df: pd.DataFrame, people_by_title: dict, nconst_to_name: dict[str, str]) -> pd.DataFrame:
    actors_map = {t: _names_to_tokens(v.get("actors", []), nconst_to_name) for t, v in people_by_title.items()}
    directors_map = {t: _names_to_tokens(v.get("directors", []), nconst_to_name) for t, v in people_by_title.items()}
    writers_map = {t: _names_to_tokens(v.get("writers", []), nconst_to_name) for t, v in people_by_title.items()}

    df["actors"] = df["tconst"].map(actors_map).fillna("")
    df["directors"] = df["tconst"].map(directors_map).fillna("")
    df["writers"] = df["tconst"].map(writers_map).fillna("")

    # Optional legacy column for backward compatibility / quick tests
    df["people"] = (df["directors"].astype(str) + " " + df["actors"].astype(str) + " " + df["writers"].astype(str)).str.strip()

    return df


def build_genre_tokens(genres_str: str) -> str:
    """
    Create richer genre features:
      - single tokens: g:Action
      - pair tokens: gpair:Action|Sci-Fi
    """
    if not genres_str:
        return ""
    parts = [g for g in genres_str.replace(",", " ").split() if g and g not in BAD_GENRES]
    parts = sorted(set(parts))
    if not parts:
        return ""

    singles = [f"g:{g}" for g in parts]
    pairs = [f"gpair:{a}|{b}" for a, b in combinations(parts, 2)]
    return " ".join(singles + pairs)


def build_content(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["primaryTitle"] = df["primaryTitle"].fillna("").astype(str).str.strip()
    df["genres"] = clean_genres_str(df["genres"])

    # Drop rows with empty genres after cleaning
    df = df[df["genres"].ne("")].copy()

    # Build genre_tokens
    df["genre_tokens"] = df["genres"].astype(str).apply(build_genre_tokens)

    # Drop rows with no director (director is strong signal)
    df["directors"] = df["directors"].fillna("").astype(str).str.strip()
    df = df[df["directors"].ne("")].copy()

    # Actors/writers can be empty, but usually we want at least one actor token
    df["actors"] = df["actors"].fillna("").astype(str).str.strip()
    df = df[df["actors"].ne("")].copy()

    # Start year stays numeric for later models, but can be used in a text "content" field too
    if "startYear" in df.columns:
        df["startYear"] = df["startYear"].astype("Int64")
        year_str = df["startYear"].astype("string").fillna("")
    else:
        year_str = ""

    # A light "content" field (optional; may be useful for quick baselines)
    df["content"] = (
        df["primaryTitle"].astype("string")
        + " "
        + df["genres"].astype("string")
        + " "
        + df["directors"].astype("string")
        + " "
        + df["actors"].astype("string")
        + " "
        + df["writers"].astype("string")
        + " "
        + df["genre_tokens"].astype("string")
        + " "
        + year_str
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    # runtime sanity filter
    df = df[df["runtimeMinutes"].between(10, 360)]

    return df


# -------------------------
# MAIN
# -------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Build a clean movie dataset from IMDb dumps (roles separated).")
    p.add_argument("--imdb_dir", default="data/raw/imdb", help="Folder with IMDb TSV files")
    p.add_argument("--out", default="data/processed/movie_imdb.parquet", help="Output parquet path")
    p.add_argument("--top_actors", type=int, default=3, help="Number of main actors/actresses to keep")
    p.add_argument("--top_writers", type=int, default=2, help="Number of writers to keep")
    p.add_argument("--min_votes", type=int, default=200, help="Minimum number of votes to keep a movie")
    args = p.parse_args()

    imdb_dir = Path(args.imdb_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    basics_path = imdb_dir / "title.basics.tsv"
    ratings_path = imdb_dir / "title.ratings.tsv"
    principals_path = imdb_dir / "title.principals.tsv"
    names_path = imdb_dir / "name.basics.tsv"

    print("1/6 Loading basics (movies only)...")
    basics = read_movies_basics(basics_path)
    print(f"   -> {len(basics):,} movies")

    print("2/6 Loading ratings and merging...")
    ratings = read_ratings(ratings_path)
    df = basics.merge(ratings, on="tconst", how="left")

    df["numVotes"] = df["numVotes"].fillna(0)
    df = df[df["numVotes"] >= args.min_votes]
    print(f"   -> after votes filter: {len(df):,} movies")

    movie_ids = set(df["tconst"].astype("string").tolist())

    print("3/6 Collecting people by role (actors/directors/writers) (chunked)...")
    people_by_title, needed_nconst = collect_people_by_role(
        principals_path,
        movie_ids,
        top_actors=args.top_actors,
        top_writers=args.top_writers,
        include_director=True,
    )
    print(f"   -> titles with any collected people: {len(people_by_title):,}")
    print(f"   -> unique people needed: {len(needed_nconst):,}")

    print("4/6 Loading names for needed people only (chunked)...")
    nconst_to_name = load_names(names_path, needed_nconst)
    print(f"   -> names resolved: {len(nconst_to_name):,}")

    print("5/6 Building features (actors/directors/writers/genre_tokens)...")
    df = build_people_columns(df, people_by_title, nconst_to_name)
    df = build_content(df)
    print(f"   -> after content filters: {len(df):,} movies")

    # Keep only useful columns (you can add/remove as needed)
    keep_cols = [
        "tconst",
        "primaryTitle",
        "originalTitle",
        "startYear",
        "runtimeMinutes",
        "genres",
        "genre_tokens",
        "actors",
        "directors",
        "writers",
        "people",
        "averageRating",
        "numVotes",
        "content",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    print("6/6 Saving parquet...")
    df.to_parquet(out_path, index=False)
    print(f"\nSAVED: {out_path} ({out_path.stat().st_size / (1024**2):.2f} MB)")


if __name__ == "__main__":
    main()