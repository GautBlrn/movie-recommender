from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
from itertools import combinations
import math
import sys
import pandas as pd
from tqdm import tqdm

# ---- load config ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config as cfg
print("USING CONFIG:", cfg.__file__)

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


# Poids des genres simples
GENRE_BASE_WEIGHT = {
    "Drama": 0.4,
    "Comedy": 0.6,
    "Romance": 0.6,

    "Action": 1.1,
    "Thriller": 1.1,
    "Crime": 1.1,

    "Horror": 1.6,
    "Sci-Fi": 1.3,

    "Adventure": 1.0,
    "Fantasy": 1.2,

    "Animation": 1.6,
    "Documentary": 1.3,
    "War": 1.2,
    "Western": 1.3,
    "Film-Noir": 1.4,
    "Sport": 1.5,
}

# ------------------------------------------
# Helper : répétition pondérée
# ------------------------------------------
def _repeat_token(
    token: str,
    weight: float,
    scale: int,
    min_rep: int = 1,
    max_rep: int = 3,
) -> list[str]:
    reps = int(round(weight * scale))
    reps = max(min_rep, min(max_rep, reps))
    return [token] * reps

# ------------------------------------------
# build_genre_tokens (pure, config-driven)
# ------------------------------------------
def build_genre_tokens(
    genres_str: str,
    *,
    runtime_min: int | None = None,
    rating: float | None = None,
    year: int | None = None,
    pair_df: dict[str, int] | None = None,
    triple_df: dict[str, int] | None = None,
) -> str:
    """
    Create *richer* genre tokens with:
      - pairs:    gpair:A|B
      - triples: gtriple:A|B|C
      - ambience / subgenre tokens
    Does NOT include solo tokens g:Genre — that info is in `genres` multi-hot.
    All thresholds & weights come from config.py.
    """

    if not genres_str:
        return ""

    parts = [g for g in genres_str.replace(",", " ").split() if g and g not in BAD_GENRES]
    parts = sorted(set(parts))
    if len(parts) < 1:
        return ""

    tokens: list[str] = []

    # -------------------------
    # Pairs (config-driven)
    # -------------------------
    for a, b in combinations(parts, 2):
        pair = f"{a}|{b}"

        # DF thresholds (si on a pair_df)
        dfp = pair_df.get(pair, 0) if pair_df else 0

        # Filtre DF trop faible / trop fréquent
        if pair_df:
            if dfp < cfg.GENRE_PAIR_MIN_DF:
                continue

            max_ratio = getattr(cfg, "GENRE_PAIR_MAX_DF_RATIO", None)
            if max_ratio is not None:
                n_docs = pair_df.get("_n_docs", 0) or 1
                if (dfp / n_docs) > max_ratio:
                    continue

        # Pondération
        w = 1.0
        if a == "Drama" or b == "Drama":
            w *= cfg.GENRE_PAIR_DRAMA_PENALTY
        if a in ("Horror", "Animation") or b in ("Horror", "Animation"):
            w *= cfg.GENRE_PAIR_FOCUS_BONUS

        w = min(w, cfg.GENRE_PAIR_FOCUS_BONUS)  # clip
        tokens.extend(
            _repeat_token(
                f"gpair:{pair}",
                w,
                cfg.GENRE_PAIR_SCALE,
                min_rep=1,
                max_rep=cfg.GENRE_PAIR_MAX_REP,
            )
        )

    # -------------------------
    # Triples (config-driven)
    # -------------------------
    if cfg.USE_GENRE_TRIPLES and triple_df is not None and len(parts) >= 3:
        for a, b, c in combinations(parts, 3):
            triple = f"{a}|{b}|{c}"

            dft = triple_df.get(triple, 0)
            # seuil flexible
            min_df = cfg.GENRE_TRIPLE_MIN_DF
            if "Horror" in (a, b, c) or "Animation" in (a, b, c):
                min_df = cfg.GENRE_TRIPLE_MIN_DF_FOCUS

            if dft < min_df:
                continue

            tokens.extend(
                _repeat_token(
                    f"gtriple:{triple}",
                    1.2,
                    cfg.GENRE_TRIPLE_SCALE,
                    min_rep=1,
                    max_rep=cfg.GENRE_TRIPLE_MAX_REP,
                )
            )

    # -------------------------
    # Ambience / subgenre tokens (config-driven)
    # -------------------------
    if cfg.USE_AMBIENCE_TOKENS:
        s = set(parts)

        # exemple de sous-genres utiles
        # Horror
        if "Horror" in s:
            if {"Mystery", "Thriller"} & s:
                tokens.append("horror:psych")
            if "Action" in s:
                tokens.append("horror:slasher")
            if "Fantasy" in s:
                tokens.append("horror:supernatural")
            if "Comedy" in s:
                tokens.append("horror:campy")
            if runtime_min is not None and runtime_min >= 120:
                tokens.append("horror:slowburn")
            if "Crime" in s:
                tokens.append("horror:brutal")

        # Animation
        if "Animation" in s:
            if "Family" in s:
                tokens.append("anim:family")
            if {"Crime", "War", "Horror", "Thriller"} & s and "Family" not in s:
                tokens.append("anim:adult")
            if {"Music", "Musical"} & s:
                tokens.append("anim:musical")
            if {"Adventure", "Fantasy"} & s:
                tokens.append("anim:adventure")
            if runtime_min is not None and runtime_min < 70:
                tokens.append("anim:short")

        # ambiances transversales
        if "Romance" in s and {"Drama", "Comedy"} & s:
            tokens.append("mood:romantic")
        if "War" in s and "Drama" in s:
            tokens.append("mood:war_drama")
        if "Adventure" in s and "Fantasy" in s:
            tokens.append("mood:epic")

        # pace
        if runtime_min is not None:
            if runtime_min <= cfg.PACE_FAST_RUNTIME_MAX and {"Action", "Thriller", "Horror"} & s:
                tokens.append("pace:fast")
            if runtime_min >= cfg.PACE_SLOW_RUNTIME_MIN and {"Drama", "History", "Documentary"} & s:
                tokens.append("pace:slow")

        # quality
        if rating is not None:
            if rating >= cfg.QUAL_HIGH_MIN:
                tokens.append("qual:high")
            elif rating <= cfg.QUAL_LOW_MAX:
                tokens.append("qual:low")

        # era
        if year is not None and year > 0:
            decade = (int(year) // 10) * 10
            tokens.append(f"era:{decade}s")

    return " ".join(tokens)


def _parse_genre_parts(genres_str: str) -> list[str]:
    if not genres_str:
        return []
    parts = [g for g in genres_str.replace(",", " ").split() if g and g not in BAD_GENRES]
    return sorted(set(parts))


def compute_pair_triple_df(genres_series: pd.Series) -> tuple[dict[str, int], dict[str, int]]:
    """
    DF = nombre de films contenant la paire/triple de genres (indépendamment des autres genres).
    """
    pair_counts = Counter()
    triple_counts = Counter()

    for s in genres_series.fillna("").astype(str):
        parts = _parse_genre_parts(s)
        if len(parts) < 2:
            continue

        for a, b in combinations(parts, 2):
            pair_counts[f"{a}|{b}"] += 1

        if len(parts) >= 3:
            for a, b, c in combinations(parts, 3):
                triple_counts[f"{a}|{b}|{c}"] += 1

    return dict(pair_counts), dict(triple_counts)


def build_content(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["primaryTitle"] = df["primaryTitle"].fillna("").astype(str).str.strip()
    df["genres"] = clean_genres_str(df["genres"])

    # Drop rows with empty genres after cleaning
    df = df[df["genres"].ne("")].copy()

    # Build genre_tokens
    # --- compute DF maps for pairs/triples (data-driven) ---
    pair_df, triple_df = compute_pair_triple_df(df["genres"].astype(str))
    # injecter le nombre de docs pour les ratios :
    pair_df["_n_docs"] = len(df)
    triple_df["_n_docs"] = len(df)

    df["genre_tokens"] = df.apply(
        lambda r: build_genre_tokens(
            r["genres"],
            runtime_min=int(r["runtimeMinutes"]) if pd.notna(r["runtimeMinutes"]) else None,
            rating=float(r["averageRating"]) if pd.notna(r.get("averageRating", None)) else None,
            year=int(r["startYear"]) if pd.notna(r.get("startYear", None)) else None,
            pair_df=pair_df,
            triple_df=triple_df,
        ),
        axis=1,
    )

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