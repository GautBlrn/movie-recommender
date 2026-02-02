# scripts/predict.py
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.preprocessing import normalize

# ---- load config ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config as cfg


# --------------------------------------------------
# One-hot encoding with fixed columns
# --------------------------------------------------
def one_hot_fixed(series: pd.Series, prefix: str, expected_cols: list[str]) -> sparse.csr_matrix:
    """
    One-hot encode `series` then align to `expected_cols` (train-time columns).
    Always returns a numeric CSR matrix (n_rows, len(expected_cols)).
    """
    n = len(series)
    if not expected_cols:
        return sparse.csr_matrix((n, 0), dtype=np.float32)

    tmp = pd.get_dummies(series.fillna("UNK").astype(str), prefix=prefix)
    tmp = tmp.reindex(columns=expected_cols, fill_value=0)

    arr = tmp.to_numpy(dtype=np.float32, copy=False)
    return sparse.csr_matrix(arr)


def split_genres(series: pd.Series) -> list[list[str]]:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(",", " ", regex=False)
        .str.strip()
        .str.split()
        .tolist()
    )


# --------------------------------------------------
# Engineered columns (must match train.py)
# --------------------------------------------------
def ensure_engineered_cols(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    df = df.copy()

    cfg_bundle = bundle.get("config", {}) or {}
    prior = float(cfg_bundle.get("rating_prior", getattr(cfg, "RATING_PRIOR", 500)))
    k_conf = float(cfg_bundle.get("confidence_k", getattr(cfg, "CONFIDENCE_K", 500)))
    tau_years = float(cfg_bundle.get("votes_per_year_bayes_tau", getattr(cfg, "VOTES_PER_YEAR_BAYES_TAU", 3.0)))

    votes = (
        pd.to_numeric(df["numVotes"], errors="coerce").fillna(0).astype(float)
        if "numVotes" in df.columns
        else pd.Series(0.0, index=df.index)
    )

    rating = (
        pd.to_numeric(df["averageRating"], errors="coerce").fillna(0).astype(float)
        if "averageRating" in df.columns
        else pd.Series(0.0, index=df.index)
    )


    # confidence
    if "confidence" not in df.columns:
        df["confidence"] = votes / (votes + k_conf)

    # rating_bayes
    if "rating_bayes" not in df.columns:
        global_mean = float(bundle.get("global_mean_rating", float(rating.mean() if len(rating) else 0.0)))
        df["rating_bayes"] = (global_mean * prior + votes * rating) / (prior + votes)

    if "votes_per_year_bayes" not in df.columns:
        current_year = int(cfg_bundle.get("train_year", datetime.now().year))

        year = (
            pd.to_numeric(df["startYear"], errors="coerce")
            if "startYear" in df.columns
            else pd.Series(np.nan, index=df.index)
        )

        # MUST match train.py
        age = (current_year - year).clip(lower=0)
        age = (age + 1).fillna(1).astype(float)

        votes_per_year = votes / age

        # median robust (sur Series)
        mu_vpy = float(bundle.get(
            "mu_votes_per_year_median",
            float(pd.Series(votes_per_year).median())
        ))

        df["votes_per_year_bayes"] = (mu_vpy * tau_years + votes) / (age + tau_years)

    return df

# --------------------------------------------------
# Feature builder (SAME ORDER + SAME WEIGHTS AS TRAIN)
# --------------------------------------------------
def build_features(df: pd.DataFrame, bundle: dict) -> sparse.csr_matrix | np.ndarray:
    # --- from bundle
    mlb = bundle["mlb_genres"]
    tfidf_blocks: dict = bundle["tfidf_blocks"]
    scaler = bundle.get("scaler")
    num_cols = bundle.get("num_cols", [])
    svd = bundle.get("svd")

    # IMPORTANT:
    # train.py applies scalar weights to blocks BEFORE fitting NN.
    # Therefore, at predict time we MUST apply the SAME weights to the query features,
    # otherwise the query lives in a different feature space and kneighbors() is biased.

    # ---- genres multi-hot
    genres_series = df["genres"] if "genres" in df.columns else pd.Series([""] * len(df))
    X_genres = mlb.transform(split_genres(genres_series))

    # ---- tfidf blocks
    def tfidf_or_empty(col: str, vec_key: str) -> sparse.csr_matrix:
        vec = tfidf_blocks.get(vec_key)
        if vec is None:
            return sparse.csr_matrix((df.shape[0], 0), dtype=np.float32)
        if col not in df.columns:
            text = pd.Series([""] * len(df))
        else:
            text = df[col].fillna("").astype(str)
        return vec.transform(text)

    X_genre_tokens = tfidf_or_empty("genre_tokens", "genre_tokens")
    X_directors = tfidf_or_empty("directors", "directors")
    X_writers = tfidf_or_empty("writers", "writers")
    X_actors = tfidf_or_empty("actors", "actors")

    # ---- apply TRAIN-TIME weights from bundle (fallback to config if missing)
    cfg_bundle = bundle.get("config", {}) or {}
    w = (cfg_bundle.get("weights", {}) or {})

    w_genres = float(w.get("genres", getattr(cfg, "W_GENRES", 1.0)))
    w_gtok = float(w.get("genre_tokens", getattr(cfg, "W_GENRE_TOKENS", 1.0)))
    w_dir = float(w.get("directors", getattr(cfg, "W_DIRECTORS", 1.0)))
    w_wri = float(w.get("writers", getattr(cfg, "W_WRITERS", 1.0)))
    w_act = float(w.get("actors", getattr(cfg, "W_ACTORS", 1.0)))

    if w_genres != 1.0:
        X_genres = X_genres * w_genres
    if w_gtok != 1.0:
        X_genre_tokens = X_genre_tokens * w_gtok
    if w_dir != 1.0:
        X_directors = X_directors * w_dir
    if w_wri != 1.0:
        X_writers = X_writers * w_wri
    if w_act != 1.0:
        X_actors = X_actors * w_act

    # ---- numeric (scaled)
    X_num = sparse.csr_matrix((df.shape[0], 0), dtype=np.float32)
    if scaler is not None and num_cols:
        num_df = df.reindex(columns=num_cols).copy()
        for col in num_cols:
            num_df[col] = pd.to_numeric(num_df[col], errors="coerce")

        # must match train.py
        if "numVotes" in num_df.columns and getattr(cfg, "LOG1P_VOTES", True):
            num_df["numVotes"] = np.log1p(num_df["numVotes"].fillna(0))

        X_num = sparse.csr_matrix(
            scaler.transform(num_df.fillna(0).to_numpy(dtype=np.float32))
        )

    # ---- engineered categorical features
    dd = df.copy()

    if "decade" not in dd.columns:
        dd["decade"] = (pd.to_numeric(dd["startYear"], errors="coerce").fillna(0).astype(int) // 10) * 10

    # rating bucket (config-driven)
    rating_bins = list(getattr(cfg, "RATING_BUCKET_BINS", [0, 6.0, 7.5, 10.0]))
    rating_labels = list(getattr(cfg, "RATING_BUCKET_LABELS", ["low", "mid", "high"]))

    dd["rating_bucket"] = pd.cut(
        pd.to_numeric(dd["averageRating"], errors="coerce"),
        bins=rating_bins,
        labels=rating_labels,
        include_lowest=True,
    )

    # runtime bucket (config-driven)
    runtime_bins = list(getattr(cfg, "RUNTIME_BUCKET_BINS", [0, 60, 400]))
    runtime_labels = list(getattr(cfg, "RUNTIME_BUCKET_LABELS", ["short", "long"]))

    dd["runtime_bucket"] = pd.cut(
        pd.to_numeric(dd["runtimeMinutes"], errors="coerce"),
        bins=runtime_bins,
        labels=runtime_labels,
        include_lowest=True,
    )

    # one-hot encode engineered cats (aligned to train-time columns)
    cat_cols = bundle.get("cat_cols", {}) or {}
    X_decade = one_hot_fixed(dd["decade"], "decade", cat_cols.get("decade", []))
    X_rating = one_hot_fixed(dd["rating_bucket"], "rating", cat_cols.get("rating", []))
    X_runtime = one_hot_fixed(dd["runtime_bucket"], "runtime", cat_cols.get("runtime", []))

    # ---- concat (MUST match train.py order)
    X = sparse.hstack(
        [
            X_genres,
            X_genre_tokens,
            X_directors,
            X_writers,
            X_actors,
            X_num,
            X_decade,
            X_rating,
            X_runtime,
        ],
        format="csr",
    )

    if svd is not None:
        X = svd.transform(X)

    return X # type: ignore


def pick_query_row(df: pd.DataFrame, title: str) -> tuple[pd.DataFrame, int]:
    q = title.strip().lower()

    exact = df[df["primaryTitle"].fillna("").astype(str).str.lower() == q]
    if not exact.empty:
        tmp = exact.copy()
        tmp["numVotes"] = pd.to_numeric(tmp["numVotes"], errors="coerce").fillna(0)
        row = tmp.sort_values("numVotes", ascending=False).iloc[[0]]
        return row, int(row.index[0])

    contains = df[df["primaryTitle"].fillna("").astype(str).str.lower().str.contains(q, regex=False)]
    if contains.empty:
        return pd.DataFrame(), -1

    tmp = contains.copy()
    tmp["numVotes"] = pd.to_numeric(tmp["numVotes"], errors="coerce").fillna(0)
    row = tmp.sort_values("numVotes", ascending=False).iloc[[0]]
    return row, int(row.index[0])

# PRINT PARAMS
def print_train_settings() -> None:
    print("Train/data settings:")
    print(f"  Filter: MIN_VOTES={cfg.MIN_VOTES}  MIN_RATING={cfg.MIN_RATING}")

    print("  TF-IDF blocks:")
    print(f"    MIN_DF_DIRECTORS={cfg.MIN_DF_DIRECTORS}  MAX_FEATURES_DIRECTORS={cfg.MAX_FEATURES_DIRECTORS}")
    print(f"    MIN_DF_WRITERS={cfg.MIN_DF_WRITERS}    MAX_FEATURES_WRITERS={cfg.MAX_FEATURES_WRITERS}")
    print(f"    MIN_DF_ACTORS={cfg.MIN_DF_ACTORS}      MAX_FEATURES_ACTORS={cfg.MAX_FEATURES_ACTORS}")
    print(f"    MIN_DF_GENRE_TOKENS={cfg.MIN_DF_GENRE_TOKENS}  MAX_FEATURES_GENRE_TOKENS={cfg.MAX_FEATURES_GENRE_TOKENS}")

    print("  Feature flags / buckets:")
    print(f"    USE_NUMERIC={cfg.USE_NUMERIC}  LOG1P_VOTES={cfg.LOG1P_VOTES}")
    print(f"    RATING_BUCKET_BINS={cfg.RATING_BUCKET_BINS}  labels={cfg.RATING_BUCKET_LABELS}")
    print(f"    RUNTIME_BUCKET_BINS={cfg.RUNTIME_BUCKET_BINS}  labels={cfg.RUNTIME_BUCKET_LABELS}")

    print("  Modeling:")
    print(f"    SVD_COMPONENTS={cfg.SVD_COMPONENTS}  N_NEIGHBORS={cfg.N_NEIGHBORS}  RANDOM_STATE={cfg.RANDOM_STATE}")

    print("  Bayesian smoothing:")
    print(f"    RATING_PRIOR={cfg.RATING_PRIOR}  CONFIDENCE_K={cfg.CONFIDENCE_K}  VOTES_PER_YEAR_BAYES_TAU={cfg.VOTES_PER_YEAR_BAYES_TAU}")

    print("  Genre token engineering:")
    print(f"    USE_GENRE_TRIPLES={cfg.USE_GENRE_TRIPLES}  USE_AMBIENCE_TOKENS={cfg.USE_AMBIENCE_TOKENS}")
    print(f"    GENRE_PAIR_MIN_DF={cfg.GENRE_PAIR_MIN_DF}  GENRE_PAIR_MAX_DF_RATIO={cfg.GENRE_PAIR_MAX_DF_RATIO}")
    print(f"    GENRE_TRIPLE_MIN_DF={cfg.GENRE_TRIPLE_MIN_DF}  GENRE_TRIPLE_MIN_DF_FOCUS={cfg.GENRE_TRIPLE_MIN_DF_FOCUS}")
    print(f"    GENRE_PAIR_DRAMA_PENALTY={cfg.GENRE_PAIR_DRAMA_PENALTY}  GENRE_PAIR_FOCUS_BONUS={cfg.GENRE_PAIR_FOCUS_BONUS}")
    print(f"    GENRE_PAIR_SCALE={cfg.GENRE_PAIR_SCALE}  GENRE_PAIR_MAX_REP={cfg.GENRE_PAIR_MAX_REP}")
    print(f"    GENRE_TRIPLE_SCALE={cfg.GENRE_TRIPLE_SCALE}  GENRE_TRIPLE_MAX_REP={cfg.GENRE_TRIPLE_MAX_REP}")
    print(f"    PACE_FAST_RUNTIME_MAX={cfg.PACE_FAST_RUNTIME_MAX}  PACE_SLOW_RUNTIME_MIN={cfg.PACE_SLOW_RUNTIME_MIN}")
    print(f"    QUAL_HIGH_MIN={cfg.QUAL_HIGH_MIN}  QUAL_LOW_MAX={cfg.QUAL_LOW_MAX}")
    print(f"    FOCUS_GENRES={sorted(list(cfg.FOCUS_GENRES))}")

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union

def safe_genres(val: str) -> set[str]:
    """
    Extract genre tokens from the space-separated `genres` column.
    """
    if not isinstance(val, str) or not val.strip():
        return set()
    # split on whitespace (dataset uses space-separated genres)
    return set(val.split())

def main() -> None:
    p = argparse.ArgumentParser(description="Recommend movies by title (rerank via config.py).")
    p.add_argument("--title", required=True, help="Movie title to search (case-insensitive)")
    p.add_argument("--k", type=int, default=10, help="Number of recommendations to return")
    p.add_argument("--model", default=str(cfg.MODEL_BUNDLE_PATH), help="Path to model bundle")
    p.add_argument("--data", default=str(cfg.DATA_PATH), help="Path to processed dataset parquet")
    p.add_argument("--json", action="store_true", help="Output JSON only (for scripts)")
    args = p.parse_args()

    # Support relative OR absolute paths
    bundle_path = Path(args.model)

    if not bundle_path.is_absolute():
        bundle_path = PROJECT_ROOT / bundle_path

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    if not bundle_path.exists():
        print(f"Model bundle not found: {bundle_path}")
        return
    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        return
    if not args.json:
        print("USING CONFIG:", cfg.__file__)

    bundle = joblib.load(bundle_path)

    # ---- effective params used (train vs predict)
    bundle_cfg = bundle.get("config", {}) or {}
    bundle_weights = (bundle_cfg.get("weights", {}) or {})

    if not args.json:
        print("\n=== EFFECTIVE SETTINGS ===")
        print(f"Model bundle: {bundle_path}")
        print(f"Dataset:      {data_path}")
        print(f"Pool rule:    pool=max(PRED_POOL_MIN, k*PRED_POOL_MULT) => max({cfg.PRED_POOL_MIN}, {args.k}*{cfg.PRED_POOL_MULT})")
        print(f"Rerank:       W_SIM={cfg.W_SIM}  W_RATING={cfg.W_RATING}  W_VOTES={cfg.W_VOTES}  W_YEAR={cfg.W_YEAR}")
        print(f"Norms:        VOTES_NORM_MAX={cfg.VOTES_NORM_MAX}  YEAR_GAP_MAX={cfg.YEAR_GAP_MAX}")
        print("Train weights (from bundle, used for NN/query space):")
        print(f"  genres={bundle_weights.get('genres')}  genre_tokens={bundle_weights.get('genre_tokens')}  "
              f"directors={bundle_weights.get('directors')}  writers={bundle_weights.get('writers')}  actors={bundle_weights.get('actors')}")
        print("Config weights (ignored for NN if different; only used as fallback if bundle missing):")
        print(f"  genres={getattr(cfg,'W_GENRES',None)}  genre_tokens={getattr(cfg,'W_GENRE_TOKENS',None)}  "
              f"directors={getattr(cfg,'W_DIRECTORS',None)}  writers={getattr(cfg,'W_WRITERS',None)}  actors={getattr(cfg,'W_ACTORS',None)}")
        print("==========================\n")

    df = pd.read_parquet(data_path).reset_index(drop=True)
    df = ensure_engineered_cols(df, bundle)

    meta = bundle.get("df_meta")
    if meta is None or len(meta) == 0:
        print("Bundle missing df_meta.")
        return

    row, idx = pick_query_row(df, args.title)
    if idx < 0 or row.empty:
        print("Film introuvable")
        return

    Xq = build_features(row, bundle)
    Xq = normalize(Xq, norm="l2")
    nn = bundle.get("nn")
    if nn is None:
        print("Bundle missing NearestNeighbors (nn).")
        return

    # Candidate pool for rerank (from config)
    pool = max(int(cfg.PRED_POOL_MIN), int(args.k) * int(cfg.PRED_POOL_MULT))
    distances, indices = nn.kneighbors(Xq, n_neighbors=pool)
    print(f"Candidate pool size used: {pool} (k={args.k})")

    q_title = str(row.iloc[0].get("primaryTitle", ""))
    q_year = int(pd.to_numeric(row.iloc[0].get("startYear", 0), errors="coerce") or 0)

    # weights (from config)
    w_sim = float(cfg.W_SIM)
    w_rating = float(cfg.W_RATING)
    w_votes = float(cfg.W_VOTES)
    w_year = float(cfg.W_YEAR)

    votes_norm_max = float(cfg.VOTES_NORM_MAX)
    year_gap_max = float(cfg.YEAR_GAP_MAX)

    cands: list[dict] = []
    for dist, i in zip(distances[0], indices[0]):
        i = int(i)
        if i == int(idx):
            continue

        m = meta.iloc[i]
        sim = float(1 - dist)

        # rating signal
        r = m.get("rating_bayes", m.get("averageRating", 0.0))
        r = float(pd.to_numeric(r, errors="coerce") or 0.0)
        r_norm = r / 10.0

        votes = float(pd.to_numeric(m.get("numVotes", 0.0), errors="coerce") or 0.0)
        votes_norm = float(np.log1p(votes) / np.log1p(votes_norm_max)) if votes_norm_max > 0 else 0.0
        votes_norm = min(1.0, max(0.0, votes_norm))

        y = int(pd.to_numeric(m.get("startYear", 0), errors="coerce") or 0)
        gap = abs(y - q_year) if (q_year and y) else 0
        year_score = max(0.0, 1.0 - (gap / year_gap_max)) if year_gap_max > 0 else 0.0

        # penalty for very low confidence
        conf = float(m.get("confidence", 0.0))
        conf_penalty = 0.85 + 0.15 * conf # in [0.85 ; 1.0]

        s_sim = w_sim * sim
        s_rating = w_rating * r_norm
        s_votes = w_votes * votes_norm
        s_year = w_year * year_score

        # injecter dans le score
        genre_overlap = jaccard(
            safe_genres(row.iloc[0].genres),
            safe_genres(m.get("genres", ""))
        )
        s_genre = float(cfg.W_GENRE_OVERLAP) * genre_overlap

        raw = s_sim + s_rating + s_votes + s_year + s_genre
        
        # penality
        score = raw * conf_penalty

        cands.append({
            "title": str(m.get("primaryTitle", "")),
            "year": y,
            "genre": str(m.get("genres","")),
            "score": round(score, 3),
            "raw": round(raw, 3),

            "genre_overlap": round(genre_overlap, 3),
            "s_genre": round(s_genre, 3),

            "conf": round(conf, 3),
            "conf_penalty": round(conf_penalty, 3),

            "s_sim": round(s_sim, 3),
            "s_rating": round(s_rating, 3),
            "s_votes": round(s_votes, 3),
            "s_year": round(s_year, 3),

            "sim": round(sim, 3),
            "rating": round(r, 1),
            "votes": int(votes),
            "gap": gap,
        })

    cands.sort(key=lambda d: d["score"], reverse=True)
    top = cands[: int(args.k)]

    result = {
        "query": {"title": q_title, "year": q_year},
        "k": int(args.k),
        "recs": top,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
        return

    out = pd.DataFrame(top)
    out = out.assign(
        sim_score    = out["s_sim"].map(lambda x: f"{x:.2f}"),
        rating_score = out["s_rating"].map(lambda x: f"{x:.2f}"),
        votes_score  = out["s_votes"].map(lambda x: f"{x:.2f}"),
        year_score   = out["s_year"].map(lambda x: f"{x:.2f}"),
        genre_score  = out["s_genre"].map(lambda x: f"{x:.2f}"),
        genre_overlap= out["genre_overlap"].map(lambda x: f"{x:.2f}"),
        score        = out["score"].map(lambda x: f"{float(x):.3f}"),
        sim          = out["sim"].map(lambda x: f"{float(x):.3f}"),
        rating       = out["rating"].map(lambda x: f"{float(x):.1f}"),
        votes        = out["votes"].map(int),
    )

    print(f"\nQuery: {q_title} ({q_year})")
    print("-" * 90)
    print(out.to_string(index=True))

    if getattr(cfg, "PRED_DEBUG", False):
        dbg = (
            pd.DataFrame(cands)
            .sort_values("score", ascending=False)
            .head(int(getattr(cfg, "PRED_DEBUG_TOPN", 10)))
        )[["title", "year", "score", "sim", "rating", "votes"]].copy()
        dbg["votes"] = dbg["votes"].astype(int)
        print("\nTop candidates (after rerank scoring preview):")
        print(dbg.to_string(index=False))

    print_train_settings()


if __name__ == "__main__":
    main()