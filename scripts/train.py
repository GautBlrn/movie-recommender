# scripts/train.py
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


# ---- load config ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config as cfg  # noqa: E402


# -----------------------------
# Helpers
# -----------------------------
def split_genres(series: pd.Series) -> list[list[str]]:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(",", " ", regex=False)
        .str.strip()
        .str.split()
        .tolist()
    )


def build_rating_bayes(df: pd.DataFrame, prior: float) -> tuple[pd.Series, float]:
    rating = pd.to_numeric(df["averageRating"], errors="coerce").fillna(0).astype(float)
    votes = pd.to_numeric(df["numVotes"], errors="coerce").fillna(0).astype(float)
    global_mean = float(rating.mean()) if len(rating) else 0.0
    rb = (global_mean * prior + votes * rating) / (prior + votes)
    return rb, global_mean


def add_votes_per_year_bayes(
    df: pd.DataFrame,
    tau_years: float = 3.0,
    current_year: int | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    votes_per_year_bayes = (median(votes_per_year)*tau + votes) / (age_years + tau)
    Using the median makes it robust to outliers.
    """
    out = df.copy()
    if current_year is None:
        current_year = datetime.now().year

    year = pd.to_numeric(out.get("startYear"), errors="coerce")
    votes = pd.to_numeric(out.get("numVotes"), errors="coerce").fillna(0).astype(float)

    age = (current_year - year).clip(lower=0)
    age = (age + 1).fillna(1).astype(float)  # avoid /0
    out["age_years"] = age
    out["votes_per_year"] = votes / out["age_years"]

    mu_vpy = float(out["votes_per_year"].median()) if len(out) else 0.0
    out["votes_per_year_bayes"] = (mu_vpy * tau_years + votes) / (out["age_years"] + tau_years)
    return out, mu_vpy


def fit_tfidf_block(
    df: pd.DataFrame,
    col: str,
    *,
    min_df: int,
    max_features: int,
    weight: float,
) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    """
    Fit TF-IDF on a text column and apply a scalar weight to the resulting matrix.
    """
    if col not in df.columns:
        print(f"⚠️ Column '{col}' missing in dataset -> using empty strings (block becomes mostly zero).")
        text = pd.Series([""] * len(df))
    else:
        text = df[col].fillna("").astype(str)

    vec = TfidfVectorizer(
        lowercase=False,
        min_df=min_df,
        max_features=max_features,
        token_pattern=r"(?u)\b\w+\b",
    )
    X = vec.fit_transform(text)
    if weight != 1.0:
        X = X * float(weight)
    return vec, X.tocsr()


def one_hot_dummies_fixed(df: pd.DataFrame, col: str, prefix: str) -> tuple[list[str], sparse.csr_matrix]:
    """
    Fit-time one-hot with columns captured (for alignment at predict time).
    """
    tmp = pd.get_dummies(df[col].astype(str), prefix=prefix)
    cols = tmp.columns.tolist()
    X = sparse.csr_matrix(tmp.to_numpy(dtype=np.float32, copy=False))
    return cols, X


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    data_file = PROJECT_ROOT / cfg.DATA_PATH
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    print("Loaded dataset:", data_file)
    df = pd.read_parquet(data_file).reset_index(drop=True)
    print("shape:", df.shape)

    # --- Basic cleaning (robust)
    for c in ["primaryTitle", "genres"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    # --- Bayesian rating + confidence (Section 5)
    print("\n🔹 Computing rating_bayes & confidence...")
    t0 = time.time()

    # You decided: prior=500, k=500 (update config.py accordingly)
    rating_prior = float(getattr(cfg, "RATING_PRIOR", 500))
    confidence_k = float(getattr(cfg, "CONFIDENCE_K", 500))

    df["rating_bayes"], global_mean_rating = build_rating_bayes(df, prior=rating_prior)
    votes = pd.to_numeric(df["numVotes"], errors="coerce").fillna(0).astype(float)
    df["confidence"] = votes / (votes + confidence_k)
    print(f"  → done in {time.time() - t0:.2f}s | global_mean_rating={global_mean_rating:.4f}")

    # --- Votes per year (Bayes) (Section 4)
    # Defaults: tau=3.0
    tau_years = float(getattr(cfg, "VOTES_PER_YEAR_BAYES_TAU", 3.0))
    df, mu_vpy = add_votes_per_year_bayes(df, tau_years=tau_years)
    print(f"🔹 votes_per_year_bayes: tau={tau_years} | mu(median)={mu_vpy:.3f}")

    # --- GENRES multi-hot (kept)
    print("\n🔹 Encoding genres (MultiLabelBinarizer)...")
    t0 = time.time()
    mlb = MultiLabelBinarizer(sparse_output=True)
    X_genres = mlb.fit_transform(split_genres(df["genres"]))
    print(f"  → genres shape {X_genres.shape} in {time.time() - t0:.1f}s")

    # --- TF-IDF blocks (Section 6)
    # Weights (your conclusions)
    W_DIRECTORS = float(getattr(cfg, "W_DIRECTORS", 2.0))
    W_WRITERS = float(getattr(cfg, "W_WRITERS", 1.3))
    W_ACTORS = float(getattr(cfg, "W_ACTORS", 0.6))
    W_GENRE_TOKENS = float(getattr(cfg, "W_GENRE_TOKENS", 0.8))
    W_GENRES = float(getattr(cfg, "W_GENRES", 1.0))  # multi-hot scaling

    if W_GENRES != 1.0:
        X_genres = X_genres * W_GENRES

    # Default TF-IDF caps (can be overridden in config.py if you add them)
    MAXF = int(getattr(cfg, "MAX_FEATURES_TEXT", 50_000))
    MINDF = int(getattr(cfg, "MIN_DF_TEXT", 2))
    MAXF_GENRE_TOKENS = int(getattr(cfg, "MAX_FEATURES_GENRE_TOKENS", 5_000))

    print("\n🔹 Encoding TF-IDF blocks...")
    t0 = time.time()

    tfidf_directors, X_directors = fit_tfidf_block(
        df, "directors", min_df=MINDF, max_features=MAXF, weight=W_DIRECTORS
    )
    tfidf_writers, X_writers = fit_tfidf_block(
        df, "writers", min_df=MINDF, max_features=MAXF, weight=W_WRITERS
    )
    tfidf_actors, X_actors = fit_tfidf_block(
        df, "actors", min_df=MINDF, max_features=MAXF, weight=W_ACTORS
    )
    tfidf_genre_tokens, X_genre_tokens = fit_tfidf_block(
        df, "genre_tokens", min_df=1, max_features=MAXF_GENRE_TOKENS, weight=W_GENRE_TOKENS
    )

    print(
        "  → blocks done in "
        f"{time.time() - t0:.1f}s | "
        f"directors={X_directors.shape} writers={X_writers.shape} actors={X_actors.shape} genre_tokens={X_genre_tokens.shape}"
    )

    # --- Numeric features (scaled)
    print("\n🔹 Encoding numeric features...")
    num_cols = [
        "startYear",
        "runtimeMinutes",
        "averageRating",
        "numVotes",
        "rating_bayes",
        "confidence",
        "votes_per_year_bayes",
    ]

    scaler = None
    X_num = sparse.csr_matrix((df.shape[0], 0), dtype=np.float32)
    if getattr(cfg, "USE_NUMERIC", True) and num_cols:
        t0 = time.time()
        num_df = df.reindex(columns=num_cols).copy()
        for c in num_cols:
            num_df[c] = pd.to_numeric(num_df[c], errors="coerce")

        if getattr(cfg, "LOG1P_VOTES", True) and "numVotes" in num_df.columns:
            num_df["numVotes"] = np.log1p(num_df["numVotes"].fillna(0))

        scaler = StandardScaler()
        arr = scaler.fit_transform(num_df.fillna(0).to_numpy(dtype=np.float32))
        X_num = sparse.csr_matrix(arr)
        print(f"  → numeric shape {X_num.shape} in {time.time() - t0:.1f}s")
    else:
        print("  → numeric disabled")

    # --- Categorical engineered (fixed columns)
    print("\n🔹 Encoding engineered categorical features...")
    t0 = time.time()

    # decade
    df["decade"] = (pd.to_numeric(df["startYear"], errors="coerce").fillna(0).astype(int) // 10) * 10
    decade_cols, X_decade = one_hot_dummies_fixed(df, "decade", "decade")

    # rating bucket
    df["rating_bucket"] = pd.cut(
        pd.to_numeric(df["averageRating"], errors="coerce"),
        bins=[0, 3, 6, 8, 10],
        labels=["poor", "okay", "good", "great"],
    )
    rating_cols, X_rating = one_hot_dummies_fixed(df, "rating_bucket", "rating")

    # runtime bucket
    df["runtime_bucket"] = pd.cut(
        pd.to_numeric(df["runtimeMinutes"], errors="coerce"),
        bins=[0, 30, 60, 90, 120, 180, 300, 10_000],
        labels=["<30", "30-60", "60-90", "90-120", "120-180", "180-300", ">300"],
    )
    runtime_cols, X_runtime = one_hot_dummies_fixed(df, "runtime_bucket", "runtime")

    print(f"  → cat done in {time.time() - t0:.1f}s")

    # --- Combine all features (order matters!)
    print("\n🔹 Combining feature matrix...")
    t0 = time.time()
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
    print(f"  → X shape {X.shape} in {time.time() - t0:.1f}s")

    # --- SVD
    svd = None
    X_final = X
    n_svd = int(getattr(cfg, "SVD_COMPONENTS", 0) or 0)
    if n_svd > 0:
        print(f"\n🔹 Applying SVD ({n_svd} components)...")
        t0 = time.time()
        svd = TruncatedSVD(n_components=n_svd, random_state=int(getattr(cfg, "RANDOM_STATE", 42)))
        X_final = svd.fit_transform(X)
        print(f"  → SVD done in {time.time() - t0:.1f}s | shape {X_final.shape}")
    else:
        print("\n🔹 SVD disabled")

    # --- NearestNeighbors
    print("\n🔹 Fitting NearestNeighbors...")
    t0 = time.time()
    nn = NearestNeighbors(metric="cosine", n_neighbors=int(getattr(cfg, "N_NEIGHBORS", 50)))
    nn.fit(X_final)
    print(f"  → NN done in {time.time() - t0:.1f}s")

    # --- Save bundle
    print("\n🔹 Saving model bundle...")
    model_dir = PROJECT_ROOT / cfg.MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    # Keep meta used by predicts + rerank
    df_meta_cols = ["tconst", "primaryTitle", "startYear", "averageRating", "numVotes", "genres", "rating_bayes", "confidence"]
    df_meta_cols = [c for c in df_meta_cols if c in df.columns]
    df_meta = df[df_meta_cols].copy()

    bundle = {
        # Data
        "df_meta": df_meta,

        # Encoders
        "mlb_genres": mlb,
        "tfidf_blocks": {
            "genre_tokens": tfidf_genre_tokens,
            "directors": tfidf_directors,
            "writers": tfidf_writers,
            "actors": tfidf_actors,
        },
        "scaler": scaler,
        "num_cols": num_cols,

        # Cat columns for fixed alignment in predict
        "cat_cols": {
            "decade": decade_cols,
            "rating": rating_cols,
            "runtime": runtime_cols,
        },

        # Models
        "svd": svd,
        "nn": nn,

        # Global stats
        "global_mean_rating": global_mean_rating,
        "mu_votes_per_year_median": mu_vpy,

        # Feature order (so predicts can rebuild EXACTLY)
        "feature_order": [
            "genres_multi_hot",
            "genre_tokens_tfidf",
            "directors_tfidf",
            "writers_tfidf",
            "actors_tfidf",
            "numeric_scaled",
            "decade_onehot",
            "rating_bucket_onehot",
            "runtime_bucket_onehot",
        ],

        # Config snapshot
        "config": {
            "rating_prior": rating_prior,
            "confidence_k": confidence_k,
            "votes_per_year_bayes_tau": tau_years,

            "weights": {
                "genres": W_GENRES,
                "genre_tokens": W_GENRE_TOKENS,
                "directors": W_DIRECTORS,
                "writers": W_WRITERS,
                "actors": W_ACTORS,
            },

            "tfidf": {
                "min_df_text": MINDF,
                "max_features_text": MAXF,
                "max_features_genre_tokens": MAXF_GENRE_TOKENS,
            },

            "svd_components": n_svd,
            "n_neighbors": int(getattr(cfg, "N_NEIGHBORS", 50)),
        },
    }

    out_path = PROJECT_ROOT / cfg.MODEL_BUNDLE_PATH
    joblib.dump(bundle, out_path)
    print("✅ Bundle saved to:", out_path)
    print("🚀 Training complete!")


if __name__ == "__main__":
    main()