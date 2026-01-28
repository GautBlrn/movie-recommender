# scripts/predicts.py
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

# ---- load config (project root like train.py) ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config as cfg  # noqa: E402


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
    """
    Ensure df has:
      - rating_bayes
      - confidence
      - votes_per_year_bayes
    Predict should still work if parquet doesn't store engineered cols.
    """
    df = df.copy()

    cfg_bundle = bundle.get("config", {})
    prior = float(cfg_bundle.get("rating_prior", getattr(cfg, "RATING_PRIOR", 500)))
    k_conf = float(cfg_bundle.get("confidence_k", getattr(cfg, "CONFIDENCE_K", 500)))
    tau_years = float(cfg_bundle.get("votes_per_year_bayes_tau", getattr(cfg, "VOTES_PER_YEAR_BAYES_TAU", 3.0)))

    votes = pd.to_numeric(df.get("numVotes", 0), errors="coerce").fillna(0).astype(float)
    rating = pd.to_numeric(df.get("averageRating", 0), errors="coerce").fillna(0).astype(float)

    # confidence
    if "confidence" not in df.columns:
        df["confidence"] = votes / (votes + k_conf)

    # rating_bayes
    if "rating_bayes" not in df.columns:
        global_mean = float(bundle.get("global_mean_rating", float(rating.mean() if len(rating) else 0.0)))
        df["rating_bayes"] = (global_mean * prior + votes * rating) / (prior + votes)

    # votes_per_year_bayes
    if "votes_per_year_bayes" not in df.columns:
        current_year = datetime.now().year
        year = pd.to_numeric(df.get("startYear", np.nan), errors="coerce")
        age = (current_year - year).clip(lower=0)
        age = (age + 1).fillna(1).astype(float)
        votes_per_year = votes / age

        mu_vpy = float(bundle.get("mu_votes_per_year_median", float(pd.Series(votes_per_year).median())))
        df["votes_per_year_bayes"] = (mu_vpy * tau_years + votes) / (age + tau_years)

    return df


# --------------------------------------------------
# Feature builder (SAME ORDER AS TRAIN)
# --------------------------------------------------
def build_features(df: pd.DataFrame, bundle: dict) -> sparse.csr_matrix | np.ndarray:
    # --- from bundle
    mlb = bundle["mlb_genres"]
    tfidf_blocks: dict = bundle["tfidf_blocks"]
    scaler = bundle.get("scaler")
    num_cols = bundle.get("num_cols", [])
    svd = bundle.get("svd")

    # --- weights used in training (already applied to matrices at train time)
    # In predict we must NOT re-apply weights if train already did.
    # Our train.py already multiplies matrices by weights before hstack.
    # So here: just transform raw blocks (no extra multiplication).
    # (If you later change to store unweighted blocks, then you'd reapply here.)
    # We'll keep it simple and consistent with current train.py.

    # ---- genres multi-hot
    X_genres = mlb.transform(split_genres(df["genres"]))

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

    if "rating_bucket" not in dd.columns:
        dd["rating_bucket"] = pd.cut(
            pd.to_numeric(dd["averageRating"], errors="coerce"),
            bins=[0, 3, 6, 8, 10],
            labels=["poor", "okay", "good", "great"],
        )

    if "runtime_bucket" not in dd.columns:
        dd["runtime_bucket"] = pd.cut(
            pd.to_numeric(dd["runtimeMinutes"], errors="coerce"),
            bins=[0, 30, 60, 90, 120, 180, 300, 10_000],
            labels=["<30", "30-60", "60-90", "90-120", "120-180", "180-300", ">300"],
        )

    cat_cols = bundle.get("cat_cols", {})
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

    return X


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


def main() -> None:
    p = argparse.ArgumentParser(description="Recommend movies by title (rerank via config.py).")
    p.add_argument("--title", required=True, help="Movie title to search (case-insensitive)")
    p.add_argument("--k", type=int, default=10, help="Number of recommendations to return")
    p.add_argument("--model", default=str(cfg.MODEL_BUNDLE_PATH), help="Path to model bundle")
    p.add_argument("--data", default=str(cfg.DATA_PATH), help="Path to processed dataset parquet")
    args = p.parse_args()

    bundle_path = PROJECT_ROOT / Path(args.model)
    data_path = PROJECT_ROOT / Path(args.data)

    if not bundle_path.exists():
        print(f"❌ Model bundle not found: {bundle_path}")
        return
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        return

    bundle = joblib.load(bundle_path)
    df = pd.read_parquet(data_path).reset_index(drop=True)
    df = ensure_engineered_cols(df, bundle)

    meta = bundle.get("df_meta")
    if meta is None or len(meta) == 0:
        print("❌ Bundle missing df_meta.")
        return

    row, idx = pick_query_row(df, args.title)
    if idx < 0 or row.empty:
        print("❌ Film introuvable")
        return

    Xq = build_features(row, bundle)
    nn = bundle.get("nn")
    if nn is None:
        print("❌ Bundle missing NearestNeighbors (nn).")
        return

    # Candidate pool for rerank (from config)
    pool = max(int(cfg.PRED_POOL_MIN), int(args.k) * int(cfg.PRED_POOL_MULT))
    distances, indices = nn.kneighbors(Xq, n_neighbors=pool)

    q_title = str(row.iloc[0].get("primaryTitle", ""))
    q_year = int(pd.to_numeric(row.iloc[0].get("startYear", 0), errors="coerce") or 0)

    # weights (from config)
    w_sim = float(cfg.W_SIM)
    w_rating = float(cfg.W_RATING)
    w_votes = float(cfg.W_VOTES)
    w_year = float(cfg.W_YEAR)

    votes_norm_max = float(cfg.VOTES_NORM_MAX)
    year_gap_max = float(cfg.YEAR_GAP_MAX)

    cands = []
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

        score = (w_sim * sim) + (w_rating * r_norm) + (w_votes * votes_norm) + (w_year * year_score)

        cands.append(
            {
                "score": score,
                "sim": sim,
                "rating": r,
                "votes": votes,
                "year": y,
                "gap": gap,
                "r_norm": r_norm,
                "votes_norm": votes_norm,
                "year_score": year_score,
                "title": str(m.get("primaryTitle", "")),
            }
        )

    cands.sort(key=lambda d: d["score"], reverse=True)
    top = cands[: int(args.k)]

    print(f"\n🎬 {q_title} ({q_year})")
    print("-" * 75)
    print(
        f"rerank: pool={pool} | weights sim={w_sim:.2f} rating={w_rating:.2f} votes={w_votes:.2f} year={w_year:.2f} "
        f"| votes_norm_max={int(votes_norm_max)} year_gap_max={int(year_gap_max)}"
    )
    print("")

    print(f"Recommendations for '{q_title}' ({q_year})")
    for rank, d in enumerate(top, start=1):
        print(
            f"{rank:2d}. {d['title']} ({d['year']}) - "
            f"score: {d['score']:.4f}, sim: {d['sim']:.4f}, rating: {d['rating']:.2f} ({int(d['votes'])} votes), gap: {d['gap']}"
        )
        if cfg.PRED_DEBUG and rank <= int(cfg.PRED_DEBUG_TOPN):
            print(
                f"    components: r_norm={d['r_norm']:.4f} votes_norm={d['votes_norm']:.4f} year_score={d['year_score']:.4f}"
            )


if __name__ == "__main__":
    main()