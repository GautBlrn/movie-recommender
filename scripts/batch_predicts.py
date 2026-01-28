from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

# project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg  # noqa: E402

# Import helpers from predicts.py (so we don't duplicate logic)
# Your predicts.py must expose these functions (they do in our final version):
# - ensure_engineered_cols
# - pick_query_row
# - build_features
from scripts.predicts import ensure_engineered_cols, pick_query_row, build_features  # noqa: E402


def load_titles(args) -> List[str]:
    titles: List[str] = []
    if args.titles:
        titles.extend(args.titles)
    if args.file:
        p = Path(args.file)
        if not p.exists():
            raise FileNotFoundError(f"titles file not found: {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t and not t.startswith("#"):
                titles.append(t)
    # de-dup, keep order
    seen = set()
    out = []
    for t in titles:
        k = t.lower().strip()
        if k not in seen:
            out.append(t)
            seen.add(k)
    return out


def rerank_candidates(
    meta: pd.DataFrame,
    q_title: str,
    q_year: int,
    idx: int,
    distances: np.ndarray,
    indices: np.ndarray,
    k: int,
) -> list[dict]:
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
    return cands[:k]


def main() -> None:
    p = argparse.ArgumentParser(description="Batch test recommendations for multiple titles.")
    p.add_argument("--titles", nargs="*", help="List of titles (space separated).")
    p.add_argument("--file", help="Text file with one title per line (supports # comments).")
    p.add_argument("--k", type=int, default=10, help="Top-K recommendations per title.")
    p.add_argument("--model", default=str(cfg.MODEL_BUNDLE_PATH), help="Path to model bundle.")
    p.add_argument("--data", default=str(cfg.DATA_PATH), help="Path to processed parquet.")
    p.add_argument("--pool", type=int, default=None, help="Candidate pool (default: config logic).")
    p.add_argument("--csv", default=None, help="Optional output CSV path (flat results).")
    args = p.parse_args()

    titles = load_titles(args)
    if not titles:
        print("❌ No titles provided. Use --titles or --file.")
        return

    bundle_path = PROJECT_ROOT / Path(args.model)
    data_path = PROJECT_ROOT / Path(args.data)
    if not bundle_path.exists():
        print(f"❌ Model bundle not found: {bundle_path}")
        return
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        return

    print("Loading bundle + dataset...")
    bundle = joblib.load(bundle_path)
    df = pd.read_parquet(data_path).reset_index(drop=True)
    df = ensure_engineered_cols(df, bundle)

    meta = bundle.get("df_meta")
    nn = bundle.get("nn")
    if meta is None or nn is None:
        print("❌ Bundle missing df_meta or nn.")
        return

    pool = args.pool if args.pool is not None else max(int(cfg.PRED_POOL_MIN), int(args.k) * int(cfg.PRED_POOL_MULT))

    rows_out = []

    for t in titles:
        row, idx = pick_query_row(df, t)
        if idx < 0 or row.empty:
            print(f"\n❌ Not found: {t}")
            continue

        q_title = str(row.iloc[0].get("primaryTitle", ""))
        q_year = int(pd.to_numeric(row.iloc[0].get("startYear", 0), errors="coerce") or 0)

        Xq = build_features(row, bundle)
        distances, indices = nn.kneighbors(Xq, n_neighbors=pool)

        top = rerank_candidates(meta, q_title, q_year, idx, distances, indices, k=int(args.k))

        print(f"\n🎬 {q_title} ({q_year})")
        print("-" * 70)
        for rank, d in enumerate(top, start=1):
            print(
                f"{rank:2d}. {d['title']} ({d['year']}) "
                f"| score={d['score']:.4f} sim={d['sim']:.4f} rating={d['rating']:.2f} votes={int(d['votes'])} gap={d['gap']}"
            )

            rows_out.append(
                {
                    "query": q_title,
                    "query_year": q_year,
                    "rank": rank,
                    "rec_title": d["title"],
                    "rec_year": d["year"],
                    "score": d["score"],
                    "sim": d["sim"],
                    "rating": d["rating"],
                    "votes": int(d["votes"]),
                    "gap": d["gap"],
                }
            )

    if args.csv:
        outp = Path(args.csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(outp, index=False, encoding="utf-8")
        print(f"\n✅ CSV saved: {outp}")


if __name__ == "__main__":
    main()