# scripts/sweep_weights.py
from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from scripts.predict import ensure_engineered_cols, build_features, pick_query_row


# -------------------------
# Small utils
# -------------------------
def _split_genres(x: Any) -> set[str]:
    if x is None:
        return set()
    s = str(x).replace(",", " ").strip()
    if not s:
        return set()
    return set([g for g in s.split() if g])


def _decade(y: Any) -> int | None:
    try:
        yy = int(y)
        if yy <= 0:
            return None
        return (yy // 10) * 10
    except Exception:
        return None


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    if u == 0:
        return 0.0
    return float(len(a & b) / u)


def parse_list(s: str) -> list[float]:
    # "0.55,0.6,0.65" -> [0.55, 0.6, 0.65]
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def build_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Index by (title_lower, year_int). Keep the version with most votes if duplicates.
    """
    x = df.copy()
    x["primaryTitle"] = x.get("primaryTitle", "").fillna("").astype(str)
    x["startYear"] = pd.to_numeric(x.get("startYear", np.nan), errors="coerce")
    x["numVotes"] = pd.to_numeric(x.get("numVotes", 0), errors="coerce").fillna(0)

    x["title_lower"] = x["primaryTitle"].str.lower().str.strip()
    x["year_int"] = x["startYear"].fillna(0).astype(int)

    x = x.sort_values("numVotes", ascending=False)
    x = x.drop_duplicates(subset=["title_lower", "year_int"], keep="first")

    keep = ["title_lower", "year_int", "genres", "numVotes", "averageRating", "rating_bayes", "confidence"]
    keep = [c for c in keep if c in x.columns]
    return x[keep].set_index(["title_lower", "year_int"], drop=True)


def get_meta(lookup: pd.DataFrame, title: str, year: int) -> dict[str, Any] | None:
    key = (title.lower().strip(), int(year))
    if key in lookup.index:
        row = lookup.loc[key]
        return row.to_dict() if hasattr(row, "to_dict") else dict(row)
    return None


@dataclass
class Cand:
    title: str
    year: int
    sim: float
    r_norm: float
    votes_norm: float
    year_score: float
    conf_penalty: float


def build_candidates_for_query(
    *,
    df: pd.DataFrame,
    meta: pd.DataFrame,
    lookup: pd.DataFrame,
    bundle: dict,
    title: str,
    pool: int,
    votes_norm_max: float,
    year_gap_max: float,
) -> tuple[dict[str, Any], list[Cand], set[str]]:
    """
    Returns:
      query_info, candidates list, query genres set
    """
    row, idx = pick_query_row(df, title)
    if idx < 0 or row.empty:
        raise ValueError(f"Film introuvable: {title}")

    q_title = str(row.iloc[0].get("primaryTitle", ""))
    q_year = int(pd.to_numeric(row.iloc[0].get("startYear", 0), errors="coerce") or 0)

    # query genres (from parquet lookup)
    q_meta = get_meta(lookup, q_title, q_year) if q_year else None
    q_genres = _split_genres(q_meta.get("genres")) if q_meta else set()

    Xq = build_features(row, bundle)
    nn = bundle["nn"]
    distances, indices = nn.kneighbors(Xq, n_neighbors=pool)

    cands: list[Cand] = []

    for dist, i in zip(distances[0], indices[0]):
        i = int(i)
        if i == int(idx):
            continue

        m = meta.iloc[i]
        cand_title = str(m.get("primaryTitle", ""))
        cand_year = int(pd.to_numeric(m.get("startYear", 0), errors="coerce") or 0)

        sim = float(1 - dist)

        # rating bayes if present
        r = m.get("rating_bayes", m.get("averageRating", 0.0))
        r = float(pd.to_numeric(r, errors="coerce") or 0.0)
        r_norm = r / 10.0

        votes = float(pd.to_numeric(m.get("numVotes", 0.0), errors="coerce") or 0.0)
        votes_norm = float(np.log1p(votes) / np.log1p(votes_norm_max)) if votes_norm_max > 0 else 0.0
        votes_norm = float(min(1.0, max(0.0, votes_norm)))

        gap = abs(cand_year - q_year) if (q_year and cand_year) else 0
        year_score = max(0.0, 1.0 - (gap / year_gap_max)) if year_gap_max > 0 else 0.0

        conf = float(m.get("confidence", 0.0) or 0.0)
        conf_penalty = 0.85 + 0.15 * conf  # [0.85, 1.0]

        cands.append(
            Cand(
                title=cand_title,
                year=cand_year,
                sim=sim,
                r_norm=r_norm,
                votes_norm=votes_norm,
                year_score=year_score,
                conf_penalty=conf_penalty,
            )
        )

    qinfo = {"title": q_title, "year": q_year}
    return qinfo, cands, q_genres


def score_candidates(
    cands: list[Cand],
    *,
    w_sim: float,
    w_rating: float,
    w_votes: float,
    w_year: float,
    k: int,
) -> list[dict[str, Any]]:
    scored = []
    for c in cands:
        score = c.conf_penalty * (
            w_sim * c.sim
            + w_rating * c.r_norm
            + w_votes * c.votes_norm
            + w_year * c.year_score
        )
        scored.append(
            {
                "title": c.title,
                "year": c.year,
                "score": float(score),
                "sim": float(c.sim),
                "r_norm": float(c.r_norm),
                "votes_norm": float(c.votes_norm),
                "year_score": float(c.year_score),
            }
        )
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:k]


def eval_recs_proxy(
    *,
    lookup: pd.DataFrame,
    qinfo: dict[str, Any],
    q_genres: set[str],
    recs: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Proxy eval:
      - mean genre jaccard (higher better)
      - mean year gap (lower better)
      - unique decades (higher better)
      - popularity bias (avg log1p votes) (lower is “less blockbuster-biased”)
      - coverage (how many found in parquet lookup)
    """
    q_year = int(qinfo.get("year", 0) or 0)

    jac_vals = []
    gaps = []
    decades = []
    logvotes = []
    found = 0

    for r in recs:
        title = str(r["title"])
        year = int(r["year"] or 0)
        gaps.append(abs(year - q_year) if (q_year and year) else np.nan)
        decades.append(_decade(year))

        meta = get_meta(lookup, title, year)
        if meta is not None:
            found += 1
            g = _split_genres(meta.get("genres"))
            jac_vals.append(jaccard(q_genres, g) if q_genres else np.nan)
            v = float(meta.get("numVotes", np.nan))
            logvotes.append(np.log1p(v) if not math.isnan(v) else np.nan)
        else:
            jac_vals.append(np.nan)
            logvotes.append(np.nan)

    jac = float(np.nanmean(np.array(jac_vals, dtype=float))) if len(jac_vals) else 0.0
    gap = float(np.nanmean(np.array(gaps, dtype=float))) if len(gaps) else 0.0
    uniq_dec = float(pd.Series(decades).dropna().nunique()) if len(decades) else 0.0
    pop = float(np.nanmean(np.array(logvotes, dtype=float))) if len(logvotes) else 0.0
    coverage = float(found / len(recs)) if recs else 0.0

    return {
        "mean_genre_jaccard": jac,
        "mean_year_gap": gap,
        "unique_decades": uniq_dec,
        "avg_log1p_votes": pop,
        "coverage": coverage,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep rerank weights without retraining.")
    ap.add_argument("--title", action="append", required=True, help="Repeatable: --title 'Alien' --title 'Toy Story'")
    ap.add_argument("--k", type=int, default=10, help="Top-k recs")
    ap.add_argument("--pool_mult", type=int, default=5, help="Candidate pool = max(cfg.PRED_POOL_MIN, k*pool_mult)")
    ap.add_argument("--model", default=str(cfg.MODEL_BUNDLE_PATH))
    ap.add_argument("--data", default=str(cfg.DATA_PATH))

    # Weight grids
    ap.add_argument("--w_sim", default="0.55,0.59,0.63", help="Comma list")
    ap.add_argument("--w_rating", default="0.10,0.15,0.20", help="Comma list")
    ap.add_argument("--w_votes", default="-0.25,-0.15,-0.05", help="Comma list (usually negative)")
    ap.add_argument("--w_year", default="0.00,0.03,0.06", help="Comma list")

    # How to rank configs
    ap.add_argument("--objective", default="balanced", choices=["balanced", "coherence", "diversity", "anti_popularity"])
    ap.add_argument("--out_csv", default="outputs/sweep_weights.csv")
    ap.add_argument("--show_top", type=int, default=15)
    args = ap.parse_args()

    model_path = (PROJECT_ROOT / Path(args.model)).resolve()
    data_path = (PROJECT_ROOT / Path(args.data)).resolve()
    out_csv = (PROJECT_ROOT / Path(args.out_csv)).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"Loading bundle:  {model_path}")
    bundle = joblib.load(model_path)

    print(f"Loading parquet: {data_path} (once)")
    df = pd.read_parquet(data_path).reset_index(drop=True)
    df = ensure_engineered_cols(df, bundle)

    meta = bundle.get("df_meta")
    if meta is None or len(meta) == 0:
        raise ValueError("Bundle missing df_meta")

    lookup = build_lookup(df)

    votes_norm_max = float(getattr(cfg, "VOTES_NORM_MAX", 2_500_000))
    year_gap_max = float(getattr(cfg, "YEAR_GAP_MAX", 30))

    k = int(args.k)
    pool = max(int(getattr(cfg, "PRED_POOL_MIN", 50)), k * int(args.pool_mult))
    print(f"Using pool={pool} | k={k}\n")

    w_sims = parse_list(args.w_sim)
    w_rats = parse_list(args.w_rating)
    w_vts = parse_list(args.w_votes)
    w_yrs = parse_list(args.w_year)

    grid = list(itertools.product(w_sims, w_rats, w_vts, w_yrs))
    print(f"Grid size: {len(grid)} configs\n")

    # Precompute candidates per query (the expensive part)
    queries = []
    for t in args.title:
        qinfo, cands, q_genres = build_candidates_for_query(
            df=df,
            meta=meta,
            lookup=lookup,
            bundle=bundle,
            title=t,
            pool=pool,
            votes_norm_max=votes_norm_max,
            year_gap_max=year_gap_max,
        )
        queries.append((qinfo, cands, q_genres))

    rows = []

    for (w_sim, w_rating, w_votes, w_year) in grid:
        perq = []
        for (qinfo, cands, q_genres) in queries:
            recs = score_candidates(cands, w_sim=w_sim, w_rating=w_rating, w_votes=w_votes, w_year=w_year, k=k)
            m = eval_recs_proxy(lookup=lookup, qinfo=qinfo, q_genres=q_genres, recs=recs)
            perq.append(m)

        # aggregate over queries
        agg = {k: float(np.nanmean([d[k] for d in perq])) for k in perq[0].keys()}

        # objective score
        # (tweakable but sensible defaults)
        if args.objective == "coherence":
            obj = agg["mean_genre_jaccard"]
        elif args.objective == "diversity":
            obj = agg["unique_decades"]
        elif args.objective == "anti_popularity":
            obj = -agg["avg_log1p_votes"]
        else:
            # balanced:
            #  + coherence
            #  + diversity (scaled)
            #  - year gap (scaled)
            #  - popularity bias (scaled)
            obj = (
                1.20 * agg["mean_genre_jaccard"]
                + 0.05 * agg["unique_decades"]
                - 0.02 * agg["mean_year_gap"]
                - 0.03 * agg["avg_log1p_votes"]
            )

        rows.append(
            {
                "w_sim": w_sim,
                "w_rating": w_rating,
                "w_votes": w_votes,
                "w_year": w_year,
                "objective": float(obj),
                **agg,
            }
        )

    res = pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)
    res.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}\n")

    topn = int(args.show_top)
    show = res.head(topn).copy()
    for c in ["objective", "mean_genre_jaccard", "mean_year_gap", "unique_decades", "avg_log1p_votes", "coverage"]:
        if c in show.columns:
            show[c] = show[c].map(lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.floating)) else x)

    print(f"🏁 TOP {topn} configs (objective={args.objective})")
    cols = ["w_sim", "w_rating", "w_votes", "w_year", "objective",
            "mean_genre_jaccard", "mean_year_gap", "unique_decades", "avg_log1p_votes", "coverage"]
    cols = [c for c in cols if c in show.columns]
    print(show[cols].to_string(index=False))


if __name__ == "__main__":
    main()