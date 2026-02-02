# scripts/eval.py
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config as cfg


# ----------------------------
# Helpers
# ----------------------------
def _split_genres(s: Any) -> set[str]:
    if s is None:
        return set()
    txt = str(s).replace(",", " ").strip()
    if not txt:
        return set()
    return set([g for g in txt.split() if g])


def _decade(y: Any) -> int | None:
    try:
        yy = int(y)
        if yy <= 0:
            return None
        return (yy // 10) * 10
    except Exception:
        return None


def run_predict_json(title: str, k: int, model: Path, data: Path) -> dict:
    cmd = [
        "python",
        "scripts/predict.py",
        "--title",
        title,
        "--k",
        str(k),
        "--model",
        str(model),
        "--data",
        str(data),
        "--json",
    ]
    raw = subprocess.check_output(cmd, text=True)
    raw = raw.strip()

    # Sometimes predicts prints other stuff; try to grab the last JSON object
    # If your predicts is clean (only JSON with --json), this is trivial.
    if not raw.startswith("{"):
        # take last line that looks like json
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        json_line = ""
        for ln in reversed(lines):
            if ln.startswith("{") and ln.endswith("}"):
                json_line = ln
                break
        if not json_line:
            raise ValueError(f"predicts.py did not return pure JSON. Got:\n{raw[:500]}")
        raw = json_line

    return json.loads(raw)


def build_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a lookup table keyed by normalized (title_lower, year).
    If duplicates exist, keep the row with highest numVotes.
    """
    out = df.copy()

    out["primaryTitle"] = out.get("primaryTitle", "").fillna("").astype(str)
    out["startYear"] = pd.to_numeric(out.get("startYear", np.nan), errors="coerce")
    out["numVotes"] = pd.to_numeric(out.get("numVotes", 0), errors="coerce").fillna(0)

    out["title_lower"] = out["primaryTitle"].str.lower().str.strip()
    out["year_int"] = out["startYear"].fillna(0).astype(int)

    # Keep best by votes
    out = out.sort_values("numVotes", ascending=False)
    out = out.drop_duplicates(subset=["title_lower", "year_int"], keep="first")

    # Keep only needed cols for fast access
    keep = [
        "title_lower",
        "year_int",
        "genres",
        "numVotes",
        "averageRating",
        "rating_bayes",
        "confidence",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep].set_index(["title_lower", "year_int"], drop=True)


def fetch_meta(lookup: pd.DataFrame, title: str, year: int) -> dict[str, Any] | None:
    key = (title.lower().strip(), int(year))
    if key in lookup.index:
        row = lookup.loc[key]
        # row may be Series
        return row.to_dict() if hasattr(row, "to_dict") else dict(row)
    return None


@dataclass
class QueryEval:
    title: str
    year: int
    k: int
    mean_genre_jaccard: float
    mean_year_gap: float
    unique_decades: int
    unique_titles: int
    avg_log1p_votes: float
    mean_confidence: float | None
    coverage_in_parquet: float  # fraction of recs found in parquet


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return float(inter / uni) if uni else 0.0


def eval_one(lookup: pd.DataFrame, payload: dict) -> tuple[QueryEval, pd.DataFrame]:
    q = payload["query"]
    q_title = str(q.get("title", ""))
    q_year = int(q.get("year", 0) or 0)
    recs = payload.get("recs", [])
    k = int(payload.get("k", len(recs)))

    # query genres (from parquet if possible)
    q_meta = fetch_meta(lookup, q_title, q_year) if q_year else None
    q_genres = _split_genres(q_meta.get("genres")) if q_meta else set()

    rows = []
    found = 0

    for r in recs:
        title = str(r.get("title", ""))
        year = int(r.get("year", 0) or 0)
        meta = fetch_meta(lookup, title, year) if year else None
        if meta is not None:
            found += 1

        genres = _split_genres(meta.get("genres")) if meta else set()
        jac = jaccard(q_genres, genres) if q_genres else np.nan

        gap = float(r.get("gap", abs(year - q_year) if (q_year and year) else np.nan))
        votes = meta.get("numVotes") if meta else r.get("votes", np.nan)
        conf = meta.get("confidence") if meta and "confidence" in meta else None

        rows.append(
            {
                "title": title,
                "year": year,
                "score": float(r.get("score", np.nan)),
                "sim": float(r.get("sim", np.nan)),
                "rating": float(r.get("rating", np.nan)),
                "votes": float(votes) if votes is not None else np.nan,
                "gap": gap,
                "genre_jaccard": jac,
                "decade": _decade(year),
                "confidence": float(conf) if conf is not None else np.nan,
            }
        )

    df_recs = pd.DataFrame(rows)

    # Metrics
    mean_jac = float(np.nanmean(df_recs["genre_jaccard"].to_numpy())) if len(df_recs) else 0.0
    mean_gap = float(np.nanmean(df_recs["gap"].to_numpy())) if len(df_recs) else 0.0
    unique_dec = int(pd.Series(df_recs["decade"]).dropna().nunique()) if len(df_recs) else 0
    unique_titles = int(pd.Series(df_recs["title"]).nunique()) if len(df_recs) else 0

    v = df_recs["votes"].to_numpy(dtype=float) if "votes" in df_recs.columns else np.array([])
    avg_logvotes = float(np.nanmean(np.log1p(v))) if v.size else 0.0

    conf_arr = df_recs["confidence"].to_numpy(dtype=float) if "confidence" in df_recs.columns else np.array([])
    mean_conf = float(np.nanmean(conf_arr)) if np.isfinite(conf_arr).any() else None

    coverage = float(found / len(df_recs)) if len(df_recs) else 0.0

    qe = QueryEval(
        title=q_title,
        year=q_year,
        k=k,
        mean_genre_jaccard=mean_jac,
        mean_year_gap=mean_gap,
        unique_decades=unique_dec,
        unique_titles=unique_titles,
        avg_log1p_votes=avg_logvotes,
        mean_confidence=mean_conf,
        coverage_in_parquet=coverage,
    )
    return qe, df_recs


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate recommender outputs (proxy metrics) from multiple query titles.")
    ap.add_argument("--title", action="append", required=True, help="Repeatable: --title 'Alien' --title 'Toy Story'")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model", default=str(cfg.MODEL_BUNDLE_PATH))
    ap.add_argument("--data", default=str(cfg.DATA_PATH))
    ap.add_argument("--out_csv", default="", help="Optional: save per-query summary CSV")
    ap.add_argument("--out_recs_csv", default="", help="Optional: save all rec rows CSV")
    args = ap.parse_args()

    model_path = (PROJECT_ROOT / Path(args.model)).resolve()
    data_path = (PROJECT_ROOT / Path(args.data)).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"Loading parquet once: {data_path}")
    df = pd.read_parquet(data_path)
    lookup = build_lookup(df)
    print(f"Lookup size: {len(lookup):,} (unique title/year pairs)\n")

    summaries: list[dict[str, Any]] = []
    all_recs: list[pd.DataFrame] = []

    for t in args.title:
        payload = run_predict_json(t, args.k, model_path, data_path)
        qe, recs_df = eval_one(lookup, payload)
        all_recs.append(recs_df.assign(query_title=qe.title, query_year=qe.year))

        print(f"🎬 Query: {qe.title} ({qe.year}) | k={qe.k}")
        print(f"  - mean genre Jaccard: {qe.mean_genre_jaccard:.3f}  (higher = more coherent genres)")
        print(f"  - mean year gap:      {qe.mean_year_gap:.2f} years (lower = closer era)")
        print(f"  - unique decades:     {qe.unique_decades}      (higher = more diverse)")
        print(f"  - unique titles:      {qe.unique_titles}/{qe.k}")
        print(f"  - avg log1p(votes):   {qe.avg_log1p_votes:.3f} (higher = more popular bias)")
        if qe.mean_confidence is not None:
            print(f"  - mean confidence:    {qe.mean_confidence:.3f}")
        print(f"  - parquet coverage:   {qe.coverage_in_parquet:.0%}\n")

        # quick top5 table
        show = recs_df.copy()
        show["votes"] = show["votes"].fillna(0).astype(int)
        show["genre_jaccard"] = show["genre_jaccard"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
        show["score"] = show["score"].map(lambda x: f"{x:.4f}")
        show["sim"] = show["sim"].map(lambda x: f"{x:.4f}")
        show["rating"] = show["rating"].map(lambda x: f"{x:.2f}")
        cols = ["title", "year", "score", "sim", "rating", "votes", "gap", "genre_jaccard"]
        print(show[cols].head(8).to_string(index=False))
        print("\n" + "=" * 90 + "\n")

        summaries.append(
            {
                "query_title": qe.title,
                "query_year": qe.year,
                "k": qe.k,
                "mean_genre_jaccard": qe.mean_genre_jaccard,
                "mean_year_gap": qe.mean_year_gap,
                "unique_decades": qe.unique_decades,
                "unique_titles": qe.unique_titles,
                "avg_log1p_votes": qe.avg_log1p_votes,
                "mean_confidence": qe.mean_confidence if qe.mean_confidence is not None else np.nan,
                "parquet_coverage": qe.coverage_in_parquet,
            }
        )

    sum_df = pd.DataFrame(summaries)
    if len(sum_df):
        print("📌 Global summary (mean over queries):")
        print(sum_df.drop(columns=["query_title"]).mean(numeric_only=True).to_string())
        print("")

    if args.out_csv:
        outp = (PROJECT_ROOT / Path(args.out_csv)).resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        sum_df.to_csv(outp, index=False)
        print(f"Saved summary CSV: {outp}")

    if args.out_recs_csv:
        outp = (PROJECT_ROOT / Path(args.out_recs_csv)).resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(all_recs, ignore_index=True).to_csv(outp, index=False)
        print(f"Saved recs CSV: {outp}")


if __name__ == "__main__":
    main()