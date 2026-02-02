# scripts/audit_dataset_v3.py
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

# ---- load config ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config as cfg  # noqa: E402


# -------------------------
# Utils
# -------------------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def html_escape(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log(p, 2) + (1 - p) * math.log(1 - p, 2)))


def df_to_html_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "<div class='note'>(no data)</div>"

    d = df.head(max_rows).copy()
    cols = d.columns.tolist()

    out = []
    out.append("<div class='tablewrap'>")
    out.append("<table><thead><tr>")
    out.append("".join(f"<th>{html_escape(c)}</th>" for c in cols))
    out.append("</tr></thead><tbody>")

    for row in d.itertuples(index=False):
        out.append("<tr>" + "".join(f"<td>{html_escape(v)}</td>" for v in row) + "</tr>")

    out.append("</tbody></table>")
    if len(df) > max_rows:
        out.append(f"<div class='note'>Affiché: {max_rows} / {len(df)} lignes</div>")
    out.append("</div>")
    return "\n".join(out)


def write_html(out_path: Path, sections: list[str], toc: list[tuple[str, str]], subtitle: str) -> None:
    css = """
    :root{
      --bg:#0b0f1a; --card:#11182a; --text:#e9eefc; --muted:#a8b3d6;
      --border:rgba(255,255,255,.08);
      --accent:#7aa2ff; --ok:#38d39f; --warn:#ffcc66; --bad:#ff6b6b;
      --code:#0c1324;
    }
    @media (prefers-color-scheme: light){
      :root{
        --bg:#f7f8fb; --card:#ffffff; --text:#111827; --muted:#6b7280;
        --border:rgba(17,24,39,.10);
        --accent:#2563eb; --ok:#059669; --warn:#d97706; --bad:#dc2626;
        --code:#f3f4f6;
      }
    }
    *{ box-sizing:border-box; }
    body{
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      background:var(--bg); color:var(--text);
      margin:0; padding:0;
    }
    a{ color:var(--accent); text-decoration:none; }
    a:hover{ text-decoration:underline; }
    .wrap{ max-width:1180px; margin:0 auto; padding:24px; }
    header{
      position:sticky; top:0; z-index:10;
      backdrop-filter: blur(10px);
      background: color-mix(in srgb, var(--bg) 70%, transparent);
      border-bottom:1px solid var(--border);
    }
    header .wrap{ display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 24px; }
    h1{ font-size:22px; margin:0; }
    .meta{ font-size:13px; color:var(--muted); line-height:1.4; }
    .grid{ display:grid; grid-template-columns: 340px 1fr; gap:16px; margin-top:16px; }
    @media (max-width: 980px){ .grid{ grid-template-columns:1fr; } }
    .card{
      background:var(--card);
      border:1px solid var(--border);
      border-radius:14px;
      padding:16px;
      box-shadow: 0 10px 30px rgba(0,0,0,.12);
    }
    .toc a{ display:block; padding:8px 10px; border-radius:10px; color:var(--text); }
    .toc a:hover{ background: color-mix(in srgb, var(--accent) 10%, transparent); }
    .section{ margin-bottom:18px; }
    .section h2{ font-size:18px; margin:0 0 10px; }
    .section h3{ font-size:16px; margin:16px 0 8px; }
    .section h4{ font-size:12px; margin:14px 0 8px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
    .badge{ display:inline-block; padding:3px 9px; border-radius:999px; font-size:12px; margin-right:6px; border:1px solid var(--border);}
    .badge.ok{ background: color-mix(in srgb, var(--ok) 18%, transparent); color: var(--ok); border-color: color-mix(in srgb, var(--ok) 45%, transparent); }
    .badge.warn{ background: color-mix(in srgb, var(--warn) 18%, transparent); color: var(--warn); border-color: color-mix(in srgb, var(--warn) 45%, transparent); }
    .badge.bad{ background: color-mix(in srgb, var(--bad) 18%, transparent); color: var(--bad); border-color: color-mix(in srgb, var(--bad) 45%, transparent); }
    .note{ color:var(--muted); font-size:13px; line-height:1.5; }
    .kpis{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
    .pill{ font-size:12px; padding:6px 10px; border-radius:999px; background: color-mix(in srgb, var(--bg) 35%, transparent); border:1px solid var(--border); }
    img{ max-width:100%; border-radius:12px; border:1px solid var(--border); display:block; }
    .imggrid{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    @media (max-width: 860px){ .imggrid{ grid-template-columns:1fr; } }

    .tablewrap{ overflow:auto; border-radius:12px; border:1px solid var(--border); }
    table{ width:100%; border-collapse: collapse; font-size:13px; }
    th, td{ padding:10px 10px; border-bottom:1px solid var(--border); vertical-align:top; white-space:nowrap; }
    th{ text-align:left; color:var(--muted); font-weight:600; background: color-mix(in srgb, var(--bg) 40%, transparent); position:sticky; top:0; }
    tr:hover td{ background: color-mix(in srgb, var(--accent) 6%, transparent); }

    code, pre{
      background:var(--code);
      border:1px solid var(--border);
      border-radius:12px;
    }
    pre{ padding:12px; overflow:auto; }
    code{ padding:2px 6px; }
    .toplink{
      position: fixed; bottom: 18px; right: 18px;
      background: var(--card); border:1px solid var(--border);
      border-radius: 999px; padding:10px 12px;
      box-shadow: 0 10px 30px rgba(0,0,0,.18);
      font-size: 13px;
    }
    """

    toc_html = ["<div class='card toc'><h2>Sommaire</h2>"]
    for anchor, title in toc:
        toc_html.append(f"<a href='#{anchor}'>• {html_escape(title)}</a>")
    toc_html.append("<div class='note' style='margin-top:10px;'>Astuce : Ctrl+F marche nickel ici.</div>")
    toc_html.append("</div>")

    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>",
        "<title>Dataset Audit Report</title>",
        f"<style>{css}</style>",
        "</head><body>",
        "<a id='top'></a>",
        "<header><div class='wrap'>",
        "<div>",
        "<h1>Dataset Audit Report</h1>",
        f"<div class='meta'>{html_escape(subtitle)}</div>",
        "</div>",
        f"<div class='meta'>Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",
        "</div></header>",
        "<div class='wrap'>",
        "<div class='grid'>",
        "".join(toc_html),
        "<div class='card'>",
    ]

    html.extend(sections)

    html.extend([
        "</div></div></div>",
        "<a class='toplink' href='#top'>↑ Haut</a>",
        "</body></html>",
    ])

    out_path.write_text("\n".join(html), encoding="utf-8")


def plot_hist(x: np.ndarray, out: Path, title: str, bins: int = 60, logy: bool = False) -> None:
    plt.figure()
    if x.size == 0:
        plt.title(title + " (no data)")
    else:
        plt.hist(x, bins=bins)
        plt.title(title)
        if logy:
            plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def plot_bar(items: list[tuple[str, float]], out: Path, title: str, topn: int = 20) -> None:
    plt.figure(figsize=(11, 5))
    items = items[:topn]
    if not items:
        plt.title(title + " (no data)")
        plt.tight_layout()
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        return
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def tokens_from_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.split()


def df_table(tokens_per_row: pd.Series) -> pd.DataFrame:
    c = Counter()
    for row in tokens_per_row:
        if not row:
            continue
        for t in set(row):
            if t:
                c[t] += 1
    return pd.DataFrame({"token": list(c.keys()), "df": list(c.values())})


def add_token_metrics(tbl: pd.DataFrame, n_docs: int) -> pd.DataFrame:
    t = tbl.copy()
    t["df_ratio"] = t["df"] / float(n_docs)
    t["idf"] = np.log((n_docs + 1) / (t["df"] + 1)) + 1.0
    t["H_bin"] = t["df_ratio"].apply(binary_entropy)
    t["token_value"] = t["idf"] * t["H_bin"]
    t["prefix"] = t["token"].apply(lambda s: s.split(":", 1)[0] if ":" in s else "(no_prefix)")
    return t


def kpi_pills(pairs: list[tuple[str, str]]) -> str:
    return "<div class='kpis'>" + "".join(f"<div class='pill'><b>{html_escape(k)}</b>: {html_escape(v)}</div>" for k, v in pairs) + "</div>"


# -------------------------
# Main
# -------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Audit processed dataset + token engineering metrics (pretty HTML).")
    p.add_argument("--data", default=str(cfg.DATA_PATH), help="Parquet path (relative to project root)")
    p.add_argument("--out_dir", default="outputs/audit_report_v3", help="Output folder (relative to project root)")
    p.add_argument("--topn", type=int, default=15, help="Top N display")
    p.add_argument("--too_common_df", type=float, default=0.40, help="Flag tokens with DF ratio above this")
    p.add_argument("--too_rare_df", type=int, default=2, help="Flag tokens with DF <= this")
    p.add_argument("--good_df_min", type=float, default=0.02, help="Good zone DF ratio min (heuristic)")
    p.add_argument("--good_df_max", type=float, default=0.25, help="Good zone DF ratio max (heuristic)")
    args = p.parse_args()

    data_path = (PROJECT_ROOT / Path(args.data)).resolve()
    out_dir = (PROJECT_ROOT / Path(args.out_dir)).resolve()
    figs = out_dir / "figures"
    tables = out_dir / "tables"
    safe_mkdir(figs)
    safe_mkdir(tables)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_parquet(data_path).reset_index(drop=True)
    n = len(df)

    sections: list[str] = []
    toc: list[tuple[str, str]] = []

    def add_section(anchor: str, title: str, inner_html: str) -> None:
        toc.append((anchor, title))
        sections.append(f"<div class='section' id='{anchor}'><h2>{html_escape(title)}</h2>{inner_html}</div>")

    subtitle = f"{data_path.name} • {n:,} lignes • {df.shape[1]} colonnes"

    # -------------------------
    # Dataset summary
    # -------------------------
    dup = int(df["tconst"].duplicated().sum()) if "tconst" in df.columns else 0
    dup_badge = "ok" if dup == 0 else "bad"

    dataset_html = f"""
    <div class='note'>
      <b>Path:</b> <code>{html_escape(str(data_path))}</code><br/>
      <b>Shape:</b> {df.shape[0]:,} lignes × {df.shape[1]} colonnes
    </div>
    <div style='margin-top:10px;'>
      <span class='badge {dup_badge}'>tconst duplicates: {dup}</span>
      <span class='badge warn'>token flags: common&gt;{args.too_common_df:.2f}, rare≤{args.too_rare_df}</span>
      <span class='badge ok'>good DF zone: {args.good_df_min:.2f}–{args.good_df_max:.2f}</span>
    </div>
    """

    # nulls table
    nulls = df.isna().sum().sort_values(ascending=False)
    nulls_tbl = nulls.reset_index()
    nulls_tbl.columns = ["column", "null_count"]
    nulls_tbl["null_ratio"] = (nulls_tbl["null_count"] / max(n, 1)).round(5)
    nulls_tbl.to_csv(tables / "null_counts.csv", index=False)

    dataset_html += "<h3>Missing values</h3>"
    dataset_html += "<div class='note'>CSV: <code>tables/null_counts.csv</code></div>"
    dataset_html += df_to_html_table(nulls_tbl.head(25), max_rows=25)

    add_section("dataset", "Dataset", dataset_html)

    # -------------------------
    # Distributions
    # -------------------------
    dist_imgs = []
    for col, bins, logy in [
        ("averageRating", 50, False),
        ("numVotes", 60, True),
        ("runtimeMinutes", 60, False),
        ("startYear", 80, False),
    ]:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()

        if col == "numVotes":
            x2 = np.log1p(x)
            out = figs / "votes_log1p_hist.png"
            plot_hist(x2, out, "log1p(numVotes) distribution", bins=bins, logy=logy)
            dist_imgs.append("figures/votes_log1p_hist.png")
        else:
            out = figs / f"{col}_hist.png"
            plot_hist(x, out, f"{col} distribution", bins=bins, logy=logy)
            dist_imgs.append(f"figures/{col}_hist.png")

    dist_html = ""
    if dist_imgs:
        # show in 2-col grid
        chunks = []
        for i in range(0, len(dist_imgs), 2):
            a = dist_imgs[i]
            b = dist_imgs[i + 1] if i + 1 < len(dist_imgs) else None
            if b:
                chunks.append(f"<div class='imggrid'><img src='{a}'/><img src='{b}'/></div>")
            else:
                chunks.append(f"<div class='imggrid'><img src='{a}'/></div>")
        dist_html = "\n".join(chunks)
    else:
        dist_html = "<div class='note'>(no numeric columns found)</div>"

    add_section("distributions", "Distributions", dist_html)

    # -------------------------
    # Genres
    # -------------------------
    if "genres" in df.columns:
        gl = df["genres"].fillna("").astype(str).str.replace(",", " ", regex=False).str.split()
        genre_cnt = Counter(g for row in gl for g in row if g)

        gdf = pd.DataFrame(genre_cnt.most_common(), columns=["genre", "count"])
        gdf["ratio"] = (gdf["count"] / max(n, 1)).round(2)
        gdf["ratio (%)"] = (gdf["count"] / max(n, 1) * 100).round(2)
        gdf.to_csv(tables / "genres_counts.csv", index=False)

        plot_bar([(k, float(v)) for k, v in genre_cnt.most_common(20)], figs / "genres_top20.png", "Top genres", topn=20)

        genre_html = f"""
        <div class='note'>CSV: <code>tables/genres_counts.csv</code></div>
        <div class='imggrid'><img src='figures/genres_top20.png'/><div>
          <h3>Top genres (table)</h3>
          {df_to_html_table(gdf.head(20), max_rows=20)}
        </div></div>
        """
        add_section("genres", "Genres", genre_html)

    # -------------------------
    # Token columns audit
    # -------------------------
    token_cols = [c for c in ["directors", "writers", "actors", "genre_tokens"] if c in df.columns]

    if not token_cols:
        add_section("tokens", "Token columns", "<div class='note'>(No token columns found)</div>")
    else:
        # wrapper section intro
        add_section("tokens", "Token columns", "<div class='note'>Analyse DF/IDF/entropy + flags + prefix stats.</div>")

        for col in token_cols:
            toks = tokens_from_series(df[col])
            lens = toks.apply(len).to_numpy()

            # figures
            plot_hist(lens, figs / f"{col}_tokens_per_row.png", f"{col}: tokens per row", bins=60, logy=False)

            # DF + metrics
            base_tbl = df_table(toks)
            met = add_token_metrics(base_tbl, n)
            met = met.sort_values("df", ascending=False).reset_index(drop=True)

            met.to_csv(tables / f"{col}_token_metrics.csv", index=False)

            # flags (FIXED)
            too_common_all = met[met["df_ratio"] > args.too_common_df].head(10)
            too_rare_all = met[met["df"] <= args.too_rare_df].head(10)

            # "good zone" best tokens
            good_zone = met[(met["df_ratio"] >= args.good_df_min) & (met["df_ratio"] <= args.good_df_max)]
            best = good_zone.sort_values("token_value", ascending=False).head(30)

            # prefix summary
            pref = met.groupby("prefix").agg(
                n_tokens=("token", "count"),
                avg_df_ratio=("df_ratio", "mean"),
                avg_idf=("idf", "mean"),
                avg_value=("token_value", "mean"),
            ).sort_values("n_tokens", ascending=False)

            pref.to_csv(tables / f"{col}_prefix_summary.csv")

            # plot prefix token counts
            pref_items = [(idx, float(v)) for idx, v in pref["n_tokens"].to_dict().items()]
            pref_items = sorted(pref_items, key=lambda x: x[1], reverse=True)
            plot_bar(pref_items, figs / f"{col}_prefix_counts.png", f"{col}: prefix token counts", topn=30)

            # section html
            badge_common = "warn" if len(too_common_all) else "ok"
            badge_rare = "bad" if len(too_rare_all) else "ok"

            col_anchor = f"tokens_{col}"
            pills = kpi_pills([
                ("tokens (unique)", f"{len(met):,}"),
                ("too common", f"{len(too_common_all):,}"),
                ("too rare", f"{len(too_rare_all):,}"),
                ("good-zone tokens", f"{len(good_zone):,}"),
            ])

            col_html = f"""
            <div class='note'>
              CSV: <code>tables/{html_escape(col)}_token_metrics.csv</code> •
              Prefix CSV: <code>tables/{html_escape(col)}_prefix_summary.csv</code>
            </div>

            <div style='margin-top:10px;'>
              <span class='badge {badge_common}'>Too common: df_ratio &gt; {args.too_common_df:.2f} → {len(too_common_all):,}</span>
              <span class='badge {badge_rare}'>Too rare: df ≤ {args.too_rare_df} → {len(too_rare_all):,}</span>
              <span class='badge ok'>Good DF zone: {args.good_df_min:.2f}–{args.good_df_max:.2f}</span>
            </div>

            {pills}

            <div class='imggrid' style='margin-top:12px;'>
              <img src='figures/{html_escape(col)}_tokens_per_row.png'/>
              <img src='figures/{html_escape(col)}_prefix_counts.png'/>
            </div>

            <h3>Top tokens (DF)</h3>
            {df_to_html_table(met[['token','df','df_ratio']].head(args.topn), max_rows=args.topn)}

            <h3>Best tokens (good DF zone, by token_value)</h3>
            {df_to_html_table(best[['token','df','df_ratio','idf','H_bin','token_value']], max_rows=30)}

            <h3>Prefix summary</h3>
            {df_to_html_table(pref.reset_index().head(20), max_rows=20)}
            """

            # extra for genre_tokens
            if col == "genre_tokens" and not met.empty:
                dvals = met["df"].to_numpy(dtype=float)
                plot_hist(np.log1p(dvals), figs / "genre_tokens_df_log_hist.png", "genre_tokens: log1p(DF) histogram", bins=60, logy=False)

                # best tokens by prefix (top 8 prefixes)
                top_prefixes = pref.head(8).index.tolist()
                lines = []
                for pr in top_prefixes:
                    sub = met[
                        (met["prefix"] == pr)
                        & (met["df_ratio"] >= args.good_df_min)
                        & (met["df_ratio"] <= args.good_df_max)
                    ].sort_values("token_value", ascending=False).head(8)
                    if not sub.empty:
                        lines.append({"prefix": pr, "examples": ", ".join(sub["token"].tolist())})

                bypref_df = pd.DataFrame(lines)
                if not bypref_df.empty:
                    col_html += f"""
                    <h3>genre_tokens: DF histogram</h3>
                    <div class='imggrid'><img src='figures/genre_tokens_df_log_hist.png'/></div>

                    <h3>Best tokens by prefix (top prefixes)</h3>
                    {df_to_html_table(bypref_df, max_rows=20)}
                    """

            add_section(col_anchor, f"Tokens: {col}", col_html)

    # -------------------------
    # Write report
    # -------------------------
    report = out_dir / "report.html"
    write_html(report, sections, toc, subtitle=subtitle)

    print("Audit complete.")
    print("Report:", report)
    print("Figures:", figs)
    print("Tables:", tables)


if __name__ == "__main__":
    main()