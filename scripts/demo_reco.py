# scripts/reco.py
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import pandas as pd


def run_one(title: str, k: int) -> None:
    cmd = [sys.executable, "scripts/predicts.py", "--title", title, "--k", str(k), "--json"]
    proc = subprocess.run(cmd, text=True, capture_output=True)

    if proc.returncode != 0:
        print(f"\npredicts.py a échoué pour: {title}")
        if proc.stderr.strip():
            print(proc.stderr.strip())
        else:
            print(proc.stdout.strip())
        return

    raw = proc.stdout.strip()
    if not raw:
        print(f"\nSortie vide (pas de JSON) pour: {title}")
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"\nJSON invalide pour: {title}")
        print("---- stdout ----")
        print(proc.stdout)
        print("---- stderr ----")
        print(proc.stderr)
        return

    q = data["query"]
    recs = pd.DataFrame(data["recs"])

    if recs.empty:
        print(f"\n🎬 {q['title']} ({q['year']})")
        print("Aucune reco retournée.")
        return

    # colonnes attendues
    cols = ["title", "year", "score", "sim", "rating", "votes", "gap"]
    recs = recs[[c for c in cols if c in recs.columns]].copy()

    # format
    if "votes" in recs.columns:
        recs["votes"] = recs["votes"].fillna(0).astype(int)
    for c, fmt in [("score", "{:.4f}"), ("sim", "{:.4f}"), ("rating", "{:.2f}")]:
        if c in recs.columns:
            recs[c] = recs[c].astype(float).map(lambda x: fmt.format(x))

    print(f"\n🎬 {q['title']} ({q['year']})")
    print("-" * 90)
    print(recs.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--title", action="append", help="Répétable: --title 'Alien' --title 'Toy Story'")
    args = ap.parse_args()

    for t in args.title or []:
        run_one(t, args.k)


if __name__ == "__main__":
    main()