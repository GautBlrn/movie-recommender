![Python](https://img.shields.io/badge/Python-3.10+-blue)

# Movie Recommender (IMDb)

Content-based recommender built from IMDb datasets (genres + people + numeric signals) using KNN cosine similarity and a configurable reranking step.

## Features

- Dataset builder from IMDb TSV dumps (title.basics, title.ratings, title.principals, name.basics)

- Encodings:

    - Genres (multi-hot)

    - People (TF-IDF)

    - Numeric features (scaled) + Bayesian rating + confidence

    - Decade / rating bucket / runtime bucket (one-hot)

- Retrieval: NearestNeighbors (cosine)

- Rerank: score = weighted(similarity, rating, votes, year gap) via config.py

## What it does
- Builds a clean movie dataset from IMDb TSV dumps
- Feature engineering:
  - genres (multi-hot)
  - people (TF-IDF)
  - numeric (scaled) + bayesian rating + confidence
  - decade / rating bucket / runtime bucket (one-hot)
- NearestNeighbors (cosine) retrieval
- Rerank policy controlled in `config.py` (no retrain needed)

## Design choices
- Content-based (cold-start friendly)
- No deep learning (explainability, speed, reproducibility)
- Bayesian smoothing for noisy IMDb signals

## Project structure
- scripts/ 
  - audit_dataset.py
  - demo_reco.py
  - download_imdb.py
  - eval.py
  - make_dataset.py
  - train.py
  - predict.py
  - sweep_weights.py
- tests/
  - conftest.py
  - test_coherence.py
  - test_feature_alignment.py
  - test_genre_tokens.py
  - test_predict_json.py
- notebooks/
- data/
- config.py
- requirements.txt


## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Download IMDb dumps
python scripts/download_imdb.py --out data/raw/imdb

# 2) Build dataset (parquet)
python scripts/make_dataset.py \
  --imdb_dir data/raw/imdb \
  --out data/processed/movie_imdb.parquet \
  --min_votes 200 \
  --top_actors 3

# 3) Train model bundle
python scripts/train.py

# 4) Recommend by title
python scripts/predict.py --title "Inception" --k 10
```

## Tuning
### Requires retrain (training settings)

    MIN_VOTES, TF-IDF params, SVD size, Bayesian params, n_neighbors

### No retrain (predict/rerank settings)

    W_SIM, W_RATING, W_VOTES, W_YEAR, pool sizing, debug printing

## Next step

Add synopsis/plot text as an additional TF-IDF block to improve semantic relevance.

### Author

Gautier Blairon
