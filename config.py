# Configuration for the movie recommender project
# Edit these values to tune the pipeline without touching the code.

from pathlib import Path

# ==========================================================
# TRAINING SETTINGS (CHANGING ANYTHING HERE REQUIRES RETRAIN)
# These values affect the dataset, the feature space, or the model itself.
# If you change any of them: re-run make_dataset.py + train.py
# ==========================================================

# -----------------------
# Paths
# -----------------------
DATA_PATH = Path("data/processed/movie_imdb.parquet")
MODEL_DIR = Path("data/models")
MODEL_BUNDLE_PATH = MODEL_DIR / "recommender.joblib"

# -----------------------
# Dataset / filtering (apply in make_dataset.py or downstream as needed)
# -----------------------
MIN_VOTES = 200
MIN_RATING = 0.0  # set >0 if you decide to filter low-rated movies

# -----------------------
# TF-IDF blocks (directors / writers / actors / genre_tokens)
# -----------------------
MIN_DF_TEXT = 2
MAX_FEATURES_TEXT = 50_000          # shared cap for directors/writers/actors
MAX_FEATURES_GENRE_TOKENS = 5_000   # small vocab by design

# Block weights (from notebook conclusions)
W_GENRES = 1.0
W_GENRE_TOKENS = 0.8
W_DIRECTORS = 2.0
W_WRITERS = 1.3
W_ACTORS = 0.6

# -----------------------
# Encoding: numeric features
# -----------------------
USE_NUMERIC = True
LOG1P_VOTES = True  # recommended (numVotes is very skewed)

# -----------------------
# Dimensionality reduction
# -----------------------
SVD_COMPONENTS = 100  # 0 disables SVD

# -----------------------
# Nearest neighbors
# -----------------------
N_NEIGHBORS = 50  # predict pool should be <= this ideally
RANDOM_STATE = 42

# -----------------------
# Bayesian rating / confidence
# -----------------------
RATING_PRIOR = 500
CONFIDENCE_K = 500

# -----------------------
# Temporal popularity smoothing (Bayesian votes/year)
# -----------------------
VOTES_PER_YEAR_BAYES_TAU = 3.0

# ==========================================================
# PREDICT / RERANK SETTINGS (NO RETRAIN NEEDED)
# These values only change the final ranking/printing in predicts.py.
# You can tweak them and re-run predicts.py directly.
# ==========================================================

# Candidate pool size for rerank
PRED_POOL_MIN = 50
PRED_POOL_MULT = 5  # pool = max(PRED_POOL_MIN, k * PRED_POOL_MULT)

# Weights for final score (roughly sum to 1.0; doesn't have to be exact)
# sim: cosine similarity (content)
# rating: rating_bayes (or averageRating fallback)
# votes: soft popularity prior (log-normalized)
# year: proximity in release year (penalize big gaps)
W_SIM = 0.59 # strong content similarity
W_RATING = 0.15 # reward quality
W_VOTES = -0.15 # penalize pure popularity
W_YEAR = 0.03 # moderate temporal coherence

# Votes normalization for votes component
# votes_norm = log1p(votes) / log1p(VOTES_NORM_MAX)
VOTES_NORM_MAX = 2_500_000

# Year proximity scaling
# year_score = max(0, 1 - abs(year - query_year) / YEAR_GAP_MAX)
YEAR_GAP_MAX = 30

# Debug printing: show score components for top results
PRED_DEBUG = True
PRED_DEBUG_TOPN = 10  # print components for top N