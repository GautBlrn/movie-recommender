# ==========================================================
# Configuration for the movie recommender project
# ==========================================================
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================
DATA_PATH = Path("data/processed/movie_imdb.parquet")
MODEL_DIR = Path("data/models")
MODEL_BUNDLE_PATH = MODEL_DIR / "recommender.joblib"

# ==========================================================
# DATASET FILTERING (make_dataset.py)
# ==========================================================
MIN_VOTES = 200
MIN_RATING = 0.0

# ==========================================================
# TF-IDF / TEXT FEATURE SETTINGS (train.py)
# ==========================================================
# Global defaults
MIN_DF_TEXT = 2
MAX_FEATURES_TEXT = 50_000

# Per-block min_df (controls noise)
MIN_DF_DIRECTORS = 2
MIN_DF_WRITERS = 2
MIN_DF_ACTORS = 5
MIN_DF_GENRE_TOKENS = 2

# Per-block max_features (controls RAM + overfitting)
MAX_FEATURES_DIRECTORS = 25_000
MAX_FEATURES_WRITERS = 35_000
MAX_FEATURES_ACTORS = 50_000
MAX_FEATURES_GENRE_TOKENS = 7_500

# Block weights (train-time: define NN feature space)
W_GENRES = 1.80
W_GENRE_TOKENS = 2.5
W_DIRECTORS = 0.80
W_WRITERS = 0.35
W_ACTORS = 0.40

# ==========================================================
# GENRE TOKEN ENGINEERING (make_dataset.py)
# ==========================================================
USE_GENRE_TRIPLES = True
USE_AMBIENCE_TOKENS = True

# DF filtering (avoid useless / too generic tokens)
GENRE_PAIR_MIN_DF = 10
GENRE_PAIR_MAX_DF_RATIO = 0.40

GENRE_TRIPLE_MIN_DF = 30
GENRE_TRIPLE_MIN_DF_FOCUS = 20  # exception for focus genres

# Weighting rules
GENRE_PAIR_DRAMA_PENALTY = 0.60
GENRE_PAIR_FOCUS_BONUS = 1.25

# Repetition scales / caps (how strongly tokens repeat in the text field)
GENRE_PAIR_SCALE = 2
GENRE_TRIPLE_SCALE = 2
GENRE_PAIR_MAX_REP = 4
GENRE_TRIPLE_MAX_REP = 3

FOCUS_GENRES = {"Horror", "Animation"}

# ==========================================================
# AMBIENCE / SUB-GENRE HEURISTICS (make_dataset.py)
# ==========================================================
PACE_FAST_RUNTIME_MAX = 105
PACE_SLOW_RUNTIME_MIN = 140

QUAL_HIGH_MIN = 7.5
QUAL_LOW_MAX = 5.0

# ==========================================================
# NUMERIC + CATEGORICAL FEATURES (train.py + predict.py)
# ==========================================================
USE_NUMERIC = True
LOG1P_VOTES = True

# Rating buckets
RATING_BUCKET_BINS = [0.0, 6.0, 7.5, 10.0]
RATING_BUCKET_LABELS = ["low", "mid", "high"]

# Runtime buckets
RUNTIME_BUCKET_BINS = [0, 70, 400]
RUNTIME_BUCKET_LABELS = ["short", "long"]

# ==========================================================
# MODELING (train.py)
# ==========================================================
SVD_COMPONENTS = 100
N_NEIGHBORS = 200
RANDOM_STATE = 42

# ==========================================================
# BAYESIAN SMOOTHING (train.py + predict.py)
# ==========================================================
RATING_PRIOR = 500
CONFIDENCE_K = 500
VOTES_PER_YEAR_BAYES_TAU = 3.0

# ==========================================================
# PREDICT / RERANK (predict.py only: no retrain)
# ==========================================================
# Candidate pool size used before rerank:
# pool = max(PRED_POOL_MIN, k * PRED_POOL_MULT)
PRED_POOL_MIN = 300
PRED_POOL_MULT = 10

# Final score weights
W_SIM = 0.64
W_RATING = 0.15
W_VOTES = -0.05
W_YEAR = 0.08

# Bonus genre overlap
W_GENRE_OVERLAP = 0.02

# Normalization / constraints for rerank
VOTES_NORM_MAX = 2_500_000
YEAR_GAP_MAX = 45

# Debug
PRED_DEBUG = False
PRED_DEBUG_TOPN = 10