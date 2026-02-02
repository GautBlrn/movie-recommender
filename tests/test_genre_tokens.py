import pytest

# adapte le path
# ici : scripts/make_dataset.py
from scripts.make_dataset import build_genre_tokens

def test_build_genre_tokens_empty():
    assert build_genre_tokens("") == ""

def test_build_genre_tokens_singles_and_pairs_present():
    s = "Action Sci-Fi"
    out = build_genre_tokens(s)
    # ton ancienne version faisait g: + gpair:
    # si tu es sur la version riche, adapte les asserts (gpair:, gtriple:, mood:, etc.)
    assert "gpair:" in out or "g:" in out

def test_build_genre_tokens_filters_bad_genres():
    s = "Drama Reality-TV"
    out = build_genre_tokens(s)
    assert "Reality-TV" not in out
