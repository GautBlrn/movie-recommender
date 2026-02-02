import json
import subprocess
import sys

def test_predicts_json_output_is_valid():
    cmd = [
        sys.executable, "scripts/predicts.py",
        "--title", "Alien",
        "--k", "3",
        "--json",
    ]
    out = subprocess.check_output(cmd, text=True)

    # Si tu as corrigé le print parasite: out doit être du JSON pur
    data = json.loads(out)

    assert "query" in data
    assert "recs" in data
    assert isinstance(data["recs"], list)
    assert len(data["recs"]) <= 3
