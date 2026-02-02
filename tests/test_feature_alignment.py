import subprocess
import sys

def test_reco_multiple_titles_runs():
    cmd = [
        sys.executable, "scripts/reco.py",
        "--k", "3",
        "--title", "Alien",
        "--title", "Toy Story",
    ]
    out = subprocess.check_output(cmd, text=True)

    # on vérifie juste que ça sort quelque chose de cohérent
    assert "Alien" in out
    assert "Toy Story" in out