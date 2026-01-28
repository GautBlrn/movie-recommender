import gzip
import shutil
import requests
from pathlib import Path

BASE_URL = "https://datasets.imdbws.com/"

FILES = [
    "title.basics.tsv.gz",
    "title.ratings.tsv.gz",
    "title.akas.tsv.gz",
    "name.basics.tsv.gz",
    "title.principals.tsv.gz",
]

DATA_DIR = Path("data/raw/imdb")


def download_file(url: str, dest: Path):
    print(f"Downloading {url}")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


def unzip_gz(src: Path, dest: Path):
    print(f"Extracting {src.name}")
    with gzip.open(src, "rb") as f_in:
        with open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for file in FILES:
        gz_path = DATA_DIR / file
        tsv_path = DATA_DIR / file.replace(".gz", "")

        if not tsv_path.exists():
            download_file(BASE_URL + file, gz_path)
            unzip_gz(gz_path, tsv_path)
            gz_path.unlink()  # delete .gz after extraction
        else:
            print(f"✅ {tsv_path.name} already exists, skipping")

    print("\nIMDb dataset ready in data/raw/imdb/")


if __name__ == "__main__":
    main()
