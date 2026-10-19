from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path("data")
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
DATA_FILE = DATA_DIR / "heart.csv"


def download_data() -> Path:
    """Download the UCI Cleveland Heart Disease dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        urlretrieve(DATA_URL, DATA_FILE)

    return DATA_FILE


if __name__ == "__main__":
    path = download_data()
    print(f"Dataset saved to: {path}")
