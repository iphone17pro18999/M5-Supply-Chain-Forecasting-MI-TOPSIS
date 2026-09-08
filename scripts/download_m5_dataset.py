"""
Download the M5 Forecasting Accuracy dataset from Zenodo.

Files downloaded:
- calendar.csv
- sales_train_validation.csv
- sales_train_evaluation.csv
- sell_prices.csv
- sample_submission.csv

Usage:
    python download_m5_dataset.py

Output folder:
    ./M5_dataset
"""

from pathlib import Path
import urllib.request

BASE = "https://zenodo.org/records/10203108/files"
FILES = [
    "calendar.csv",
    "sales_train_validation.csv",
    "sales_train_evaluation.csv",
    "sell_prices.csv",
    "sample_submission.csv",
]

OUT_DIR = Path("M5_dataset")
OUT_DIR.mkdir(exist_ok=True)

for name in FILES:
    url = f"{BASE}/{name}?download=1"
    out_path = OUT_DIR / name
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Already exists: {out_path}")
        continue
    print(f"Downloading {name} ...")
    urllib.request.urlretrieve(url, out_path)
    print(f"Saved: {out_path} ({out_path.stat().st_size/1024/1024:.2f} MB)")

print("Done. Dataset is in:", OUT_DIR.resolve())
