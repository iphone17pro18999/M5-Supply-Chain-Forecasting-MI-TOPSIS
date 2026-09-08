"""Stage 1: data preparation for M5 supply-chain demand forecasting.

Expected raw files in data/raw/:
- calendar.csv
- sales_train_validation.csv
- sell_prices.csv

This script provides the reproducibility entry point. It checks input files and creates
project folders. Dataset transformation details can be extended according to the
notebook workflow used in the manuscript analysis.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

REQUIRED_FILES = ["calendar.csv", "sales_train_validation.csv", "sell_prices.csv"]


def check_raw_files() -> None:
    missing = [name for name in REQUIRED_FILES if not (RAW_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw M5 files in data/raw/: " + ", ".join(missing)
        )


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    check_raw_files()
    calendar = pd.read_csv(RAW_DIR / "calendar.csv")
    sales = pd.read_csv(RAW_DIR / "sales_train_validation.csv")
    prices = pd.read_csv(RAW_DIR / "sell_prices.csv")
    print("Loaded raw M5 files:")
    print(f"calendar: {calendar.shape}")
    print(f"sales_train_validation: {sales.shape}")
    print(f"sell_prices: {prices.shape}")
    print("Extend this script with the full data-transformation workflow used in the study.")


if __name__ == "__main__":
    main()
