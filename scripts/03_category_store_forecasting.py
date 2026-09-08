"""Stage 3: category-store multi-horizon forecasting.

Models: Ridge, Random Forest, Boosting, MLP, MI-MLP Top-10/15/20.
Validation: five-fold rolling-origin chronological validation.
Forecast horizons: 1, 7, 14, and 28 days.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "tables"


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    print("Run category-store forecasting experiments and save performance summaries.")
    print("Expected output: Table_07_model_performance_summary.csv")


if __name__ == "__main__":
    main()
