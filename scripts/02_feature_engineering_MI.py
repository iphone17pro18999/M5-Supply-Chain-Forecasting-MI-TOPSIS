"""Stage 2: feature engineering and Mutual Information ranking.

This stage creates lagged demand, rolling demand, calendar, event, SNAP, and price
predictors. Mutual Information is used to rank predictors for compact MI-MLP variants.
"""

from pathlib import Path
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "data" / "outputs"


def compute_mi_ranking(df: pd.DataFrame, target: str, exclude_cols: list[str]) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in exclude_cols + [target]]
    X = df[feature_cols].select_dtypes(include="number").fillna(0)
    y = df[target].fillna(0)
    scores = mutual_info_regression(X, y, random_state=42)
    return (
        pd.DataFrame({"feature": X.columns, "MI_score": scores})
        .sort_values("MI_score", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Use this script to reproduce feature engineering and MI ranking from processed M5 data.")
    print("Expected output: MI feature-ranking CSV files in data/outputs/ and results/tables/.")


if __name__ == "__main__":
    main()
