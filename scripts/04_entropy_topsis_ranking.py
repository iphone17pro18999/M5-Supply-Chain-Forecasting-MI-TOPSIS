"""Stage 4: Entropy-TOPSIS model evaluation.

Criteria:
- Cost criteria: RMSE, MAE, sMAPE, training time, prediction time, feature count
- Benefit criterion: R²
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "tables"


def entropy_weights(matrix: np.ndarray) -> np.ndarray:
    eps = 1e-12
    col_sums = matrix.sum(axis=0) + eps
    p = matrix / col_sums
    k = 1.0 / np.log(matrix.shape[0])
    entropy = -k * np.sum(p * np.log(p + eps), axis=0)
    diversity = 1 - entropy
    return diversity / diversity.sum()


def topsis(decision_matrix: pd.DataFrame, criteria: list[str], benefit: list[str]) -> pd.Series:
    X = decision_matrix[criteria].astype(float).copy()
    # Convert cost criteria so that larger is better after inversion.
    for c in criteria:
        if c not in benefit:
            X[c] = 1 / (X[c] + 1e-12)
    norm = X / np.sqrt((X ** 2).sum(axis=0))
    w = entropy_weights(norm.values)
    weighted = norm.values * w
    ideal = weighted.max(axis=0)
    nadir = weighted.min(axis=0)
    d_pos = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - nadir) ** 2).sum(axis=1))
    return pd.Series(d_neg / (d_pos + d_neg), index=decision_matrix.index)


def main() -> None:
    print("Run Entropy-TOPSIS ranking using the model-performance summary table.")
    print("Expected output: Table_12_overall_topsis_ranking.csv")


if __name__ == "__main__":
    main()
