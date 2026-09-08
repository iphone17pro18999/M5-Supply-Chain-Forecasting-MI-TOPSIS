"""Stage 5: item-store demand-regime robustness analysis.

Demand regimes are classified using ADI and CV²:
- Smooth: ADI < 1.32 and CV² < 0.49
- Intermittent: ADI >= 1.32 and CV² < 0.49
- Erratic: ADI < 1.32 and CV² >= 0.49
- Lumpy: ADI >= 1.32 and CV² >= 0.49

ADI and CV² are used only for classification and reporting, not as forecasting predictors.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "tables"


def classify_demand_regime(series: pd.Series) -> str:
    positive = series[series > 0]
    if len(positive) == 0:
        return "Lumpy"
    demand_occurrences = (series > 0).astype(int)
    intervals = []
    last = None
    for idx, occurred in enumerate(demand_occurrences):
        if occurred:
            if last is not None:
                intervals.append(idx - last)
            last = idx
    adi = np.mean(intervals) if intervals else len(series)
    cv2 = (positive.std(ddof=0) / positive.mean()) ** 2 if positive.mean() != 0 else np.inf
    if adi < 1.32 and cv2 < 0.49:
        return "Smooth"
    if adi >= 1.32 and cv2 < 0.49:
        return "Intermittent"
    if adi < 1.32 and cv2 >= 0.49:
        return "Erratic"
    return "Lumpy"


def main() -> None:
    print("Run item-store regime classification and 28-day robustness forecasting.")
    print("Expected outputs: regime-wise performance and MI feature-ranking tables.")


if __name__ == "__main__":
    main()
