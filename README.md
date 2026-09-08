# A Neural Network-Based Decision Framework for Supply Chain Demand Forecasting with Mutual Information-Guided Features and Entropy-TOPSIS Evaluation

## Overview

This repository contains the reproducibility package for the study **"A Neural Network-Based Decision Framework for Supply Chain Demand Forecasting with Mutual Information-Guided Features and Entropy-TOPSIS Evaluation."**

The study develops a retail supply-chain demand forecasting framework using the public M5/Walmart dataset. The framework integrates feature engineering, Mutual Information-guided feature ranking, benchmark and neural machine-learning models, rolling-origin validation, item-store demand-regime classification, and Entropy-TOPSIS model evaluation.

## Main Contributions

1. A two-layer forecasting design combining category-store horizon-wise forecasting and item-store regime-wise robustness testing.
2. Mutual Information-guided feature ranking for identifying compact neural forecasting inputs.
3. Multi-horizon validation across 1-, 7-, 14-, and 28-day forecast horizons.
4. Item-store robustness analysis across Smooth, Intermittent, Erratic, and Lumpy demand regimes.
5. Entropy-TOPSIS model evaluation using forecasting accuracy, explanatory performance, training time, prediction time, and feature count.

## Repository Structure

```text
M5-Supply-Chain-Forecasting-MI-TOPSIS/
├── README.md
├── LICENSE
├── requirements.txt
├── citation.cff
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
├── notebooks/
├── scripts/
├── results/
│   ├── tables/
│   ├── figures/
│   └── supplementary_tables/
├── manuscript/
└── docs/
```

## Data

The study uses the public **M5 Forecasting Accuracy** dataset. The raw dataset is not redistributed in this repository because of file-size and redistribution considerations.

Required raw files:

- `calendar.csv`
- `sales_train_validation.csv`
- `sell_prices.csv`

Place these files in:

```text
data/raw/
```

A raw-data guide is provided in `data/raw/README_raw_data.md`.

## Workflow

Run the Python scripts in this order:

```bash
python scripts/01_data_preparation.py
python scripts/02_feature_engineering_MI.py
python scripts/03_category_store_forecasting.py
python scripts/04_entropy_topsis_ranking.py
python scripts/05_item_store_regime_analysis.py
```

The same workflow is also represented as notebooks in the `notebooks/` folder.

## Main Outputs

Main tables are stored in:

```text
results/tables/
```

Figures are stored in:

```text
results/figures/
```

Supplementary tables are stored in:

```text
results/supplementary_tables/
```

## Software Requirements

Install required packages using:

```bash
pip install -r requirements.txt
```

## Citation

Please cite the associated manuscript when using this repository or adapting the workflow.

## License

This repository is released for academic and reproducibility purposes under the MIT License.
