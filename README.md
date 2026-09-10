# A Neural Network-Based Decision Framework for Supply Chain Demand Forecasting with Mutual Information-Guided Features and Entropy-TOPSIS Evaluation

This repository provides the step-by-step Python reproducibility pipeline for the associated study using the public M5 Forecasting Accuracy dataset.

## Reproducibility objective

The repository is designed so that a reviewer can start from the original public M5 files and reproduce:

- category-store data construction;
- leakage-controlled lag, rolling, calendar, event, SNAP, and price features;
- Mutual Information feature rankings;
- 1-, 7-, 14-, and 28-day category-store rolling-origin forecasts;
- Ridge, Random Forest, HistGradientBoosting, full-feature MLP, and MI-guided compact MLP models;
- objective Entropy-TOPSIS model ranking;
- item-store ADI/CV² demand-regime classification;
- 28-day regime-wise item-store robustness experiments;
- manuscript and supplementary numerical tables;
- all analytical figures from Python code;
- a final automated reproducibility audit.

The manuscript and Supplementary Material are intentionally **not** distributed in this repository.

## Data source

The original data are distributed by the M5 Forecasting Accuracy competition on Kaggle:

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Required files:

- `calendar.csv`
- `sales_train_validation.csv`
- `sell_prices.csv`

Place these files in `data/raw/`.

The raw M5 files are not redistributed here because they are subject to the competition's data terms. See `DATA.md`.

## Environment

Recommended Python: **3.10 or 3.11**

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Step-by-step execution

Run from the repository root:

```bash
python src/01_config.py
python src/02_verify_data.py
python src/03_prepare_category_store.py
python src/04_category_store_features.py
python src/05_category_store_mi.py
python src/06_category_store_forecasting.py
python src/07_entropy_topsis.py
python src/08_item_store_regime_classification.py
python src/09_item_store_features_mi.py
python src/10_item_store_regime_forecasting.py
python src/11_generate_main_tables.py
python src/12_generate_supplementary_tables.py
python src/13_generate_main_figures.py
python src/14_generate_supplementary_figures.py
python src/15_reproducibility_audit.py
```

A convenience runner is also provided:

```bash
python run_all.py
```

## Model configuration

The repository implements the locked study configuration:

- Ridge: `alpha=1.0`
- Random Forest: `n_estimators=80`, `max_depth=16`, `min_samples_leaf=2`
- HistGradientBoosting: `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=0.1`
- MLP: hidden layers `(96, 48)`, ReLU, Adam, `alpha=0.0001`, `learning_rate_init=0.001`, early stopping with validation fraction `0.1`
- random seed: `42`
- category-store model iterations: 180
- item-store model iterations: 160
- category-store validation: 5 rolling-origin folds
- item-store validation: 3 rolling-origin folds
- forecast horizons: 1, 7, 14, and 28 days
- item-store robustness horizon: 28 days

Full settings are centralized in `src/01_config.py`.

## Reproducibility safeguards

- random seed fixed at 42;
- chronological rather than random validation;
- rolling variables shifted before aggregation;
- Mutual Information fitted on training data only;
- ADI and CV² used only for regime classification and excluded from forecast predictors;
- all model metrics are generated from predictions rather than entered manually;
- the final audit compares generated outputs with archived reference results using stated tolerances.

## Repository map

See:

- `DATA.md` — source-data access and ethical redistribution notes
- `docs/REPRODUCIBILITY.md` — detailed protocol
- `docs/OUTPUT_MAP.md` — manuscript/SM output-to-code map
- `docs/VARIABLE_DICTIONARY.md` — feature definitions
- `outputs/reference/` — compact archived result files used only for reproducibility checks
- `outputs/generated/` — generated outputs after running the pipeline
- `figures/generated/` — figures produced by Python scripts

## Citation

Use the repository citation metadata in `citation.cff`.
