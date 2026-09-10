# Reproducibility protocol

## Philosophy

Every numerical result and analytical plot should be traceable to an executable Python step. No manuscript or supplementary document is required to reproduce the analysis.

## Locked configuration

The model configuration is centralized in `src/01_config.py`. Settings were taken from the archived study Supplementary Material:

- seed = 42
- Ridge alpha = 1.0
- Random Forest: 80 trees, max depth 16, minimum leaf size 2
- HistGradientBoosting: learning rate 0.06, max leaf nodes 31, L2 regularization 0.1
- category-store max iterations = 180
- item-store max iterations = 160
- MLP hidden layers = (96, 48)
- ReLU activation
- Adam solver
- alpha = 0.0001
- learning rate = 0.001
- early stopping = True
- validation fraction = 0.1
- MI compact neural variants = Top-10, Top-15, Top-20 at category-store level and Top-10 at item-store level

## Validation

Category-store:
- 30 category-store series
- 5 expanding rolling-origin folds
- horizons 1, 7, 14, 28 days

Item-store:
- 3 departments × 3 stores screened
- ADI/CV² demand classification
- balanced sample of 30 series/regime
- 3 expanding rolling-origin folds
- 28-day horizon

## Leakage control

Demand lags use only prior observations.
Rolling demand statistics are shifted by one period before window calculation.
Rolling price statistics are shifted before aggregation.
MI is computed on historical training observations.
ADI and CV² are never included in the forecast design matrix.

## Runtime differences

Training and prediction times are hardware-dependent. The reproducibility audit therefore does not require runtime equality. Accuracy metrics are compared using a relative tolerance to accommodate package/platform numerical variation.

## Expected verification targets

The archived reference outputs are stored under `outputs/reference/`. They are used solely to verify that a fresh run reproduces the study's numerical pattern and principal conclusions.
