# Reproducibility protocol

## Raw data
Use only the three public M5 source files documented in `DATA.md`.

## Fixed experimental metadata
The `outputs/reference/` directory contains compact, publication-neutral verification artifacts:
- exact category-store fold definitions;
- exact item-store fold definitions;
- exact 120 selected item-store series;
- feature-engineered column manifests;
- reference category-store metrics;
- reference item-store MI ranking;
- reference item-store regime metrics;
- reference overall Entropy-TOPSIS ranking.

These reference artifacts are not substitutes for computation. They are used by Step 15 to verify a fresh run.

## Category-store layer
- 30 series = 3 categories × 10 stores.
- h-day-ahead targets for h ∈ {1,7,14,28}.
- 5 expanding chronological folds.
- 28-day validation and 28-day test blocks per fold.
- Ridge, Random Forest, Boosting, MLP, MI-MLP Top-10/15/20.

## Item-store layer
- Full screening: 5,313 series.
- Regime counts: Smooth 229, Intermittent 3,924, Erratic 121, Lumpy 1,039.
- Reported experiment uses the exact fixed balanced subset of 120 series: 30/regime.
- 28-day-ahead target.
- Exact 3 target-date folds are stored in `outputs/reference/item_store_fold_definition.csv`.
- Ridge, Boosting, MLP, MI-MLP Top-10.

## Leakage and provenance controls
- Demand lags use prior observations.
- Demand rolling statistics are shifted.
- Price rolling statistics are shifted.
- ADI/CV² are classification/reporting variables only and are excluded from forecast design matrices.
- The price missing-value handling reproduces the procedure reported in the Supplementary Material: within-series forward/backward filling after alignment.
- Transformers and scalers are fitted to training data inside each model pipeline.
- Random state is 42 where supported.

## Numerical verification
Accuracy metrics are verified against reference outputs with tight numerical tolerance.
Training and prediction times are reported but excluded from equality checks because they are hardware dependent.
