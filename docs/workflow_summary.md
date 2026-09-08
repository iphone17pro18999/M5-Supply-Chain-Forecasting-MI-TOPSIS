# Workflow Summary

The reproducibility workflow follows five stages.

## Stage 1: Data preparation

Raw M5 files are reshaped, merged with calendar variables, and aligned with selling-price information.

## Stage 2: Feature engineering and Mutual Information ranking

Lagged demand, rolling demand, calendar, event, SNAP, and price features are constructed. Mutual Information is used to rank predictors and create compact feature subsets.

## Stage 3: Category-store forecasting

Category-store models are trained and evaluated across 1-, 7-, 14-, and 28-day forecast horizons using rolling-origin validation.

## Stage 4: Entropy-TOPSIS ranking

Forecasting models are ranked using RMSE, MAE, sMAPE, R², training time, prediction time, and feature count.

## Stage 5: Item-store demand-regime robustness

Selected item-store series are classified into Smooth, Intermittent, Erratic, and Lumpy regimes using ADI and CV². Forecasting robustness is evaluated at the 28-day horizon.
