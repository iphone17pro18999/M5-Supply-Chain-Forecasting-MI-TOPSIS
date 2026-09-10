# Data access and redistribution

## Original dataset

This study uses the public **M5 Forecasting Accuracy** dataset hosted by Kaggle:

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Required source files:

1. `calendar.csv`
2. `sales_train_validation.csv`
3. `sell_prices.csv`

The Kaggle data page describes these as the calendar/date information, historical daily unit sales, and store/date selling-price files.

## Redistribution policy

The original M5 source files are **not included** in this repository. Users should obtain them directly from the official competition page and comply with the applicable competition/data terms.

Only compact **derived analytical outputs** required to verify the reported analysis are included in `outputs/reference/`. These files contain aggregated model metrics, feature rankings, and validation summaries rather than the original Walmart transaction-level source data.

## Local placement

After downloading, place the files at:

```text
data/raw/calendar.csv
data/raw/sales_train_validation.csv
data/raw/sell_prices.csv
```

Then run:

```bash
python src/02_verify_data.py
```
