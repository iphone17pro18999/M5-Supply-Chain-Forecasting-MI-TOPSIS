# Data access and redistribution

## Original dataset

This study uses the public **M5 Forecasting Accuracy** dataset hosted by Kaggle:

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Required source files:

1. `calendar.csv`
2. `sales_train_validation.csv`
3. `sell_prices.csv`

The source files provide calendar/date information, historical daily unit sales, and store-week selling prices.

## Redistribution policy

The original M5 source files are **not included** in this repository. Users should obtain them from the official competition source and comply with the applicable data terms.

Only compact derived analytical outputs required for verification are included in `outputs/reference/`. These contain feature rankings, fold definitions, model metrics, and validation summaries rather than the original source data.

## Source-file integrity

The source copies used for the study had the following dimensions and SHA-256 hashes. The hashes are provided as an optional integrity check; Step 02 uses dimensions and schema as the required validation because line-ending or serialization changes can alter a hash without changing the data values.

| File | Expected dimensions | SHA-256 |
|---|---:|---|
| `calendar.csv` | 1,969 × 14 | `d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5` |
| `sales_train_validation.csv` | 30,490 × 1,919 | `f368e66ed1dbecb48b2cc8fc589bf68b3deddbbb36bf5c88b4d6d0a09b9b6724` |
| `sell_prices.csv` | 6,841,121 × 4 | `9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef` |

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
