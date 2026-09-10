# Variable dictionary

Core predictors include:

- `lag_1`, `lag_7`, `lag_14`, `lag_28`: prior demand observations
- `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28`: shifted demand moving averages
- `rolling_std_7`, `rolling_std_14`, `rolling_std_28`: shifted demand volatility
- `dayofweek`, `dayofmonth`, `weekofyear`, `quarter`, `is_weekend`: calendar descriptors
- `month_sin`, `month_cos`, `dow_sin`, `dow_cos`: cyclical calendar encodings
- `has_event_1`, `has_event_2`, `has_any_event`: event flags
- `snap_active`: state-matched SNAP indicator
- `sell_price` / `avg_sell_price`: current selling-price descriptor
- `price_lag_7`: seven-day lagged price
- `price_change_7`: current minus seven-day lagged price
- `price_rolling_mean_28`, `price_rolling_std_28`: shifted 28-day price descriptors
- `ADI`, `CV2`: demand-regime descriptors used for classification/reporting only

`ADI` and `CV2` are excluded from all forecasting design matrices.
