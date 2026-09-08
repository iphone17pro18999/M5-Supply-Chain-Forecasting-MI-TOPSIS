# Variable Dictionary

| Variable group | Variables | Description |
|---|---|---|
| Demand lags | `lag_1`, `lag_7`, `lag_14`, `lag_28` | Previous demand values used to capture short-, weekly-, biweekly-, and monthly-memory effects. |
| Rolling demand means | `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28` | Smoothed demand level over trailing windows. |
| Rolling demand standard deviations | `rolling_std_7`, `rolling_std_14`, `rolling_std_28` | Local demand volatility over trailing windows. |
| Calendar variables | day, week, month, quarter, weekend indicators | Time-based explanatory predictors derived from the M5 calendar file. |
| Event variables | event indicators | Calendar event predictors derived from the M5 calendar file. |
| SNAP variables | state-specific SNAP indicators | SNAP assistance indicators mapped to corresponding store states. |
| Price variables | current price, lagged price, price change, rolling price statistics | Selling-price predictors aligned by item, store, and calendar week. |
| Regime variables | ADI, CV², demand regime | Used only for item-store classification and reporting, not as forecasting predictors. |
