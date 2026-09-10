"""Step 04 — leakage-controlled category-store feature engineering with aggregated price features."""
import numpy as np
import pandas as pd
from common import RAW, PROCESSED, OUT, encode_known_calendar, add_shifted_demand_features, ensure_dirs, save_json

def main():
    ensure_dirs()
    df = pd.read_csv(PROCESSED/"category_store_daily.csv", parse_dates=["date"])
    prices = pd.read_csv(RAW/"sell_prices.csv")
    sales = pd.read_csv(RAW/"sales_train_validation.csv", usecols=["item_id","cat_id","store_id"])
    price_map = prices.merge(sales.drop_duplicates(), on=["item_id","store_id"], how="left")
    price_agg = price_map.groupby(["cat_id","store_id","wm_yr_wk"])["sell_price"].agg(
        avg_sell_price="mean", median_sell_price="median",
        min_sell_price="min", max_sell_price="max", std_sell_price="std"
    ).reset_index()
    df = df.merge(price_agg, on=["cat_id","store_id","wm_yr_wk"], how="left")
    df = encode_known_calendar(df)
    df = add_shifted_demand_features(df, ["cat_id","store_id"])

    # Category-store price history: shifted by one day before rolling summaries.
    g = df.sort_values(["cat_id","store_id","date"]).groupby(["cat_id","store_id"], sort=False)["avg_sell_price"]
    df["price_lag_7"] = g.shift(7)
    df["price_change_7"] = df["avg_sell_price"] - df["price_lag_7"]
    shifted_price = g.shift(1)
    df["price_rolling_mean_28"] = shifted_price.groupby([df["cat_id"],df["store_id"]]).rolling(28).mean().reset_index(level=[0,1], drop=True)
    df["price_rolling_std_28"] = shifted_price.groupby([df["cat_id"],df["store_id"]]).rolling(28).std(ddof=0).reset_index(level=[0,1], drop=True)

    # Fill category-store price aggregates within series only.
    pcols = ["avg_sell_price","median_sell_price","min_sell_price","max_sell_price","std_sell_price",
             "price_lag_7","price_change_7","price_rolling_mean_28","price_rolling_std_28"]
    df[pcols] = df.groupby(["cat_id","store_id"])[pcols].ffill().bfill()
    required = ["lag_1","lag_7","lag_14","lag_28","rolling_mean_7","rolling_std_7",
                "rolling_mean_14","rolling_std_14","rolling_mean_28","rolling_std_28"]
    df = df.dropna(subset=required).reset_index(drop=True)
    df.to_csv(PROCESSED/"category_store_features.csv", index=False)
    save_json({"rows": len(df), "columns": len(df.columns)}, OUT/"category_store_feature_summary.json")
    print(f"Step 04 complete: {df.shape}")

if __name__ == "__main__":
    main()
