"""Step 04 — reproduce category-store lag/rolling/calendar/event/SNAP/price features."""
import pandas as pd
from common import RAW,PROCESSED,OUT,ensure_dirs,save_json,encode_known_calendar,add_shifted_demand_features

def main():
    ensure_dirs()
    df=pd.read_csv(PROCESSED/"category_store_daily.csv",parse_dates=["date"])
    sales_map=pd.read_csv(RAW/"sales_train_validation.csv",usecols=["item_id","cat_id","store_id"]).drop_duplicates()
    prices=pd.read_csv(RAW/"sell_prices.csv").merge(sales_map,on=["item_id","store_id"],how="left")
    pa=prices.groupby(["cat_id","store_id","wm_yr_wk"])["sell_price"].agg(
        avg_sell_price="mean",median_sell_price="median",min_sell_price="min",
        max_sell_price="max",std_sell_price="std",n_priced_items="count"
    ).reset_index()
    df=df.merge(pa,on=["cat_id","store_id","wm_yr_wk"],how="left")
    df=add_shifted_demand_features(df,["cat_id","store_id"])
    df=encode_known_calendar(df)

    # Price handling follows the documented study protocol: alignment by category-store-week,
    # then within-series forward/backward filling; rolling price descriptors are shifted.
    pbase=["avg_sell_price","median_sell_price","min_sell_price","max_sell_price","std_sell_price","n_priced_items"]
    df[pbase]=df.groupby(["cat_id","store_id"])[pbase].ffill().bfill()
    g=df.groupby(["cat_id","store_id"],sort=False)["avg_sell_price"]
    df["avg_price_lag_7"]=g.shift(7)
    df["avg_price_change_7"]=df["avg_sell_price"]-df["avg_price_lag_7"]
    sp=g.shift(1)
    keys=[df["cat_id"],df["store_id"]]
    df["avg_price_rolling_mean_28"]=sp.groupby(keys).rolling(28).mean().reset_index(level=[0,1],drop=True)
    df["avg_price_rolling_std_28"]=sp.groupby(keys).rolling(28).std(ddof=0).reset_index(level=[0,1],drop=True)

    required=["lag_1","lag_7","lag_14","lag_28","rolling_mean_7","rolling_std_7",
              "rolling_mean_14","rolling_std_14","rolling_mean_28","rolling_std_28",
              "avg_price_lag_7","avg_price_rolling_mean_28","avg_price_rolling_std_28"]
    df=df.dropna(subset=required).reset_index(drop=True)
    df.to_csv(PROCESSED/"category_store_features.csv",index=False)
    save_json({"rows":len(df),"columns":len(df.columns),"date_min":str(df.date.min().date()),
               "date_max":str(df.date.max().date())},OUT/"category_store_feature_summary.json")
    print("Step 04 complete:",df.shape)

if __name__=="__main__":
    main()
