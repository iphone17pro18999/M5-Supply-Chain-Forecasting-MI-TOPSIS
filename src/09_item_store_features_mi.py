"""Step 09 — item-store feature engineering and 28-day Mutual Information ranking."""
import time
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from common import RAW, PROCESSED, OUT, SEED, ensure_dirs, encode_known_calendar, add_shifted_demand_features
from modeling import prepare_numeric_matrix

def main():
    ensure_dirs()
    df=pd.read_csv(PROCESSED/"item_store_selected_daily.csv",parse_dates=["date"])
    prices=pd.read_csv(RAW/"sell_prices.csv")
    df=df.merge(prices,on=["store_id","item_id","wm_yr_wk"],how="left")
    df["sell_price"]=df.groupby(["item_id","store_id"])["sell_price"].ffill().bfill()
    df=encode_known_calendar(df)
    df=add_shifted_demand_features(df,["item_id","store_id"])
    g=df.sort_values(["item_id","store_id","date"]).groupby(["item_id","store_id"],sort=False)["sell_price"]
    df["price_lag_7"]=g.shift(7)
    df["price_change_7"]=df["sell_price"]-df["price_lag_7"]
    sp=g.shift(1)
    df["price_rolling_mean_28"]=sp.groupby([df["item_id"],df["store_id"]]).rolling(28).mean().reset_index(level=[0,1],drop=True)
    df["price_rolling_std_28"]=sp.groupby([df["item_id"],df["store_id"]]).rolling(28).std(ddof=0).reset_index(level=[0,1],drop=True)
    needed=["lag_1","lag_7","lag_14","lag_28","rolling_mean_7","rolling_std_7",
            "rolling_mean_14","rolling_std_14","rolling_mean_28","rolling_std_28","sell_price"]
    df=df.dropna(subset=needed).reset_index(drop=True)
    df.to_csv(PROCESSED/"item_store_features.csv",index=False)

    tmp=df.sort_values(["item_id","store_id","date"]).copy()
    tmp["target"]=tmp.groupby(["item_id","store_id"])["demand"].shift(-28)
    tmp=tmp.dropna(subset=["target"])
    dates=sorted(tmp["date"].unique())
    cutoff=dates[-84]  # first of final three non-overlapping 28-day folds
    tr=tmp[tmp["date"]<cutoff].copy()
    X,_,cols=prepare_numeric_matrix(tr,tr.iloc[:0].copy(),"target")
    t0=time.perf_counter()
    mi=mutual_info_regression(X,tr["target"].to_numpy(),random_state=SEED)
    rt=time.perf_counter()-t0
    rank=pd.DataFrame({"horizon":28,"feature":cols,"MI_score":mi}).sort_values("MI_score",ascending=False).reset_index(drop=True)
    rank["MI_rank"]=rank.index+1; rank["MI_runtime_seconds"]=rt
    rank.to_csv(OUT/"item_store_mi_feature_ranking.csv",index=False)

    summary=df.groupby("demand_regime").agg(
        n_observations=("demand","size"),n_items=("item_id","nunique"),n_stores=("store_id","nunique"),
        mean_demand=("demand","mean"),std_demand=("demand","std"),
        zero_ratio=("demand",lambda s:float((s==0).mean())),
        mean_sell_price=("sell_price","mean")
    ).reset_index()
    summary.to_csv(OUT/"item_store_regime_summary.csv",index=False)
    print("Step 09 complete.")

if __name__=="__main__":
    main()
