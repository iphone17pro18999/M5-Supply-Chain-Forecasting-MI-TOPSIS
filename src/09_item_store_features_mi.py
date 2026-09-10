"""Step 09 — item-store features and 28-day MI ranking using the fixed experimental subset."""
import time,pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from common import RAW,PROCESSED,OUT,REF,SEED,ensure_dirs,encode_known_calendar,add_shifted_demand_features
from modeling import candidate_columns

def main():
    ensure_dirs()
    df=pd.read_csv(PROCESSED/"item_store_selected_daily.csv",parse_dates=["date"])
    prices=pd.read_csv(RAW/"sell_prices.csv")
    df=df.merge(prices,on=["store_id","item_id","wm_yr_wk"],how="left")
    # This reproduces the explicitly reported SM protocol.
    df["sell_price"]=df.groupby(["item_id","store_id"])["sell_price"].ffill().bfill()
    df=add_shifted_demand_features(df,["item_id","store_id"])
    df=encode_known_calendar(df)
    g=df.groupby(["item_id","store_id"],sort=False)["sell_price"]
    df["price_lag_7"]=g.shift(7)
    df["price_change_7"]=df["sell_price"]-df["price_lag_7"]
    sp=g.shift(1);keys=[df.item_id,df.store_id]
    df["price_rolling_mean_28"]=sp.groupby(keys).rolling(28).mean().reset_index(level=[0,1],drop=True)
    df["price_rolling_std_28"]=sp.groupby(keys).rolling(28).std(ddof=0).reset_index(level=[0,1],drop=True)
    req=["lag_1","lag_7","lag_14","lag_28","rolling_mean_7","rolling_std_7",
         "rolling_mean_14","rolling_std_14","rolling_mean_28","rolling_std_28",
         "price_lag_7","price_rolling_mean_28","price_rolling_std_28"]
    df=df.dropna(subset=req).reset_index(drop=True)
    if len(df)!=226200:
        raise AssertionError(f"Expected 226,200 feature-engineered rows, found {len(df)}")
    df.to_csv(PROCESSED/"item_store_features.csv",index=False)

    tmp=df.sort_values(["item_id","store_id","date"]).copy()
    tmp["target"]=tmp.groupby(["item_id","store_id"])["demand"].shift(-28)
    tmp["target_date"]=tmp.date+pd.to_timedelta(28,unit="D")
    tmp=tmp.dropna(subset=["target"])
    fd=pd.read_csv(REF/"item_store_fold_definition.csv",parse_dates=["train_target_start","train_target_end"]).iloc[0]
    tr=tmp[(tmp.target_date>=fd.train_target_start)&(tmp.target_date<=fd.train_target_end)].copy()

    cols=candidate_columns(tr)
    cat=[c for c in cols if str(tr[c].dtype) in ("object","category")]
    num=[c for c in cols if c not in cat]
    prep=ColumnTransformer([
        ("num",SimpleImputer(strategy="median"),num),
        ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),
                         ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)
    ],verbose_feature_names_out=False)
    X=prep.fit_transform(tr[cols]);names=list(prep.get_feature_names_out())
    if X.shape[1]!=132:
        raise AssertionError(f"Expected 132 encoded item-store predictors, found {X.shape[1]}")
    t0=time.perf_counter();mi=mutual_info_regression(X,tr.target.to_numpy(),random_state=SEED,n_jobs=1);rt=time.perf_counter()-t0
    rank=pd.DataFrame({"horizon":28,"feature":names,"MI_score":mi}).sort_values("MI_score",ascending=False).reset_index(drop=True)
    rank["MI_rank"]=rank.index+1;rank["MI_runtime_seconds"]=rt
    rank.to_csv(OUT/"item_store_mi_feature_ranking.csv",index=False)

    sm=df.groupby("demand_regime").agg(
        n_observations=("demand","size"),n_items=("item_id","nunique"),n_stores=("store_id","nunique"),
        mean_demand=("demand","mean"),std_demand=("demand","std"),
        zero_ratio=("demand",lambda s:float((s==0).mean())),mean_sell_price=("sell_price","mean")
    ).reset_index()
    sm.to_csv(OUT/"item_store_regime_summary.csv",index=False)
    print("Step 09 complete.")

if __name__=="__main__":
    main()
