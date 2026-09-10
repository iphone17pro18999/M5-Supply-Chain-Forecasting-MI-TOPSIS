"""Step 03 — category-store aggregation: 3 categories × 10 stores × 1,913 days."""
import pandas as pd
from common import RAW,PROCESSED,OUT,ensure_dirs,save_json

IDS=["id","item_id","dept_id","cat_id","store_id","state_id"]

def main():
    ensure_dirs()
    sales=pd.read_csv(RAW/"sales_train_validation.csv")
    cal=pd.read_csv(RAW/"calendar.csv")
    day=[c for c in sales.columns if c.startswith("d_")]
    long=sales[IDS+day].melt(id_vars=IDS,value_vars=day,var_name="d",value_name="demand")
    agg=long.groupby(["cat_id","store_id","state_id","d"],as_index=False)["demand"].sum()
    cc=["d","date","wm_yr_wk","weekday","wday","month","year","event_name_1","event_type_1",
        "event_name_2","event_type_2","snap_CA","snap_TX","snap_WI"]
    out=agg.merge(cal[[c for c in cc if c in cal.columns]],on="d",how="left")
    out["date"]=pd.to_datetime(out["date"])
    out["is_weekend"]=out["date"].dt.dayofweek.isin([5,6]).astype(int)
    for c in ["event_name_1","event_type_1","event_name_2","event_type_2"]:
        if c in out: out[c]=out[c].fillna("NoEvent")
    out=out.sort_values(["cat_id","store_id","date"]).reset_index(drop=True)
    assert len(out)==57390
    assert out["cat_id"].nunique()==3
    assert out["store_id"].nunique()==10
    assert out[["cat_id","store_id"]].drop_duplicates().shape[0]==30
    out.to_csv(PROCESSED/"category_store_daily.csv",index=False)
    save_json({"rows":len(out),"columns":len(out.columns),"n_series":30,
               "date_min":str(out.date.min().date()),"date_max":str(out.date.max().date())},
              OUT/"category_store_dataset_summary.json")
    print("Step 03 complete: category-store daily dataset verified.")

if __name__=="__main__":
    main()
