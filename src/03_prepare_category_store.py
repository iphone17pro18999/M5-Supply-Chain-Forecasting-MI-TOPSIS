"""Step 03 — category-store aggregation: 3 categories × 10 stores × 1,913 days."""
import numpy as np
import pandas as pd
from common import RAW,PROCESSED,OUT,ensure_dirs,save_json


def _regime_stats(values):
    x=np.asarray(values,dtype=float)
    positive=x[x>0]
    adi=float(len(x)/len(positive)) if len(positive)>0 else float(len(x))
    cv2=float((positive.std(ddof=1)/positive.mean())**2) if len(positive)>1 and positive.mean()!=0 else float("inf")
    if adi < 1.32 and cv2 < 0.49: regime="Smooth"
    elif adi >= 1.32 and cv2 < 0.49: regime="Intermittent"
    elif adi < 1.32 and cv2 >= 0.49: regime="Erratic"
    else: regime="Lumpy"
    return adi,cv2,regime


def main():
    ensure_dirs()
    sales=pd.read_csv(RAW/"sales_train_validation.csv")
    cal=pd.read_csv(RAW/"calendar.csv")
    day=[c for c in sales.columns if c.startswith("d_")]

    # Aggregate in wide form before reshaping. This is mathematically identical to
    # melting item-level data first, but avoids materializing ~58 million rows.
    wide=sales.groupby(["cat_id","store_id","state_id"],as_index=False)[day].sum()
    regime_rows=[]
    for _,r in wide.iterrows():
        adi,cv2,reg=_regime_stats(r[day].to_numpy())
        regime_rows.append({"cat_id":r.cat_id,"store_id":r.store_id,"ADI":adi,"CV2":cv2,"demand_regime":reg})
    regime=pd.DataFrame(regime_rows)

    agg=wide.melt(id_vars=["cat_id","store_id","state_id"],value_vars=day,
                  var_name="d",value_name="demand")
    cc=["d","date","wm_yr_wk","weekday","wday","month","year","event_name_1","event_type_1",
        "event_name_2","event_type_2","snap_CA","snap_TX","snap_WI"]
    out=agg.merge(cal[[c for c in cc if c in cal.columns]],on="d",how="left")
    out=out.merge(regime,on=["cat_id","store_id"],how="left")
    out["date"]=pd.to_datetime(out["date"])
    out["is_weekend"]=out["date"].dt.dayofweek.isin([5,6]).astype(int)
    for c in ["event_name_1","event_type_1","event_name_2","event_type_2"]:
        if c in out: out[c]=out[c].fillna("NoEvent")
    out=out.sort_values(["cat_id","store_id","date"]).reset_index(drop=True)
    assert len(out)==57390
    assert out["cat_id"].nunique()==3
    assert out["store_id"].nunique()==10
    assert out[["cat_id","store_id"]].drop_duplicates().shape[0]==30
    assert out["demand_regime"].eq("Smooth").all()
    out.to_csv(PROCESSED/"category_store_daily.csv",index=False)
    save_json({"rows":len(out),"columns":len(out.columns),"n_series":30,
               "regime_counts":regime.demand_regime.value_counts().to_dict(),
               "date_min":str(out.date.min().date()),"date_max":str(out.date.max().date())},
              OUT/"category_store_dataset_summary.json")
    print("Step 03 complete: 57,390 rows, 30 series, all Smooth.")

if __name__=="__main__":
    main()
