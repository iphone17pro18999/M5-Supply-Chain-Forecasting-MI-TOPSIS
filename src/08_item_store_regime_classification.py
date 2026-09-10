"""Step 08 — construct the item-store subset and classify Smooth/Intermittent/Erratic/Lumpy regimes."""
import numpy as np
import pandas as pd
from common import RAW, PROCESSED, OUT, SEED, ensure_dirs, save_json

DEPTS = ["FOODS_3","HOUSEHOLD_1","HOBBIES_1"]
STORES = ["CA_1","TX_1","WI_1"]

def regime_stats(x):
    x=np.asarray(x,float)
    nz=np.flatnonzero(x>0)
    if len(nz)<2:
        adi=float(len(x))
    else:
        adi=float(np.diff(nz).mean())
    positive=x[x>0]
    cv2=float((positive.std(ddof=0)/positive.mean())**2) if len(positive) and positive.mean()!=0 else float("inf")
    if adi < 1.32 and cv2 < 0.49: reg="Smooth"
    elif adi >= 1.32 and cv2 < 0.49: reg="Intermittent"
    elif adi < 1.32 and cv2 >= 0.49: reg="Erratic"
    else: reg="Lumpy"
    return pd.Series({"ADI":adi,"CV2":cv2,"demand_regime":reg})

def main():
    ensure_dirs()
    sales=pd.read_csv(RAW/"sales_train_validation.csv")
    cal=pd.read_csv(RAW/"calendar.csv")
    day_cols=[c for c in sales.columns if c.startswith("d_")]
    sub=sales[sales["dept_id"].isin(DEPTS) & sales["store_id"].isin(STORES)].copy()
    stats=[]
    for _,r in sub.iterrows():
        s=regime_stats(r[day_cols].to_numpy())
        stats.append({"id":r["id"],"item_id":r["item_id"],"dept_id":r["dept_id"],"cat_id":r["cat_id"],
                      "store_id":r["store_id"],"state_id":r["state_id"],**s.to_dict()})
    stats=pd.DataFrame(stats)
    stats.to_csv(OUT/"item_store_regime_all_series.csv",index=False)

    rng=np.random.default_rng(SEED)
    chosen=[]
    for reg,g in stats.groupby("demand_regime"):
        if len(g)<30:
            raise ValueError(f"Regime {reg} has fewer than 30 series.")
        idx=rng.choice(g.index.to_numpy(),30,replace=False)
        chosen.append(g.loc[idx])
    chosen=pd.concat(chosen,ignore_index=True).sort_values(["demand_regime","id"])
    chosen.to_csv(OUT/"item_store_selected_series.csv",index=False)

    ids=set(chosen["id"])
    selected=sub[sub["id"].isin(ids)].copy()
    long=selected[["id","item_id","dept_id","cat_id","store_id","state_id"]+day_cols].melt(
        id_vars=["id","item_id","dept_id","cat_id","store_id","state_id"],
        value_vars=day_cols,var_name="d",value_name="demand"
    )
    cal_cols=["d","date","wm_yr_wk","weekday","wday","month","year","event_name_1","event_type_1",
              "event_name_2","event_type_2","snap_CA","snap_TX","snap_WI"]
    cal_cols=[c for c in cal_cols if c in cal.columns]
    long=long.merge(cal[cal_cols],on="d",how="left").merge(
        chosen[["id","ADI","CV2","demand_regime"]],on="id",how="left"
    )
    long["date"]=pd.to_datetime(long["date"])
    long.to_csv(PROCESSED/"item_store_selected_daily.csv",index=False)
    counts=stats["demand_regime"].value_counts().to_dict()
    save_json({"full_counts":counts,"selected_counts":chosen["demand_regime"].value_counts().to_dict()}, OUT/"item_store_regime_counts.json")
    print("Step 08 complete:", counts)

if __name__=="__main__":
    main()
