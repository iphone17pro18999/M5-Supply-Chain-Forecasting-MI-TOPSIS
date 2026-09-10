"""Step 08 — full regime screening plus exact 120-series experimental subset."""
import numpy as np,pandas as pd
from common import RAW,PROCESSED,OUT,REF,ensure_dirs,save_json

DEPTS=["FOODS_3","HOUSEHOLD_1","HOBBIES_1"]
STORES=["CA_1","TX_1","WI_1"]

def stats(x):
    x=np.asarray(x,float);nz=np.flatnonzero(x>0)
    adi=float(len(x)/len(nz)) if len(nz)>0 else float(len(x))
    pos=x[x>0]
    cv2=float((pos.std(ddof=1)/pos.mean())**2) if len(pos) and pos.mean()!=0 else float("inf")
    if adi<1.32 and cv2<0.49:r="Smooth"
    elif adi>=1.32 and cv2<0.49:r="Intermittent"
    elif adi<1.32 and cv2>=0.49:r="Erratic"
    else:r="Lumpy"
    return adi,cv2,r

def main():
    ensure_dirs()
    sales=pd.read_csv(RAW/"sales_train_validation.csv")
    cal=pd.read_csv(RAW/"calendar.csv")
    days=[c for c in sales.columns if c.startswith("d_")]
    sub=sales[sales.dept_id.isin(DEPTS)&sales.store_id.isin(STORES)].copy()
    rows=[]
    for _,r in sub.iterrows():
        adi,cv2,reg=stats(r[days].to_numpy())
        rows.append([r.id,r.item_id,r.dept_id,r.cat_id,r.store_id,r.state_id,adi,cv2,reg])
    full=pd.DataFrame(rows,columns=["id","item_id","dept_id","cat_id","store_id","state_id","ADI","CV2","demand_regime"])
    counts=full.demand_regime.value_counts().to_dict()
    expected={"Smooth":229,"Intermittent":3924,"Erratic":121,"Lumpy":1039}
    if len(full)!=5313 or any(counts.get(k)!=v for k,v in expected.items()):
        raise AssertionError(f"Regime screening does not reproduce reference counts: {counts}")
    full.to_csv(OUT/"item_store_regime_all_series.csv",index=False)

    selected_ref=pd.read_csv(REF/"selected_item_store_series.csv")
    if selected_ref.id.nunique()!=120:
        raise AssertionError("Reference selected-series list must contain 120 unique IDs.")
    chosen=full[full.id.isin(selected_ref.id)].copy()
    if chosen.id.nunique()!=120:
        missing=sorted(set(selected_ref.id)-set(chosen.id))
        raise AssertionError(f"Selected IDs not recovered from M5 source data: {missing[:10]}")
    chosen=chosen.merge(selected_ref[["id","demand_regime"]],on="id",suffixes=("_recomputed","_reference"))
    if not chosen.demand_regime_recomputed.eq(chosen.demand_regime_reference).all():
        raise AssertionError("Regime labels differ from selected-series reference.")
    chosen.to_csv(OUT/"item_store_selected_series.csv",index=False)

    ids=set(selected_ref.id)
    sel=sub[sub.id.isin(ids)]
    idcols=["id","item_id","dept_id","cat_id","store_id","state_id"]
    long=sel[idcols+days].melt(id_vars=idcols,value_vars=days,var_name="d",value_name="demand")
    cc=["d","date","wm_yr_wk","weekday","wday","month","year","event_name_1","event_type_1",
        "event_name_2","event_type_2","snap_CA","snap_TX","snap_WI"]
    long=long.merge(cal[[c for c in cc if c in cal.columns]],on="d",how="left")
    long=long.merge(selected_ref[["id","ADI","CV2","demand_regime"]],on="id",how="left")
    long["date"]=pd.to_datetime(long["date"])
    long["is_weekend"]=long.date.dt.dayofweek.isin([5,6]).astype(int)
    for c in ["event_name_1","event_type_1","event_name_2","event_type_2"]:
        if c in long:long[c]=long[c].fillna("NoEvent")
    long.to_csv(PROCESSED/"item_store_selected_daily.csv",index=False)
    save_json({"full_counts":counts,"selected_counts":selected_ref.demand_regime.value_counts().to_dict()},
              OUT/"item_store_regime_counts.json")
    print("Step 08 complete.")

if __name__=="__main__":
    main()
