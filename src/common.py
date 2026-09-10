from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "generated"
REF = ROOT / "outputs" / "reference"
FIG = ROOT / "figures" / "generated"
SEED = 42
EVAL_BLOCK_DAYS = 28

def ensure_dirs():
    for p in (RAW, PROCESSED, OUT, REF, FIG):
        p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def smape(y_true, y_pred):
    y_true=np.asarray(y_true,float); y_pred=np.asarray(y_pred,float)
    den=(np.abs(y_true)+np.abs(y_pred))/2.0
    mask=den>0
    return 0.0 if not np.any(mask) else float(100*np.mean(np.abs(y_true[mask]-y_pred[mask])/den[mask]))

def metrics(y_true,y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true,y_pred))),
        "MAE": float(mean_absolute_error(y_true,y_pred)),
        "sMAPE": smape(y_true,y_pred),
        "R2": float(r2_score(y_true,y_pred))
    }

def encode_known_calendar(df):
    out=df.copy()
    out["date"]=pd.to_datetime(out["date"])
    out["dayofweek"]=out["date"].dt.dayofweek.astype(int)
    out["dayofmonth"]=out["date"].dt.day.astype(int)
    out["weekofyear"]=out["date"].dt.isocalendar().week.astype(int)
    out["quarter"]=out["date"].dt.quarter.astype(int)
    out["is_weekend"]=out["dayofweek"].isin([5,6]).astype(int)
    out["month_sin"]=np.sin(2*np.pi*out["month"]/12)
    out["month_cos"]=np.cos(2*np.pi*out["month"]/12)
    out["dow_sin"]=np.sin(2*np.pi*out["dayofweek"]/7)
    out["dow_cos"]=np.cos(2*np.pi*out["dayofweek"]/7)
    out["has_event_1"]=out.get("event_name_1",pd.Series(index=out.index,dtype=object)).fillna("NoEvent").ne("NoEvent").astype(int)
    out["has_event_2"]=out.get("event_name_2",pd.Series(index=out.index,dtype=object)).fillna("NoEvent").ne("NoEvent").astype(int)
    out["has_any_event"]=((out["has_event_1"]+out["has_event_2"])>0).astype(int)
    snap=np.zeros(len(out),dtype=int)
    if "state_id" in out.columns:
        for state in ("CA","TX","WI"):
            col=f"snap_{state}"
            if col in out.columns:
                snap=np.where(out["state_id"].eq(state),out[col].fillna(0).astype(int),snap)
    out["snap_active"]=snap
    return out

def add_shifted_demand_features(df, group_cols, target_col="demand"):
    out=df.sort_values(list(group_cols)+["date"]).copy()
    g=out.groupby(list(group_cols),sort=False)[target_col]
    for lag in (1,7,14,28):
        out[f"lag_{lag}"]=g.shift(lag)
    shifted=g.shift(1)
    keys=[out[c] for c in group_cols]
    for win in (7,14,28):
        out[f"rolling_mean_{win}"]=shifted.groupby(keys).rolling(win).mean().reset_index(level=list(range(len(group_cols))),drop=True)
        out[f"rolling_std_{win}"]=shifted.groupby(keys).rolling(win).std(ddof=0).reset_index(level=list(range(len(group_cols))),drop=True)
    return out

def category_fold_windows(target_dates, n_folds=5, block_days=28):
    """Five expanding folds with a 28-day validation block and a 28-day test block."""
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(target_dates).dropna().unique())))
    initial=len(dates)-(n_folds+1)*block_days
    if initial <= 0:
        raise ValueError("Insufficient dates for category-store rolling-origin design.")
    out=[]
    for i in range(n_folds):
        k=initial+i*block_days
        tr=dates[:k]
        va=dates[k:k+block_days]
        te=dates[k+block_days:k+2*block_days]
        out.append((i+1,tr,va,te))
    return out

def item_fold_windows_from_reference():
    p=REF/"item_store_fold_definition.csv"
    return pd.read_csv(p,parse_dates=[
        "train_target_start","train_target_end","test_target_start","test_target_end"
    ])
