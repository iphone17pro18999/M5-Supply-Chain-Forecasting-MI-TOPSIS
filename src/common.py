from __future__ import annotations
from pathlib import Path
import json
import time
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

def ensure_dirs() -> None:
    for p in (RAW, PROCESSED, OUT, REF, FIG):
        p.mkdir(parents=True, exist_ok=True)

def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return float(100.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))

def metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "sMAPE": smape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }

def timed_fit(model, X, y):
    t0 = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - t0

def timed_predict(model, X):
    t0 = time.perf_counter()
    pred = model.predict(X)
    return pred, time.perf_counter() - t0

def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def encode_known_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["dayofweek"] = out["date"].dt.dayofweek.astype(int)
    out["dayofmonth"] = out["date"].dt.day.astype(int)
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    out["quarter"] = out["date"].dt.quarter.astype(int)
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    out["month_sin"] = np.sin(2*np.pi*out["month"]/12)
    out["month_cos"] = np.cos(2*np.pi*out["month"]/12)
    out["dow_sin"] = np.sin(2*np.pi*out["dayofweek"]/7)
    out["dow_cos"] = np.cos(2*np.pi*out["dayofweek"]/7)
    for c in ["event_name_1","event_name_2"]:
        if c in out:
            out[f"has_{c}"] = out[c].notna().astype(int)
    out["has_event_1"] = out.get("has_event_name_1", 0)
    out["has_event_2"] = out.get("has_event_name_2", 0)
    out["has_any_event"] = ((out["has_event_1"] + out["has_event_2"]) > 0).astype(int)
    if "state_id" in out.columns:
        snap = np.zeros(len(out), dtype=int)
        for s in ("CA","TX","WI"):
            col = f"snap_{s}"
            if col in out.columns:
                snap = np.where(out["state_id"].eq(s), out[col].fillna(0).astype(int), snap)
        out["snap_active"] = snap
    return out

def add_shifted_demand_features(df: pd.DataFrame, group_cols, target_col="demand") -> pd.DataFrame:
    out = df.sort_values(list(group_cols)+["date"]).copy()
    g = out.groupby(list(group_cols), sort=False)[target_col]
    for lag in (1,7,14,28):
        out[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(1)
    for win in (7,14,28):
        out[f"rolling_mean_{win}"] = (
            shifted.groupby([out[c] for c in group_cols]).rolling(win).mean()
            .reset_index(level=list(range(len(group_cols))), drop=True)
        )
        out[f"rolling_std_{win}"] = (
            shifted.groupby([out[c] for c in group_cols]).rolling(win).std(ddof=0)
            .reset_index(level=list(range(len(group_cols))), drop=True)
        )
    return out

def rolling_origin_date_windows(unique_dates, n_folds: int, horizon: int):
    """Expanding-window folds using the last n_folds non-overlapping horizon blocks."""
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(unique_dates).dropna().unique())))
    need = n_folds * horizon
    if len(dates) <= need:
        raise ValueError(f"Need more than {need} dates; found {len(dates)}")
    start = len(dates) - need
    windows = []
    for fold in range(n_folds):
        test_dates = dates[start + fold*horizon : start + (fold+1)*horizon]
        train_dates = dates[: start + fold*horizon]
        windows.append((fold+1, train_dates, test_dates))
    return windows
