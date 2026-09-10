"""Step 03 — construct the 30 category-store daily series from M5 sales and calendar data."""
import pandas as pd
from common import RAW, PROCESSED, OUT, ensure_dirs, save_json

ID_COLS = ["id","item_id","dept_id","cat_id","store_id","state_id"]

def main():
    ensure_dirs()
    sales = pd.read_csv(RAW/"sales_train_validation.csv")
    calendar = pd.read_csv(RAW/"calendar.csv")
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales[ID_COLS + day_cols].melt(
        id_vars=ID_COLS, value_vars=day_cols, var_name="d", value_name="demand"
    )
    agg = (
        long.groupby(["cat_id","store_id","state_id","d"], as_index=False)["demand"]
        .sum()
    )
    cal_cols = [
        "d","date","wm_yr_wk","weekday","wday","month","year",
        "event_name_1","event_type_1","event_name_2","event_type_2",
        "snap_CA","snap_TX","snap_WI"
    ]
    cal_cols = [c for c in cal_cols if c in calendar.columns]
    out = agg.merge(calendar[cal_cols], on="d", how="left")
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["cat_id","store_id","date"])
    path = PROCESSED/"category_store_daily.csv"
    out.to_csv(path, index=False)
    summary = {
        "rows": len(out),
        "n_categories": int(out["cat_id"].nunique()),
        "n_stores": int(out["store_id"].nunique()),
        "n_series": int(out[["cat_id","store_id"]].drop_duplicates().shape[0]),
        "date_min": str(out["date"].min().date()),
        "date_max": str(out["date"].max().date()),
    }
    save_json(summary, OUT/"category_store_dataset_summary.json")
    print("Step 03 complete:", summary)

if __name__ == "__main__":
    main()
