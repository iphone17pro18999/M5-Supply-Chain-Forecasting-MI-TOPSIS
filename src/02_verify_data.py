"""Step 02 — verify presence and basic structure of the three M5 source files."""
import pandas as pd
from common import RAW, OUT, ensure_dirs, save_json

FILES = {
    "calendar": "calendar.csv",
    "sales": "sales_train_validation.csv",
    "prices": "sell_prices.csv",
}

def main():
    ensure_dirs()
    report = {}
    for key, name in FILES.items():
        path = RAW / name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download from "
                "https://www.kaggle.com/competitions/m5-forecasting-accuracy/data"
            )
        df = pd.read_csv(path, nrows=5)
        report[key] = {
            "file": name,
            "sample_columns": list(df.columns),
            "file_size_bytes": path.stat().st_size,
        }
    required_calendar = {"d","date","wm_yr_wk","wday","month","year"}
    required_sales = {"id","item_id","dept_id","cat_id","store_id","state_id"}
    required_prices = {"store_id","item_id","wm_yr_wk","sell_price"}
    samples = {
        "calendar": pd.read_csv(RAW/FILES["calendar"], nrows=2),
        "sales": pd.read_csv(RAW/FILES["sales"], nrows=2),
        "prices": pd.read_csv(RAW/FILES["prices"], nrows=2),
    }
    checks = {
        "calendar_required_columns": required_calendar.issubset(samples["calendar"].columns),
        "sales_required_columns": required_sales.issubset(samples["sales"].columns),
        "prices_required_columns": required_prices.issubset(samples["prices"].columns),
    }
    if not all(checks.values()):
        raise ValueError(f"M5 schema check failed: {checks}")
    report["schema_checks"] = checks
    save_json(report, OUT/"raw_data_validation.json")
    print("Step 02 complete: source files found and schema checks passed.")

if __name__ == "__main__":
    main()
