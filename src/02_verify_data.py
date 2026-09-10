"""Step 02 — verify M5 source files, dimensions and expected schema."""
import pandas as pd
from common import RAW,OUT,ensure_dirs,save_json

EXPECTED={
 "calendar.csv":(1969,14),
 "sales_train_validation.csv":(30490,1919),
 "sell_prices.csv":(6841121,4)
}

def main():
    ensure_dirs()
    report={}
    for name,shape in EXPECTED.items():
        p=RAW/name
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing. See DATA.md.")
        df=pd.read_csv(p)
        actual=df.shape
        report[name]={"expected_shape":shape,"actual_shape":actual,"match":actual==shape}
        if actual!=shape:
            raise ValueError(f"{name}: expected {shape}, found {actual}")
    sales=pd.read_csv(RAW/"sales_train_validation.csv",nrows=1)
    day_cols=[c for c in sales.columns if c.startswith("d_")]
    if len(day_cols)!=1913 or day_cols[0]!="d_1" or day_cols[-1]!="d_1913":
        raise ValueError("Unexpected M5 day-column structure.")
    save_json(report,OUT/"raw_data_validation.json")
    print("Step 02 complete: all source dimensions match.")

if __name__=="__main__":
    main()
