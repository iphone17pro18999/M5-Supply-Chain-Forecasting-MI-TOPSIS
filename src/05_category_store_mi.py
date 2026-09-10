"""Step 05 — category-store Mutual Information ranking for each forecast horizon."""
import time
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from common import PROCESSED, OUT, SEED, ensure_dirs
from modeling import prepare_numeric_matrix

def main():
    ensure_dirs()
    df = pd.read_csv(PROCESSED/"category_store_features.csv", parse_dates=["date"])
    rows = []
    for h in [1,7,14,28]:
        tmp = df.sort_values(["cat_id","store_id","date"]).copy()
        tmp["target"] = tmp.groupby(["cat_id","store_id"])["demand"].shift(-h)
        tmp = tmp.dropna(subset=["target"]).copy()
        # MI is estimated using historical training observations only; use all but final 5*h dates.
        dates = sorted(tmp["date"].unique())
        cutoff = dates[-5*h] if len(dates) > 5*h else dates[-h]
        tr = tmp[tmp["date"] < cutoff]
        X, _, cols = prepare_numeric_matrix(tr, tr.iloc[:0].copy(), "target")
        y = tr["target"].to_numpy()
        t0 = time.perf_counter()
        mi = mutual_info_regression(X, y, random_state=SEED)
        runtime = time.perf_counter() - t0
        rank = pd.DataFrame({"horizon":h,"feature":cols,"MI_score":mi})
        rank = rank.sort_values("MI_score", ascending=False).reset_index(drop=True)
        rank["MI_rank"] = rank.index + 1
        rank["MI_runtime_seconds"] = runtime
        rows.append(rank)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT/"category_store_mi_feature_ranking.csv", index=False)
    print("Step 05 complete.")

if __name__ == "__main__":
    main()
