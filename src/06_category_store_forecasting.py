"""Step 06 — five-fold rolling-origin category-store forecasting for four horizons."""
import time
import numpy as np
import pandas as pd
from common import PROCESSED, OUT, SEED, ensure_dirs, metrics, rolling_origin_date_windows
from modeling import prepare_numeric_matrix, build_model

MODELS = ["Ridge_All","RandomForest_All","Boosting_All","MLP_All",
          "MI_MLP_Top10","MI_MLP_Top15","MI_MLP_Top20"]

def main():
    ensure_dirs()
    df = pd.read_csv(PROCESSED/"category_store_features.csv", parse_dates=["date"])
    mi = pd.read_csv(OUT/"category_store_mi_feature_ranking.csv")
    results, preds = [], []

    for h in [1,7,14,28]:
        tmp = df.sort_values(["cat_id","store_id","date"]).copy()
        tmp["target"] = tmp.groupby(["cat_id","store_id"])["demand"].shift(-h)
        tmp["target_date"] = tmp["date"] + pd.to_timedelta(h, unit="D")
        tmp = tmp.dropna(subset=["target"]).copy()

        windows = rolling_origin_date_windows(tmp["target_date"].unique(), 5, h)
        top = {k: mi[mi["horizon"].eq(h)].nsmallest(k, "MI_rank")["feature"].tolist()
               for k in (10,15,20)}

        for fold, train_dates, test_dates in windows:
            tr = tmp[tmp["target_date"].isin(train_dates)].copy()
            te = tmp[tmp["target_date"].isin(test_dates)].copy()
            Xtr_all, Xte_all, all_cols = prepare_numeric_matrix(tr, te, "target")
            col_lookup = set(all_cols)

            for model_name in MODELS:
                if model_name.startswith("MI_MLP_Top"):
                    k = int(model_name.split("Top")[-1])
                    selected = [c for c in top[k] if c in col_lookup]
                    # MI ranking may contain pre-one-hot base names; fall back to prefix-matched columns.
                    expanded = []
                    for c in top[k]:
                        expanded += [x for x in all_cols if x == c or x.startswith(c+"_")]
                    selected = list(dict.fromkeys(selected + expanded))[:k]
                else:
                    selected = all_cols

                Xtr = Xtr_all[selected]
                Xte = Xte_all[selected]
                model = build_model(model_name, category=True, seed=SEED)

                t0 = time.perf_counter(); model.fit(Xtr, tr["target"]); train_time = time.perf_counter()-t0
                t0 = time.perf_counter(); yhat = model.predict(Xte); pred_time = time.perf_counter()-t0
                m = metrics(te["target"], yhat)
                results.append({
                    "horizon":h,"fold":fold,"model":model_name,"n_features":len(selected),
                    "feature_set":"All" if not model_name.startswith("MI_") else f"Top{len(selected)}",
                    "train_end":str(pd.to_datetime(train_dates[-1]).date()),
                    "test_start":str(pd.to_datetime(test_dates[0]).date()),
                    "test_end":str(pd.to_datetime(test_dates[-1]).date()),
                    "n_train":len(tr),"n_test":len(te),
                    "training_time_seconds":train_time,"prediction_time_seconds":pred_time,
                    **m
                })
                p = te[["cat_id","store_id","date","target_date","target"]].copy()
                p["horizon"]=h; p["fold"]=fold; p["model"]=model_name; p["predicted"]=yhat
                preds.append(p)

    fold_df = pd.DataFrame(results)
    fold_df.to_csv(OUT/"category_store_fold_performance.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(OUT/"category_store_predictions.csv", index=False)

    summary = fold_df.groupby(["horizon","model","n_features","feature_set"], as_index=False).agg(
        RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"),
        MAE_mean=("MAE","mean"), MAE_std=("MAE","std"),
        sMAPE_mean=("sMAPE","mean"), sMAPE_std=("sMAPE","std"),
        R2_mean=("R2","mean"), R2_std=("R2","std"),
        training_time_mean=("training_time_seconds","mean"),
        training_time_std=("training_time_seconds","std"),
        prediction_time_mean=("prediction_time_seconds","mean"),
        prediction_time_std=("prediction_time_seconds","std"),
    )
    summary.to_csv(OUT/"category_store_model_performance.csv", index=False)
    print("Step 06 complete.")

if __name__ == "__main__":
    main()
