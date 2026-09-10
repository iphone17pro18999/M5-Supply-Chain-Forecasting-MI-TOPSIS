"""Step 10 — three-fold 28-day item-store rolling-origin robustness experiment."""
import time
import pandas as pd
from common import PROCESSED, OUT, SEED, ensure_dirs, metrics, rolling_origin_date_windows
from modeling import prepare_numeric_matrix, build_model

MODELS=["Ridge_All","Boosting_All","MLP_All","MI_MLP_Top10"]

def main():
    ensure_dirs()
    df=pd.read_csv(PROCESSED/"item_store_features.csv",parse_dates=["date"])
    mi=pd.read_csv(OUT/"item_store_mi_feature_ranking.csv")
    top10=mi.nsmallest(10,"MI_rank")["feature"].tolist()

    tmp=df.sort_values(["item_id","store_id","date"]).copy()
    tmp["target"]=tmp.groupby(["item_id","store_id"])["demand"].shift(-28)
    tmp["target_date"]=tmp["date"]+pd.to_timedelta(28,unit="D")
    tmp=tmp.dropna(subset=["target"]).copy()

    rows=[]; pred_rows=[]
    for fold,train_dates,test_dates in rolling_origin_date_windows(tmp["target_date"].unique(),3,28):
        tr=tmp[tmp["target_date"].isin(train_dates)].copy()
        te=tmp[tmp["target_date"].isin(test_dates)].copy()
        Xtr_all,Xte_all,all_cols=prepare_numeric_matrix(tr,te,"target")
        for model_name in MODELS:
            if model_name=="MI_MLP_Top10":
                expanded=[]
                for c in top10:
                    expanded += [x for x in all_cols if x==c or x.startswith(c+"_")]
                selected=list(dict.fromkeys(expanded))[:10]
            else:
                selected=all_cols
            Xtr=Xtr_all[selected]; Xte=Xte_all[selected]
            model=build_model(model_name,category=False,seed=SEED)
            t0=time.perf_counter(); model.fit(Xtr,tr["target"]); tt=time.perf_counter()-t0
            t0=time.perf_counter(); yp=model.predict(Xte); pt=time.perf_counter()-t0

            te2=te.copy(); te2["predicted"]=yp
            for reg,g in [("All_Regimes",te2)]+list(te2.groupby("demand_regime")):
                m=metrics(g["target"],g["predicted"])
                rows.append({
                    "horizon":28,"fold":fold,"model":model_name,"demand_regime":reg,
                    "n_features":len(selected),
                    "train_end":str(pd.to_datetime(train_dates[-1]).date()),
                    "test_start":str(pd.to_datetime(test_dates[0]).date()),
                    "test_end":str(pd.to_datetime(test_dates[-1]).date()),
                    "n_train":len(tr),"n_test":len(g),
                    "training_time_seconds":tt,"prediction_time_seconds":pt,
                    **m
                })
            p=te2[["item_id","dept_id","cat_id","store_id","state_id","date","target_date","demand_regime","target","predicted"]].copy()
            p["horizon"]=28;p["fold"]=fold;p["model"]=model_name
            pred_rows.append(p)

    fold=pd.DataFrame(rows)
    fold.to_csv(OUT/"item_store_regime_fold_performance.csv",index=False)
    pd.concat(pred_rows,ignore_index=True).to_csv(OUT/"item_store_predictions.csv",index=False)
    summary=fold.groupby(["model","demand_regime","n_features"],as_index=False).agg(
        RMSE_mean=("RMSE","mean"),RMSE_std=("RMSE","std"),
        MAE_mean=("MAE","mean"),MAE_std=("MAE","std"),
        sMAPE_mean=("sMAPE","mean"),sMAPE_std=("sMAPE","std"),
        R2_mean=("R2","mean"),R2_std=("R2","std"),
        training_time_mean=("training_time_seconds","mean"),
        prediction_time_mean=("prediction_time_seconds","mean")
    )
    summary.to_csv(OUT/"item_store_model_performance_summary.csv",index=False)
    print("Step 10 complete.")

if __name__=="__main__":
    main()
