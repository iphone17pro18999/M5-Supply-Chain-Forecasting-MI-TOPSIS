"""Step 10 — reproduce the exact three fixed target-date folds for 28-day item-store forecasting."""
import time,pandas as pd
from common import PROCESSED,OUT,REF,SEED,ensure_dirs,metrics
from modeling import candidate_columns,build_pipeline

MODELS=["Ridge_All","Boosting_All","MLP_All","MI_MLP_Top10"]

def main():
    ensure_dirs()
    df=pd.read_csv(PROCESSED/"item_store_features.csv",parse_dates=["date"])
    mi=pd.read_csv(OUT/"item_store_mi_feature_ranking.csv")
    top10=mi.nsmallest(10,"MI_rank")["feature"].tolist()
    tmp=df.sort_values(["item_id","store_id","date"]).copy()
    tmp["target"]=tmp.groupby(["item_id","store_id"])["demand"].shift(-28)
    tmp["target_date"]=tmp.date+pd.to_timedelta(28,unit="D")
    tmp=tmp.dropna(subset=["target"])
    folds=pd.read_csv(REF/"item_store_fold_definition.csv",parse_dates=[
        "train_target_start","train_target_end","test_target_start","test_target_end"])
    rows=[];preds=[]
    for _,fd in folds.iterrows():
        tr=tmp[(tmp.target_date>=fd.train_target_start)&(tmp.target_date<=fd.train_target_end)].copy()
        te=tmp[(tmp.target_date>=fd.test_target_start)&(tmp.target_date<=fd.test_target_end)].copy()
        if len(tr)!=fd.n_train or len(te)!=fd.n_test:
            raise AssertionError(f"Fold {fd.fold} size mismatch: {len(tr)}, {len(te)}")
        allcols=candidate_columns(tr)
        for name in MODELS:
            if name=="MI_MLP_Top10":
                selected=[]
                for feat in top10:
                    if feat in allcols:selected.append(feat)
                    else:
                        base=feat.split("_")[0]
                        if base in allcols and base not in selected:selected.append(base)
                selected=selected[:10]
            else:selected=allcols
            pipe=build_pipeline(name,tr,selected,category=False,seed=SEED)
            t0=time.perf_counter();pipe.fit(tr[selected],tr.target);fit_s=time.perf_counter()-t0
            t0=time.perf_counter();yp=pipe.predict(te[selected]);pred_s=time.perf_counter()-t0
            te2=te.copy();te2["predicted"]=yp
            for reg,g in [("All_Regimes",te2)]+list(te2.groupby("demand_regime")):
                m=metrics(g.target,g.predicted)
                rows.append({"horizon":28,"fold":int(fd.fold),"model":name,"demand_regime":reg,
                             "n_features":132 if name!="MI_MLP_Top10" else 10,
                             "n_train":len(tr),"n_test":len(g),
                             "training_time_seconds":fit_s,"prediction_time_seconds":pred_s,**m})
            p=te2[["item_id","dept_id","cat_id","store_id","state_id","date","target_date",
                    "demand","demand_regime","target","predicted"]].copy()
            p["horizon"]=28;p["fold"]=int(fd.fold);p["model"]=name;preds.append(p)
    fold=pd.DataFrame(rows)
    fold.to_csv(OUT/"item_store_regime_fold_performance.csv",index=False)
    pd.concat(preds,ignore_index=True).to_csv(OUT/"item_store_predictions.csv",index=False)
    sm=fold.groupby(["model","demand_regime","n_features"],as_index=False).agg(
        RMSE_mean=("RMSE","mean"),RMSE_std=("RMSE","std"),MAE_mean=("MAE","mean"),MAE_std=("MAE","std"),
        sMAPE_mean=("sMAPE","mean"),sMAPE_std=("sMAPE","std"),R2_mean=("R2","mean"),R2_std=("R2","std"),
        training_time_mean=("training_time_seconds","mean"),prediction_time_mean=("prediction_time_seconds","mean"))
    sm.to_csv(OUT/"item_store_model_performance_summary.csv",index=False)
    print("Step 10 complete.")

if __name__=="__main__":
    main()
