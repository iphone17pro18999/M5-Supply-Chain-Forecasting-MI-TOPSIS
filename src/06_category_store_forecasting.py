"""Step 06 — reproduce five rolling-origin category-store folds with fixed 28-day validation/test blocks."""
import time, pandas as pd
from common import PROCESSED,OUT,REF,SEED,ensure_dirs,metrics
from modeling import candidate_columns,build_pipeline,build_exact_mi_mlp_pipeline

MODELS=["Ridge_All","RandomForest_All","Boosting_All","MLP_All",
        "MI_MLP_Top10","MI_MLP_Top15","MI_MLP_Top20"]

def main():
    ensure_dirs()
    df=pd.read_csv(PROCESSED/"category_store_features.csv",parse_dates=["date"])
    folddef=pd.read_csv(REF/"category_store_fold_definition.csv",parse_dates=[
        "train_target_start","train_target_end","validation_target_start","validation_target_end",
        "test_target_start","test_target_end"])
    mi=pd.read_csv(OUT/"category_store_mi_feature_ranking.csv")
    rows=[];preds=[]

    for h in [1,7,14,28]:
        tmp=df.sort_values(["cat_id","store_id","date"]).copy()
        tmp["target"]=tmp.groupby(["cat_id","store_id"])["demand"].shift(-h)
        tmp["target_date"]=tmp["date"]+pd.to_timedelta(h,unit="D")
        tmp=tmp.dropna(subset=["target"])
        top={k:mi[mi.horizon.eq(h)].nsmallest(k,"MI_rank")["feature"].tolist() for k in (10,15,20)}

        for _,fd in folddef[folddef.horizon.eq(h)].iterrows():
            tr=tmp[(tmp.target_date>=fd.train_target_start)&(tmp.target_date<=fd.train_target_end)].copy()
            va=tmp[(tmp.target_date>=fd.validation_target_start)&(tmp.target_date<=fd.validation_target_end)].copy()
            te=tmp[(tmp.target_date>=fd.test_target_start)&(tmp.target_date<=fd.test_target_end)].copy()
            if len(tr)!=fd.n_train or len(va)!=fd.n_validation or len(te)!=fd.n_test:
                raise AssertionError(f"Fold-size mismatch H{h} F{fd.fold}: {len(tr)}, {len(va)}, {len(te)}")

            all_cols=candidate_columns(tr)
            for name in MODELS:
                if name.startswith("MI_MLP_Top"):
                    k=int(name.split("Top")[-1])
                    selected_encoded=top[k]
                    selected=all_cols
                    pipe=build_exact_mi_mlp_pipeline(tr,all_cols,selected_encoded,category=True,seed=SEED)
                else:
                    selected=all_cols
                    pipe=build_pipeline(name,tr,selected,category=True,seed=SEED)
                t0=time.perf_counter();pipe.fit(tr[selected],tr["target"]);fit_s=time.perf_counter()-t0

                # Validation block is retained chronologically for audit/tuning provenance.
                t0=time.perf_counter();vhat=pipe.predict(va[selected]);val_pred_s=time.perf_counter()-t0
                val_m=metrics(va["target"],vhat)

                t0=time.perf_counter();that=pipe.predict(te[selected]);pred_s=time.perf_counter()-t0
                test_m=metrics(te["target"],that)

                rows.append({
                    "horizon":h,"fold":int(fd.fold),"model":name,
                    "n_features_declared":60 if not name.startswith("MI_") else int(name.split("Top")[-1]),
                    "n_train":len(tr),"n_validation":len(va),"n_test":len(te),
                    "training_time_seconds":fit_s,"validation_prediction_time_seconds":val_pred_s,
                    "prediction_time_seconds":pred_s,
                    "validation_RMSE":val_m["RMSE"],"validation_MAE":val_m["MAE"],
                    "validation_sMAPE":val_m["sMAPE"],"validation_R2":val_m["R2"],
                    **test_m
                })
                p=te[["cat_id","store_id","date","target_date","target"]].copy()
                p["horizon"]=h;p["fold"]=int(fd.fold);p["model"]=name;p["predicted"]=that
                preds.append(p)

    fold=pd.DataFrame(rows)
    fold.to_csv(OUT/"category_store_fold_performance.csv",index=False)
    pd.concat(preds,ignore_index=True).to_csv(OUT/"category_store_predictions.csv",index=False)
    summary=fold.groupby(["horizon","model","n_features_declared"],as_index=False).agg(
        RMSE_mean=("RMSE","mean"),RMSE_std=("RMSE","std"),
        MAE_mean=("MAE","mean"),MAE_std=("MAE","std"),
        sMAPE_mean=("sMAPE","mean"),sMAPE_std=("sMAPE","std"),
        R2_mean=("R2","mean"),R2_std=("R2","std"),
        training_time_mean=("training_time_seconds","mean"),training_time_std=("training_time_seconds","std"),
        prediction_time_mean=("prediction_time_seconds","mean"),prediction_time_std=("prediction_time_seconds","std")
    ).rename(columns={"n_features_declared":"n_features"})
    summary["feature_set"]=summary["n_features"].map(lambda n:"All" if n==60 else f"Top{n}")
    summary.to_csv(OUT/"category_store_model_performance.csv",index=False)
    print("Step 06 complete.")

if __name__=="__main__":
    main()
