"""Step 15 — strict structural/numerical audit against reference study outputs."""
import json,numpy as np,pandas as pd
from common import OUT,REF,PROCESSED,ensure_dirs,save_json

def close_series(g,r,keys,cols,rtol=1e-3,atol=1e-6):
    m=g.merge(r,on=keys,suffixes=("_g","_r"))
    if len(m)!=len(r):return False
    return all(np.allclose(m[f"{c}_g"],m[f"{c}_r"],rtol=rtol,atol=atol,equal_nan=True) for c in cols)

def main():
    ensure_dirs()
    report={"checks":{}}
    required=["category_store_model_performance.csv","item_store_mi_feature_ranking.csv",
              "item_store_model_performance_summary.csv","entropy_topsis_reference_check.csv"]
    report["checks"]["required_outputs_exist"]=all((OUT/f).exists() for f in required)

    # Fold metadata
    cfd=pd.read_csv(REF/"category_store_fold_definition.csv")
    report["checks"]["category_fold_rows_20"]=len(cfd)==20
    report["checks"]["category_validation_and_test_each_840"]=bool((cfd.n_validation.eq(840)&cfd.n_test.eq(840)).all())
    if (OUT/"category_store_fold_performance.csv").exists():
        cf=pd.read_csv(OUT/"category_store_fold_performance.csv")
        report["checks"]["category_5_folds_x_4_horizons"]=cf[["horizon","fold"]].drop_duplicates().shape[0]==20

    # Item series and folds
    sel=pd.read_csv(REF/"selected_item_store_series.csv")
    report["checks"]["selected_item_series_120"]=sel.id.nunique()==120
    report["checks"]["selected_30_each_regime"]=sel.demand_regime.value_counts().eq(30).all()
    if (OUT/"item_store_regime_all_series.csv").exists():
        full=pd.read_csv(OUT/"item_store_regime_all_series.csv")
        counts=full.demand_regime.value_counts().to_dict()
        report["checks"]["full_regime_counts_match"]=counts=={"Intermittent":3924,"Lumpy":1039,"Smooth":229,"Erratic":121}

    # Numerical checks exclude runtime because it is hardware dependent.
    if (OUT/"category_store_model_performance.csv").exists():
        g=pd.read_csv(OUT/"category_store_model_performance.csv")
        r=pd.read_csv(REF/"category_store_model_performance_reference.csv")
        report["checks"]["category_metrics_match_reference"]=close_series(
            g,r,["horizon","model"],["RMSE_mean","MAE_mean","sMAPE_mean","R2_mean"],rtol=5e-3)

    if (OUT/"item_store_model_performance_summary.csv").exists():
        g=pd.read_csv(OUT/"item_store_model_performance_summary.csv")
        r=pd.read_csv(REF/"item_store_model_performance_summary_reference.csv")
        report["checks"]["item_metrics_match_reference"]=close_series(
            g,r,["model","demand_regime"],["RMSE_mean","MAE_mean","sMAPE_mean","R2_mean"],rtol=5e-3)

    if (OUT/"item_store_mi_feature_ranking.csv").exists():
        g=pd.read_csv(OUT/"item_store_mi_feature_ranking.csv").nsmallest(10,"MI_rank")
        r=pd.read_csv(REF/"item_store_mi_feature_ranking_reference.csv").nsmallest(10,"MI_rank")
        report["checks"]["item_top10_features_match"]=g.feature.tolist()==r.feature.tolist()

    if (OUT/"entropy_topsis_reference_check.csv").exists():
        q=pd.read_csv(OUT/"entropy_topsis_reference_check.csv")
        report["checks"]["topsis_rank_order_matches_reported"]=bool(q.rank_match.all())

    report["all_checks_pass"]=all(report["checks"].values()) if report["checks"] else False
    save_json(report,OUT/"reproducibility_audit.json")
    print(json.dumps(report,indent=2))
    if not report["all_checks_pass"]:
        raise SystemExit("Reproducibility audit failed. Inspect outputs/generated/reproducibility_audit.json.")

if __name__=="__main__":
    main()
