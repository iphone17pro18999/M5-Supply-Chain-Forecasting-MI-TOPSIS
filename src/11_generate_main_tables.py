"""Step 11 — generate manuscript-facing numerical tables from analysis outputs."""
import pandas as pd
from common import OUT, ensure_dirs

def main():
    ensure_dirs()
    # Table: category-store horizon RMSE subset
    perf=pd.read_csv(OUT/"category_store_model_performance.csv")
    keep=["Ridge_All","Boosting_All","MLP_All","MI_MLP_Top10"]
    p=perf[perf["model"].isin(keep)].pivot(index="horizon",columns="model",values="RMSE_mean").reset_index()
    p["Best_RMSE_model"]=p[[c for c in p.columns if c!="horizon"]].idxmin(axis=1)
    p["Best_RMSE"]=p[[c for c in p.columns if c!="horizon" and c!="Best_RMSE_model"]].min(axis=1)
    p.to_csv(OUT/"main_table_category_store_horizon_rmse.csv",index=False)

    # Overall TOPSIS table
    pd.read_csv(OUT/"overall_entropy_topsis_ranking.csv").to_csv(
        OUT/"main_table_entropy_topsis_ranking.csv",index=False)

    # Regime-wise best metrics
    s=pd.read_csv(OUT/"item_store_model_performance_summary.csv")
    rows=[]
    for reg,g in s.groupby("demand_regime"):
        a=g.loc[g["RMSE_mean"].idxmin()]
        b=g.loc[g["sMAPE_mean"].idxmin()]
        rows.append({
            "demand_regime":reg,"best_RMSE_model":a["model"],"best_RMSE":a["RMSE_mean"],
            "MAE":a["MAE_mean"],"sMAPE":a["sMAPE_mean"],"R2":a["R2_mean"],
            "best_sMAPE_model":b["model"],"best_sMAPE":b["sMAPE_mean"]
        })
    pd.DataFrame(rows).to_csv(OUT/"main_table_item_store_regime_performance.csv",index=False)
    print("Step 11 complete.")

if __name__=="__main__":
    main()
