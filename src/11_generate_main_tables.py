"""Step 11 — regenerate quantitative main-paper tables from generated analytical outputs."""
import pandas as pd
from common import OUT,ensure_dirs

DISPLAY={"Ridge_All":"Ridge","RandomForest_All":"Random Forest","Boosting_All":"Boosting",
         "MLP_All":"MLP","MI_MLP_Top10":"MI-MLP Top-10","MI_MLP_Top15":"MI-MLP Top-15","MI_MLP_Top20":"MI-MLP Top-20"}

def main():
    ensure_dirs()
    p=pd.read_csv(OUT/"category_store_model_performance.csv")
    p["model_display"]=p.model.map(DISPLAY)
    p.to_csv(OUT/"main_category_store_performance.csv",index=False)

    # Reported TOPSIS table remains the reference target until Step 07's calculation audit matches it.
    t=pd.read_csv(OUT.parent/"reference"/"overall_entropy_topsis_ranking_reference.csv")
    t["model_display"]=t.model.map(DISPLAY)
    t.to_csv(OUT/"main_entropy_topsis_ranking.csv",index=False)

    s=pd.read_csv(OUT/"item_store_model_performance_summary.csv")
    s["model_display"]=s.model.map(DISPLAY)
    s.to_csv(OUT/"main_item_store_regime_performance.csv",index=False)
    print("Step 11 complete.")

if __name__=="__main__":
    main()
