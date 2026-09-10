"""Step 12 — assemble supplementary analytical tables as CSV files."""
import pandas as pd
from common import OUT, ensure_dirs

def main():
    ensure_dirs()
    mapping={
        "supp_table_model_settings.csv": pd.DataFrame([
            ["Ridge_All","alpha=1.0","StandardScaler"],
            ["RandomForest_All","n_estimators=80; max_depth=16; min_samples_leaf=2; random_state=42","none"],
            ["Boosting_All","HistGradientBoostingRegressor; learning_rate=0.06; max_leaf_nodes=31; l2_regularization=0.1; random_state=42","none"],
            ["MLP_All","hidden_layer_sizes=(96,48); relu; adam; alpha=0.0001; learning_rate_init=0.001; early_stopping=True; validation_fraction=0.1; random_state=42","StandardScaler"],
            ["MI_MLP_Top10","same MLP architecture; Top-10 MI predictors","StandardScaler"],
            ["MI_MLP_Top15","same MLP architecture; Top-15 MI predictors; category-store only","StandardScaler"],
            ["MI_MLP_Top20","same MLP architecture; Top-20 MI predictors; category-store only","StandardScaler"],
        ],columns=["model","configuration","preprocessing"]),
    }
    for name,df in mapping.items():
        df.to_csv(OUT/name,index=False)

    # Copy generated result layers with publication-neutral names.
    files=[
        "category_store_mi_feature_ranking.csv",
        "category_store_fold_performance.csv",
        "category_store_model_performance.csv",
        "entropy_weights.csv",
        "horizon_entropy_topsis_ranking.csv",
        "item_store_regime_all_series.csv",
        "item_store_selected_series.csv",
        "item_store_regime_summary.csv",
        "item_store_mi_feature_ranking.csv",
        "item_store_regime_fold_performance.csv",
        "item_store_model_performance_summary.csv",
    ]
    for f in files:
        p=OUT/f
        if p.exists():
            pd.read_csv(p).to_csv(OUT/("supp_"+f),index=False)
    print("Step 12 complete.")

if __name__=="__main__":
    main()
