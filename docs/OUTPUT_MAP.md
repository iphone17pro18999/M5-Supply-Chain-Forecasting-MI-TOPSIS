# Output-to-code map

| Analytical output | Generating step | Generated file |
|---|---|---|
| Raw M5 schema verification | 02 | `raw_data_validation.json` |
| Category-store daily data | 03 | `data/processed/category_store_daily.csv` |
| Category-store engineered data | 04 | `data/processed/category_store_features.csv` |
| Category-store MI ranking | 05 | `category_store_mi_feature_ranking.csv` |
| Category-store fold metrics | 06 | `category_store_fold_performance.csv` |
| Category-store performance summary | 06 | `category_store_model_performance.csv` |
| Horizon and overall Entropy-TOPSIS | 07 | `horizon_entropy_topsis_ranking.csv`, `overall_entropy_topsis_ranking.csv` |
| Entropy criterion weights | 07 | `entropy_weights.csv` |
| Full item-store regime classification | 08 | `item_store_regime_all_series.csv` |
| Balanced item-store series list | 08 | `item_store_selected_series.csv` |
| Item-store engineered data | 09 | `data/processed/item_store_features.csv` |
| Item-store MI ranking | 09 | `item_store_mi_feature_ranking.csv` |
| Item-store fold robustness | 10 | `item_store_regime_fold_performance.csv` |
| Item-store aggregated robustness | 10 | `item_store_model_performance_summary.csv` |
| Main numerical tables | 11 | `main_table_*.csv` |
| Supplementary numerical tables | 12 | `supp_*.csv` |
| Main Figures 1–6 | 13 | `figures/generated/figure_*.png/pdf` |
| Supplementary diagnostic figures | 14 | `figures/generated/supp_figure_*.png/pdf` |
| Final reproducibility audit | 15 | `reproducibility_audit.json` |
