"""Step 01 — immutable study settings used by all downstream scripts."""
from common import ensure_dirs, save_json, OUT, SEED, EVAL_BLOCK_DAYS

CONFIG={
 "random_seed":SEED,
 "category_store_series":30,
 "category_store_horizons":[1,7,14,28],
 "category_store_folds":5,
 "category_store_validation_days":28,
 "category_store_test_days":28,
 "item_store_horizon":28,
 "item_store_folds":3,
 "item_store_test_days":28,
 "screened_item_store_series":5313,
 "regime_counts":{"Smooth":229,"Intermittent":3924,"Erratic":121,"Lumpy":1039},
 "selected_series_per_regime":30,
 "selected_item_store_series":120,
 "lags":[1,7,14,28],
 "rolling_windows":[7,14,28],
 "regime_thresholds":{"ADI":1.32,"CV2":0.49},
 "full_category_store_encoded_predictors_expected":60,
 "full_item_store_encoded_predictors_expected":132
}

def main():
    ensure_dirs()
    save_json(CONFIG,OUT/"study_configuration.json")
    print("Step 01 complete.")

if __name__=="__main__":
    main()
