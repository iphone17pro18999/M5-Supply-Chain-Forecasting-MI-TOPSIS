"""Step 01 — locked study configuration and directory initialization."""
from common import ensure_dirs, save_json, OUT, SEED

CONFIG = {
    "random_seed": SEED,
    "forecast_horizons_days": [1, 7, 14, 28],
    "category_store_folds": 5,
    "item_store_horizon_days": 28,
    "item_store_folds": 3,
    "lags": [1, 7, 14, 28],
    "rolling_windows": [7, 14, 28],
    "item_store_departments": ["FOODS_3", "HOUSEHOLD_1", "HOBBIES_1"],
    "item_store_stores": ["CA_1", "TX_1", "WI_1"],
    "balanced_series_per_regime": 30,
    "regime_thresholds": {"ADI": 1.32, "CV2": 0.49},
    "ridge": {"alpha": 1.0},
    "random_forest": {
        "n_estimators": 80, "max_depth": 16, "min_samples_leaf": 2,
        "random_state": SEED, "n_jobs": -1
    },
    "boosting_category": {
        "max_iter": 180, "learning_rate": 0.06, "max_leaf_nodes": 31,
        "l2_regularization": 0.1, "random_state": SEED
    },
    "boosting_item": {
        "max_iter": 160, "learning_rate": 0.06, "max_leaf_nodes": 31,
        "l2_regularization": 0.1, "random_state": SEED
    },
    "mlp_category": {
        "hidden_layer_sizes": [96, 48], "activation": "relu", "solver": "adam",
        "alpha": 0.0001, "learning_rate_init": 0.001, "max_iter": 180,
        "early_stopping": True, "validation_fraction": 0.1, "random_state": SEED
    },
    "mlp_item": {
        "hidden_layer_sizes": [96, 48], "activation": "relu", "solver": "adam",
        "alpha": 0.0001, "learning_rate_init": 0.001, "max_iter": 160,
        "early_stopping": True, "validation_fraction": 0.1, "random_state": SEED
    }
}

def main():
    ensure_dirs()
    save_json(CONFIG, OUT / "study_configuration.json")
    print("Step 01 complete: configuration written to outputs/generated/study_configuration.json")

if __name__ == "__main__":
    main()
