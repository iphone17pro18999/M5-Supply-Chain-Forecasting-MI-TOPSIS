from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

DROP_ALWAYS = {
    "date","d","demand","target","target_date","ADI","CV2","demand_regime",
    "event_name_1","event_name_2","event_type_1","event_type_2","weekday",
    "id"
}

def prepare_numeric_matrix(train, test, target, exclude=()):
    cols = [c for c in train.columns if c not in DROP_ALWAYS and c not in set(exclude) and c != target]
    Xtr = train[cols].copy()
    Xte = test[cols].copy()
    # one-hot categoricals jointly, deterministic column order
    allx = pd.concat([Xtr, Xte], axis=0, ignore_index=True)
    allx = pd.get_dummies(allx, columns=allx.select_dtypes(include=["object","category"]).columns, dummy_na=False)
    allx = allx.replace([np.inf,-np.inf], np.nan)
    med = allx.iloc[:len(Xtr)].median(numeric_only=True)
    allx = allx.fillna(med).fillna(0)
    return allx.iloc[:len(Xtr)].copy(), allx.iloc[len(Xtr):].copy(), list(allx.columns)

def build_model(name, *, category=True, seed=42):
    if name == "Ridge_All":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    if name == "RandomForest_All":
        return RandomForestRegressor(
            n_estimators=80, max_depth=16, min_samples_leaf=2,
            random_state=seed, n_jobs=-1
        )
    if name == "Boosting_All":
        return HistGradientBoostingRegressor(
            max_iter=180 if category else 160, learning_rate=0.06,
            max_leaf_nodes=31, l2_regularization=0.1, random_state=seed
        )
    if name.startswith("MLP") or name.startswith("MI_MLP"):
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(96,48), activation="relu", solver="adam",
                alpha=0.0001, learning_rate_init=0.001,
                max_iter=180 if category else 160,
                early_stopping=True, validation_fraction=0.1,
                random_state=seed
            ))
        ])
    raise KeyError(name)
