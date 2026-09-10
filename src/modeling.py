from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

DROP_ALWAYS={
    "date","d","demand","target","target_date","ADI","CV2","demand_regime","id",
    "event_name_1","event_name_2","event_type_1","event_type_2","weekday"
}

def candidate_columns(df,target="target"):
    return [c for c in df.columns if c not in DROP_ALWAYS and c != target]

def split_column_types(df, cols):
    cat=[c for c in cols if str(df[c].dtype) in ("object","category")]
    num=[c for c in cols if c not in cat]
    return num,cat

def make_preprocessor(train, cols, scale_numeric=False):
    num,cat=split_column_types(train,cols)
    num_steps=[("impute",SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale",StandardScaler()))
    num_pipe=Pipeline(num_steps)
    cat_pipe=Pipeline([
        ("impute",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
    ])
    return ColumnTransformer(
        [("num",num_pipe,num),("cat",cat_pipe,cat)],
        remainder="drop", verbose_feature_names_out=False
    )

def estimator(name,category=True,seed=42):
    if name=="Ridge_All":
        return Ridge(alpha=1.0)
    if name=="RandomForest_All":
        return RandomForestRegressor(n_estimators=80,max_depth=16,min_samples_leaf=2,random_state=seed,n_jobs=-1)
    if name=="Boosting_All":
        return HistGradientBoostingRegressor(
            max_iter=180 if category else 160,
            learning_rate=0.06,max_leaf_nodes=31,l2_regularization=0.1,random_state=seed
        )
    if name.startswith("MLP") or name.startswith("MI_MLP"):
        return MLPRegressor(
            hidden_layer_sizes=(96,48),activation="relu",solver="adam",
            alpha=0.0001,learning_rate_init=0.001,
            max_iter=180 if category else 160,
            early_stopping=True,validation_fraction=0.1,random_state=seed
        )
    raise KeyError(name)

def build_pipeline(name, train, cols, category=True, seed=42):
    scale = (name=="Ridge_All") or name.startswith("MLP") or name.startswith("MI_MLP")
    prep=make_preprocessor(train,cols,scale_numeric=scale)
    return Pipeline([("preprocess",prep),("model",estimator(name,category=category,seed=seed))])
