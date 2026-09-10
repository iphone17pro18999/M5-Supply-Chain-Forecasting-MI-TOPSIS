"""Step 05 — horizon-specific Mutual Information using only the initial training block."""
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from common import PROCESSED,OUT,REF,SEED,ensure_dirs
from modeling import candidate_columns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import time

def encode_for_mi(train, cols):
    cat=[c for c in cols if str(train[c].dtype) in ("object","category")]
    num=[c for c in cols if c not in cat]
    prep=ColumnTransformer([
        ("num",SimpleImputer(strategy="median"),num),
        ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),
                         ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)
    ],verbose_feature_names_out=False)
    X=prep.fit_transform(train[cols])
    return X,list(prep.get_feature_names_out())

def main():
    ensure_dirs()
    df=pd.read_csv(PROCESSED/"category_store_features.csv",parse_dates=["date"])
    folds=pd.read_csv(REF/"category_store_fold_definition.csv",parse_dates=["train_target_start","train_target_end"])
    allrows=[]
    for h in [1,7,14,28]:
        tmp=df.sort_values(["cat_id","store_id","date"]).copy()
        tmp["target"]=tmp.groupby(["cat_id","store_id"])["demand"].shift(-h)
        tmp["target_date"]=tmp["date"]+pd.to_timedelta(h,unit="D")
        tmp=tmp.dropna(subset=["target"])
        f1=folds[(folds.horizon==h)&(folds.fold==1)].iloc[0]
        tr=tmp[(tmp.target_date>=f1.train_target_start)&(tmp.target_date<=f1.train_target_end)].copy()
        cols=candidate_columns(tr)
        X,names=encode_for_mi(tr,cols)
        t0=time.perf_counter()
        mi=mutual_info_regression(X,tr["target"].to_numpy(),random_state=SEED,n_jobs=1)
        rt=time.perf_counter()-t0
        r=pd.DataFrame({"horizon":h,"feature":names,"MI_score":mi}).sort_values("MI_score",ascending=False).reset_index(drop=True)
        r["MI_rank"]=r.index+1;r["MI_runtime_seconds"]=rt
        allrows.append(r)
    pd.concat(allrows,ignore_index=True).to_csv(OUT/"category_store_mi_feature_ranking.csv",index=False)
    print("Step 05 complete.")

if __name__=="__main__":
    main()
