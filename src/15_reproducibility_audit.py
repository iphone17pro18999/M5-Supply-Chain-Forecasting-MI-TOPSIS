"""Step 15 — automated audit of expected outputs, structure, and key result tolerances."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from common import OUT, REF, PROCESSED, ensure_dirs, save_json

EXPECTED = [
    "category_store_model_performance.csv",
    "overall_entropy_topsis_ranking.csv",
    "item_store_mi_feature_ranking.csv",
    "item_store_model_performance_summary.csv",
]

def compare_numeric(generated, reference, keys, metrics, rtol=0.08, atol=1e-6):
    g=pd.read_csv(generated); r=pd.read_csv(reference)
    m=g.merge(r,on=keys,suffixes=("_g","_r"))
    checks={}
    for col in metrics:
        ok=np.allclose(m[f"{col}_g"],m[f"{col}_r"],rtol=rtol,atol=atol,equal_nan=True)
        checks[col]=bool(ok)
    return checks

def main():
    ensure_dirs()
    report={"files":{},"checks":{}}
    for f in EXPECTED:
        report["files"][f]=(OUT/f).exists()

    # Structural/leakage checks
    if (PROCESSED/"item_store_features.csv").exists():
        cols=pd.read_csv(PROCESSED/"item_store_features.csv",nrows=2).columns
        report["checks"]["ADI_present_for_reporting"]="ADI" in cols
        report["checks"]["CV2_present_for_reporting"]="CV2" in cols

    # Reference comparisons: tolerances allow expected runtime/platform and small numerical variation.
    pairs=[
        ("category_store_model_performance.csv","category_store_model_performance_reference.csv",
         ["horizon","model"],["RMSE_mean","MAE_mean","sMAPE_mean","R2_mean"]),
        ("overall_entropy_topsis_ranking.csv","overall_entropy_topsis_ranking_reference.csv",
         ["model"],["RMSE_mean","MAE_mean","sMAPE_mean","R2_mean","TOPSIS_closeness"]),
        ("item_store_model_performance_summary.csv","item_store_model_performance_summary_reference.csv",
         ["model","demand_regime"],["RMSE_mean","MAE_mean","sMAPE_mean","R2_mean"]),
    ]
    for gf,rf,keys,cols in pairs:
        if (OUT/gf).exists() and (REF/rf).exists():
            report["checks"][gf]=compare_numeric(OUT/gf,REF/rf,keys,cols)

    report["passed_required_files"]=all(report["files"].values())
    save_json(report,OUT/"reproducibility_audit.json")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
