"""Step 07 — Entropy-TOPSIS computation and reference-result verification.

The repository retains the exact reported overall ranking as a reference artifact.
This script recomputes the decision-support layer from generated performance data
and explicitly reports agreement/disagreement rather than silently hard-coding results.
"""
import numpy as np,pandas as pd
from common import OUT,REF,ensure_dirs,save_json

CRITERIA=["RMSE_mean","MAE_mean","sMAPE_mean","R2_mean","training_time_mean","prediction_time_mean","n_features"]
COST={"RMSE_mean","MAE_mean","sMAPE_mean","training_time_mean","prediction_time_mean","n_features"}

def orient_for_entropy(df):
    x=df[CRITERIA].to_numpy(float)
    y=np.empty_like(x)
    for j,c in enumerate(CRITERIA):
        col=x[:,j]
        if c in COST:
            y[:,j]=1.0/(col+1e-12)
        else:
            y[:,j]=col-col.min()+1e-12
    return y

def entropy_weights(df):
    y=orient_for_entropy(df)
    p=y/(y.sum(axis=0)+1e-30)
    k=1/np.log(len(df))
    e=-k*np.sum(np.where(p>0,p*np.log(p),0.0),axis=0)
    d=1-e
    return d/(d.sum()+1e-30)

def topsis(df,w):
    x=df[CRITERIA].to_numpy(float)
    z=x/(np.sqrt((x*x).sum(axis=0))+1e-30)
    v=z*w
    best=[];worst=[]
    for j,c in enumerate(CRITERIA):
        if c in COST:
            best.append(v[:,j].min());worst.append(v[:,j].max())
        else:
            best.append(v[:,j].max());worst.append(v[:,j].min())
    best=np.asarray(best);worst=np.asarray(worst)
    dp=np.sqrt(((v-best)**2).sum(axis=1));dm=np.sqrt(((v-worst)**2).sum(axis=1))
    out=df.copy()
    out["TOPSIS_D_plus"]=dp;out["TOPSIS_D_minus"]=dm
    out["TOPSIS_closeness"]=dm/(dp+dm)
    out["TOPSIS_rank"]=out["TOPSIS_closeness"].rank(ascending=False,method="min").astype(int)
    return out.sort_values("TOPSIS_rank")

def main():
    ensure_dirs()
    perf=pd.read_csv(OUT/"category_store_model_performance.csv")
    weight_rows=[];rank_rows=[]
    for h,g in perf.groupby("horizon"):
        w=entropy_weights(g)
        weight_rows.extend([{"horizon":h,"criterion":c,"entropy_weight":float(v)} for c,v in zip(CRITERIA,w)])
        r=topsis(g,w);r.insert(0,"ranking_scope",f"horizon_{h}");rank_rows.append(r)

    overall=perf.groupby("model",as_index=False).agg(
        n_features=("n_features","first"),
        RMSE_mean=("RMSE_mean","mean"),MAE_mean=("MAE_mean","mean"),sMAPE_mean=("sMAPE_mean","mean"),
        R2_mean=("R2_mean","mean"),training_time_mean=("training_time_mean","mean"),
        prediction_time_mean=("prediction_time_mean","mean"))
    w=entropy_weights(overall)
    weight_rows.extend([{"horizon":"Overall","criterion":c,"entropy_weight":float(v)} for c,v in zip(CRITERIA,w)])
    r=topsis(overall,w);r.insert(0,"horizon","Overall")
    r.to_csv(OUT/"overall_entropy_topsis_ranking_recomputed.csv",index=False)
    pd.DataFrame(weight_rows).to_csv(OUT/"entropy_weights.csv",index=False)
    pd.concat(rank_rows,ignore_index=True).to_csv(OUT/"horizon_entropy_topsis_ranking.csv",index=False)

    ref=pd.read_csv(REF/"overall_entropy_topsis_ranking_reference.csv")
    check=r[["model","TOPSIS_rank"]].merge(ref[["model","TOPSIS_rank","TOPSIS_closeness"]],on="model",suffixes=("_recomputed","_reported"))
    check["rank_match"]=check["TOPSIS_rank_recomputed"].eq(check["TOPSIS_rank_reported"])
    check.to_csv(OUT/"entropy_topsis_reference_check.csv",index=False)
    save_json({"all_reported_ranks_reproduced":bool(check.rank_match.all())},OUT/"entropy_topsis_check.json")
    print("Step 07 complete. See entropy_topsis_reference_check.csv.")

if __name__=="__main__":
    main()
