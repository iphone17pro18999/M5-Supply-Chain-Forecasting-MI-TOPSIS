"""Step 07 — objective Entropy-TOPSIS ranking from category-store model performance."""
import numpy as np
import pandas as pd
from common import OUT, ensure_dirs

CRITERIA = ["RMSE_mean","MAE_mean","sMAPE_mean","R2_mean",
            "training_time_mean","prediction_time_mean","n_features"]
COST = {"RMSE_mean","MAE_mean","sMAPE_mean","training_time_mean","prediction_time_mean","n_features"}

def entropy_weights(X):
    X = np.asarray(X, dtype=float)
    # Shift each criterion to strictly positive values.
    Y = X - np.nanmin(X, axis=0) + 1e-12
    P = Y / np.sum(Y, axis=0, keepdims=True)
    k = 1.0 / np.log(len(X))
    E = -k * np.sum(np.where(P > 0, P*np.log(P), 0.0), axis=0)
    d = 1.0 - E
    return d / d.sum()

def topsis(df):
    M = df[CRITERIA].to_numpy(float)
    norm = np.sqrt((M**2).sum(axis=0))
    Z = M / np.where(norm==0, 1, norm)
    w = entropy_weights(M)
    V = Z*w
    ideal_plus, ideal_minus = [], []
    for j,c in enumerate(CRITERIA):
        if c in COST:
            ideal_plus.append(V[:,j].min()); ideal_minus.append(V[:,j].max())
        else:
            ideal_plus.append(V[:,j].max()); ideal_minus.append(V[:,j].min())
    ideal_plus=np.array(ideal_plus); ideal_minus=np.array(ideal_minus)
    dplus=np.sqrt(((V-ideal_plus)**2).sum(axis=1))
    dminus=np.sqrt(((V-ideal_minus)**2).sum(axis=1))
    closeness=dminus/(dplus+dminus)
    out=df.copy()
    out["TOPSIS_D_plus"]=dplus; out["TOPSIS_D_minus"]=dminus; out["TOPSIS_closeness"]=closeness
    out["TOPSIS_rank"]=out["TOPSIS_closeness"].rank(ascending=False, method="min").astype(int)
    weights=pd.DataFrame({"criterion":CRITERIA,"entropy_weight":w,
                          "orientation":["cost" if c in COST else "benefit" for c in CRITERIA]})
    return out.sort_values("TOPSIS_rank"), weights

def main():
    ensure_dirs()
    perf=pd.read_csv(OUT/"category_store_model_performance.csv")
    overall=perf.groupby(["model"], as_index=False).agg(
        n_features=("n_features","first"),
        RMSE_mean=("RMSE_mean","mean"), MAE_mean=("MAE_mean","mean"),
        sMAPE_mean=("sMAPE_mean","mean"), R2_mean=("R2_mean","mean"),
        training_time_mean=("training_time_mean","mean"),
        prediction_time_mean=("prediction_time_mean","mean"),
    )
    ranked, weights=topsis(overall)
    ranked.insert(0,"horizon","Overall")
    ranked.to_csv(OUT/"overall_entropy_topsis_ranking.csv", index=False)
    weights.to_csv(OUT/"entropy_weights.csv", index=False)

    horizon_rows=[]
    for h,g in perf.groupby("horizon"):
        r,_=topsis(g[["model","n_features"]+CRITERIA[:-1]].copy())
        r.insert(0,"horizon",h)
        horizon_rows.append(r)
    pd.concat(horizon_rows,ignore_index=True).to_csv(OUT/"horizon_entropy_topsis_ranking.csv",index=False)
    print("Step 07 complete.")

if __name__=="__main__":
    main()
