"""Step 14 — supplementary diagnostic figures generated from analytical outputs."""
import pandas as pd
import matplotlib.pyplot as plt
from common import OUT, FIG, ensure_dirs

def save(fig,name):
    fig.tight_layout()
    fig.savefig(FIG/f"{name}.png",dpi=600,bbox_inches="tight")
    fig.savefig(FIG/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)

def main():
    ensure_dirs()
    # S1: Entropy weights
    w=pd.read_csv(OUT/"entropy_weights.csv").sort_values("entropy_weight")
    fig,ax=plt.subplots(figsize=(8,5));ax.barh(w["criterion"],w["entropy_weight"])
    ax.set_xlabel("Entropy weight");ax.set_title("Entropy-derived criterion weights")
    save(fig,"supp_figure_01_entropy_weights")

    # S2: fold-wise category RMSE
    d=pd.read_csv(OUT/"category_store_fold_performance.csv")
    fig,ax=plt.subplots(figsize=(9,5))
    for m,g in d.groupby("model"):
        g2=g.groupby("horizon")["RMSE"].mean()
        ax.plot(g2.index,g2.values,marker="o",label=m.replace("_All",""))
    ax.set_xlabel("Forecast horizon (days)");ax.set_ylabel("Fold-mean RMSE");ax.legend(fontsize=7)
    ax.set_title("Fold-aggregated category-store RMSE")
    save(fig,"supp_figure_02_category_fold_rmse")

    # S3: regime counts
    r=pd.read_csv(OUT/"item_store_regime_all_series.csv")["demand_regime"].value_counts()
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(r.index,r.values)
    ax.set_ylabel("Number of item-store series");ax.set_title("Full item-store demand-regime distribution")
    save(fig,"supp_figure_03_regime_distribution")
    print("Step 14 complete.")

if __name__=="__main__":
    main()
