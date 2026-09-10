"""Step 13 — generate all six main-paper figures using Python."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from common import OUT, FIG, ensure_dirs

def save(fig,name):
    fig.tight_layout()
    fig.savefig(FIG/f"{name}.png",dpi=600,bbox_inches="tight")
    fig.savefig(FIG/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)

def box(ax,xy,w,h,text):
    p=FancyBboxPatch(xy,w,h,boxstyle="round,pad=0.02",fill=False,linewidth=1.4)
    ax.add_patch(p); ax.text(xy[0]+w/2,xy[1]+h/2,text,ha="center",va="center",fontsize=10)
    return p

def fig1():
    fig,ax=plt.subplots(figsize=(10,6)); ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis("off")
    labels=[("M5 sales + calendar + prices",(0.5,4.9)),("Feature engineering",(3.0,4.9)),
            ("Mutual Information ranking",(5.5,4.9)),("Forecasting models",(8.0,4.9)),
            ("Category-store\n1/7/14/28-day validation",(2.0,2.5)),
            ("Item-store\nregime validation",(5.0,2.5)),("Entropy-TOPSIS\nmodel evaluation",(8.0,2.5))]
    boxes=[]
    for t,(x,y) in labels: boxes.append(box(ax,(x,y),1.6,0.7,t))
    arrows=[((2.1,5.25),(3.0,5.25)),((4.6,5.25),(5.5,5.25)),((7.1,5.25),(8.0,5.25)),
            ((8.8,4.9),(2.8,3.2)),((8.8,4.9),(5.8,3.2)),((6.6,2.85),(8.0,2.85))]
    for a,b in arrows: ax.add_patch(FancyArrowPatch(a,b,arrowstyle="->",mutation_scale=14))
    ax.set_title("Conceptual workflow of the forecasting decision-support framework")
    save(fig,"figure_01_conceptual_framework")

def fig2():
    fig,ax=plt.subplots(figsize=(10,5)); ax.axis("off");ax.set_xlim(0,10);ax.set_ylim(0,5)
    box(ax,(0.5,3.2),2.2,0.8,"Layer 1: category-store")
    box(ax,(3.4,3.2),2.2,0.8,"5 rolling-origin folds")
    box(ax,(6.3,3.2),2.8,0.8,"Horizons: 1, 7, 14, 28 days")
    box(ax,(0.5,1.3),2.2,0.8,"Layer 2: item-store")
    box(ax,(3.4,1.3),2.2,0.8,"3 rolling-origin folds")
    box(ax,(6.3,1.3),2.8,0.8,"28-day regime robustness")
    for y in (3.6,1.7):
        ax.add_patch(FancyArrowPatch((2.7,y),(3.4,y),arrowstyle="->",mutation_scale=14))
        ax.add_patch(FancyArrowPatch((5.6,y),(6.3,y),arrowstyle="->",mutation_scale=14))
    ax.set_title("Two-layer rolling-origin validation design")
    save(fig,"figure_02_validation_design")

def fig3():
    df=pd.read_csv(OUT/"item_store_mi_feature_ranking.csv").nsmallest(13,"MI_rank").sort_values("MI_score")
    fig,ax=plt.subplots(figsize=(8,6));ax.barh(df["feature"],df["MI_score"])
    ax.set_xlabel("Mutual Information score");ax.set_ylabel("Feature")
    ax.set_title("Leading item-store predictors for 28-day forecasting")
    save(fig,"figure_03_mi_feature_profile")

def fig4():
    df=pd.read_csv(OUT/"category_store_model_performance.csv")
    models=["Ridge_All","Boosting_All","MLP_All","MI_MLP_Top10"]
    fig,ax=plt.subplots(figsize=(8,5))
    for m in models:
        g=df[df["model"].eq(m)].sort_values("horizon")
        ax.plot(g["horizon"],g["RMSE_mean"],marker="o",label=m.replace("_All",""))
    ax.set_xlabel("Forecast horizon (days)");ax.set_ylabel("Mean RMSE");ax.legend()
    ax.set_title("Category-store forecasting performance across horizons")
    save(fig,"figure_04_horizon_rmse")

def fig5():
    df=pd.read_csv(OUT/"overall_entropy_topsis_ranking.csv").sort_values("TOPSIS_closeness")
    fig,ax=plt.subplots(figsize=(8,5));ax.barh(df["model"].str.replace("_All","",regex=False),df["TOPSIS_closeness"])
    ax.set_xlabel("TOPSIS closeness coefficient");ax.set_ylabel("Model")
    ax.set_title("Overall Entropy-TOPSIS model ranking")
    save(fig,"figure_05_entropy_topsis_ranking")

def fig6():
    df=pd.read_csv(OUT/"item_store_model_performance_summary.csv")
    regs=["Smooth","Intermittent","Erratic","Lumpy"]
    models=["Ridge_All","Boosting_All","MLP_All","MI_MLP_Top10"]
    fig,ax=plt.subplots(figsize=(9,5))
    x=range(len(regs))
    for m in models:
        g=df[df["model"].eq(m)].set_index("demand_regime").reindex(regs)
        ax.plot(list(x),g["sMAPE_mean"],marker="o",label=m.replace("_All",""))
    ax.set_xticks(list(x),regs);ax.set_ylabel("Mean sMAPE (%)");ax.set_xlabel("Demand regime");ax.legend()
    ax.set_title("Regime-wise 28-day item-store robustness")
    save(fig,"figure_06_regime_robustness")

def main():
    ensure_dirs()
    fig1();fig2();fig3();fig4();fig5();fig6()
    print("Step 13 complete: figures written to figures/generated/.")

if __name__=="__main__":
    main()
