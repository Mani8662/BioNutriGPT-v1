"""Draw figures from current metadata and saved measured results only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE = Path(__file__).resolve().parent / "Submission_Pack"
DATA = json.loads((BASE / "Sources/submission_data.json").read_text(encoding="utf-8"))
OUT = BASE / "Figures"
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11, "svg.fonttype":"none", "axes.spines.top":False,"axes.spines.right":False})


def save(fig, name):
    for ext in ("png","svg","pdf"):
        fig.savefig(OUT / f"{name}.{ext}",dpi=300,facecolor="white",bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, fill="#eef4f7"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.015",linewidth=1,edgecolor="#647b8a",facecolor=fill))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",linespacing=1.5)


def main():
    OUT.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,7)); ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    stages=[("Stage 1   Screening","87 features · tabular Transformer\nPrevalent outcomes in the NHANES setting"),
            ("Stage 2   Planning rules","Versioned nutrient bounds\n30 active fields · 9 reporting or exposure fields"),
            ("Stage 3   Meal optimization","Whole standardized portions\nIndependent validation before acceptance"),
            ("Stage 4   Controlled plan assistant","Grounded answers · checked revisions\nPersistent preferences · version history")]
    for i,(title,body) in enumerate(stages):
        y=.77-i*.205
        box(ax,.22,y,.60,.15,title+"\n"+body)
        if i<3: ax.annotate("",xy=(.52,y-.055),xytext=(.52,y),arrowprops={"arrowstyle":"->","color":"#425e6c","lw":1.5})
    ax.annotate("",xy=(.85,.43),xytext=(.85,.23),arrowprops={"arrowstyle":"->","connectionstyle":"arc3,rad=-.9","lw":1.5,"color":"#28796d"})
    ax.text(.945,.335,"Revise\nand check",ha="center",va="center",fontsize=10,color="#205c53")
    ax.text(.52,.985,"BioNutriGPT workflow",ha="center",va="top",fontsize=17,weight="bold")
    ax.text(.52,.03,"Observed patient inputs remain separate from proposed and accepted meals.",ha="center",fontsize=10)
    save(fig,"Figure_1_Architecture")

    fig,ax=plt.subplots(figsize=(8,4.8))
    names=["Diabetes","Obesity","Cardiovascular disease"]
    vals=[DATA["stage1"]["metrics"][k]["auroc"] for k in ("diabetes","obesity","cvd")]
    bars=ax.bar(names,vals,color=["#3d728c","#5b8b82","#8a7399"],width=.55)
    ax.set(ylim=(0,1),ylabel="Unweighted AUROC")
    ax.set_title("Saved temporal holdout discrimination",loc="left",fontsize=15,pad=18)
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+.025,f"{v:.4f}",ha="center")
    ax.axhline(.5,color="#84929b",linestyle="--",linewidth=.8)
    ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=.23)
    fig.text(.12,.05,"NHANES 2017–March 2020 · N = 7,046\nSaved checkpoint evaluation; no new inference, calibration or superiority claim.",fontsize=9,color="#444444")
    save(fig,"Figure_2_Stage1_Saved_Discrimination")

    fig,axs=plt.subplots(2,1,figsize=(9,5.2),gridspec_kw={"hspace":1.0})
    counts=[30,7,2]; colors=["#427f91","#b7c4ca","#d7bc85"]
    left=0
    for n,col,label in zip(counts,colors,["Active planning","Report only","Exposure"]):
        axs[0].barh([0],[n],left=left,color=col,label=f"{label} ({n})",height=.5);left+=n
    axs[0].set(xlim=(0,39),yticks=[],xlabel="Fields in the 39-field schema")
    axs[0].legend(loc="upper left",bbox_to_anchor=(0,1.55),ncol=3,frameon=False,fontsize=9)
    cat=DATA["snapshot"]["catalog"];good=cat["complete_active_records_before_diet_or_preference_filters"];total=cat["count"]
    axs[1].barh([0],[good],color="#427f91",height=.5,label=f"No declared active gaps ({good})")
    axs[1].barh([0],[total-good],left=good,color="#bd8a76",height=.5,label=f"Active gaps ({total-good})")
    axs[1].set(xlim=(0,total),yticks=[],xlabel="Catalog records before dietary and preference filtering")
    axs[1].legend(loc="upper left",bbox_to_anchor=(0,1.55),ncol=2,frameon=False,fontsize=9)
    for ax in axs:
        ax.spines["left"].set_visible(False);ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
    fig.suptitle("Current rule roles and catalog completeness",x=.125,ha="left",fontsize=15)
    fig.subplots_adjust(top=.76,bottom=.22)
    fig.text(.125,.04,"Catalog v5 · 49 records · 20 INDB variants + 29 USDA proxies\nCompleteness is not laboratory validation, clinical adequacy or an allergen-free guarantee.",fontsize=9,color="#444444")
    save(fig,"Figure_3_Current_Catalog_Coverage")

    fig,ax=plt.subplots(figsize=(12,3.5));ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
    labels=["Screen\nPrevalent outcomes","Set rules\nVersioned bounds","Optimize\nWhole portions","Discuss and revise\nVerified plan versions"]
    for i,label in enumerate(labels):
        x=.025+i*.25;box(ax,x,.35,.20,.38,label)
        if i<3:ax.annotate("",xy=(x+.245,.54),xytext=(x+.205,.54),arrowprops={"arrowstyle":"->","lw":1.3,"color":"#425e6c"})
    ax.text(.5,.95,"Auditable conversational meal planning",ha="center",fontsize=17,weight="bold")
    ax.text(.5,.12,"Food provenance + persistent preferences + independent validation",ha="center",fontsize=12,color="#245b66")
    ax.text(.5,-.02,"Computational research prototype · clinical benefit and Indian screening validity not established",ha="center",fontsize=9,color="#444444")
    save(fig,"Graphical_Abstract_Draft")
    v4=DATA['stage4_v4'];a=v4['statistics']['ablation']['same_input_state_pairs']
    fig,axs=plt.subplots(1,2,figsize=(10,4.8))
    bars=axs[0].bar(['Guarded','Qwen 3B'],[100,100*2/24],color=['#3d728c','#a7aeb5'],width=.55)
    axs[0].set(ylim=(0,110),ylabel='Protocol passes (%)',title='Matched 12 participants')
    for b,label in zip(bars,['24/24','2/24']):axs[0].text(b.get_x()+b.get_width()/2,b.get_height()+3,label,ha='center')
    bars=axs[1].bar(['With penalty','Without penalty'],[a['full_mean'],a['no_penalty_mean']],color=['#3d728c','#a7aeb5'],width=.55)
    axs[1].set(ylim=(0,20),ylabel='Mean serving change',title='25 same input state successful pairs')
    for b in bars:axs[1].text(b.get_x()+b.get_width()/2,b.get_height()+.5,f'{b.get_height():.2f}',ha='center')
    for ax in axs:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=.25,wspace=.4,top=.83)
    fig.suptitle('Verified v4 conversational benchmark',fontsize=15,x=.125,ha='left')
    fig.text(.125,.05,'Selected scripted LASI profiles; protocol passes include justified refusals.\nAblation is conditional on paired successful revisions. Clinical benefit was not measured.',fontsize=9)
    save(fig,'Figure_4_Verified_Stage4_Comparison')
    print("Created four evidence-based figures and an optional graphical abstract in PNG, SVG and PDF.")


if __name__ == "__main__":
    main()
