"""Draw the held-out particle validation without modifying any main figure."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR",str(Path(os.getenv("TMPDIR","/tmp"))/"ei-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"figures"))
from paper_style import E, I, CROSS, JOINT, MUTED, apply_style, panel_label, save_figure


def main():
    d = json.loads((HERE/"validation.json").read_text())
    p400 = [json.loads(p.read_text()) for p in sorted((HERE/"runs").glob("*_P400_*.json"))]
    apply_style()
    fig,axes = plt.subplots(1,3,figsize=(8.05,2.88))
    fig.subplots_adjust(left=.08,right=.985,top=.865,bottom=.21,wspace=.46)
    a,b,c = axes
    for ax,letter in zip(axes,"ABC"):
        panel_label(ax,letter)
    colors = (E,I)
    shapes = ("o","s")
    styles = ("-","--")
    a.axhline(0,color=MUTED,lw=.85,ls=":")
    q = d["quantile_levels"]
    ref = np.array(d["by_chi"]["0"]["gaussian"]["mu_quantiles"])
    for i in range(2):
        particle = d["by_chi"]["0"]["particle"]["400"]["mu_quantiles"]["mean"][i]
        a.plot(q,particle-ref[i],styles[i],color=colors[i],lw=1.6)
        for held in [h for h in d["held_out"] if h["chi"]==0]:
            a.scatter(q,np.array(held["mu_quantiles"][i])-ref[i],marker=shapes[i],s=14,
                      edgecolors=colors[i],facecolors="none",linewidths=.8)
    a.set(xlabel="Quantile probability",ylabel=r"Input quantile $-h_{\rm Gaussian}$",
          xticks=[.1,.5,.9],xlim=(.04,.96))
    handles = [Line2D([],[],color=color,marker=shape,ls=style,mfc="white",ms=3.5,label=name)
               for name,color,shape,style in zip(("E","I"),colors,shapes,styles)]
    a.legend(handles=handles,loc="upper left",handlelength=1.4,borderpad=0)

    chi = np.arange(3)
    gaussian = [d["by_chi"][str(k)]["gaussian"]["dimensions"][2] for k in chi]
    b.plot(chi,gaussian,color=JOINT,lw=1.15,ls=":",label="Gaussian")
    for ch in chi:
        pred = [r["dimensions"][2] for r in p400 if r["chi"]==ch and r["pr_admitted"]]
        measured = [r["dimensions"][2] for r in d["held_out"] if r["chi"]==ch]
        b.scatter(ch-.07+np.linspace(-.025,.025,len(measured)),measured,
                  color=JOINT,s=15,marker="o",label="Network" if ch==0 else None)
        b.scatter(ch+.07+np.linspace(-.025,.025,len(pred)),pred,facecolors="none",
                  edgecolors=CROSS,s=20,marker="D",label="Particle" if ch==0 else None)
    b.set(xlabel=r"Degree dispersion $\chi$",ylabel=r"Joint dimension $D/N$",xticks=chi,
          yscale="log",xlim=(-.24,2.22),ylim=(.009,.28))
    b.yaxis.set_major_locator(FixedLocator([.01,.03,.1,.2]))
    b.yaxis.set_major_formatter(ScalarFormatter())
    b.legend(loc="upper right",handlelength=1.25,ncol=1,borderpad=0,labelspacing=.22,fontsize=7)

    c.axhline(1,color=MUTED,lw=.85,ls=":")
    for i in range(2):
        means = []
        for ch in chi:
            values = [r["conditional_gain_rmse_with_cell_q"][i]/r["conditional_gain_rmse_with_shared_q"][i]
                      for r in d["held_out"] if r["chi"]==ch]
            x = ch+(-.04 if i==0 else .04)
            c.scatter(np.full(3,x),values,color=colors[i],marker=shapes[i],s=12,alpha=.45)
            means.append(np.mean(values))
        c.plot(chi+(-.04 if i==0 else .04),means,styles[i],lw=1.6,color=colors[i])
    c.set(xlabel=r"Degree dispersion $\chi$",ylabel="Gain RMSE, cell / shared",xticks=chi,
          xlim=(-.2,2.2),ylim=(.42,1.07))
    save_figure(fig,HERE/"particle_validation",formats=("pdf","png"))


if __name__=="__main__":
    main()
