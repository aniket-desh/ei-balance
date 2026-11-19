"""Draw fixed-size readout, total E dimension, and pair-energy comparisons."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figures"))
from paper_style import E, I, JOINT, LIGHT, apply_style, panel_label, save_figure


def historical_groups():
    source = json.loads((HERE.parent/"dense_sweeps/summary.json").read_text())
    groups = sorted((g for g in source["groups"] if g["experiment"] == "composition" and g["N"] == 2000),
                    key=lambda g: g["control"])
    assert len(groups) == 9 and all(len(g["seeds"]) == 3 for g in groups)
    return groups


def estimate_points(axis, x, estimates, color, historical=False):
    """Fraction-method means have one SE across networks, with cohort fill encoding."""
    valid = [(f,e) for f,e in zip(x,estimates,strict=True) if e["mean"] is not None]
    if not valid:
        return
    for f, estimate in valid:
        axis.errorbar(f,estimate["mean"],yerr=estimate["se"],fmt="o" if historical else "s",
                      ms=3.9, capsize=1.8, mew=.95, elinewidth=.9, color=color,
                      mfc="white" if historical else color, zorder=4)


def make_figure(theory, old, window, cohort_label="Independent confirmation"):
    apply_style()
    fig, ((total,pair),(fixed,change)) = plt.subplots(2,2,figsize=(7.7,5.45))
    fig.subplots_adjust(left=.095,right=.965,bottom=.185,top=.955,wspace=.34,hspace=.48)
    curve = [r for r in theory["curve"] if .59 <= r["fraction_e"] <= .948]
    f = np.array([r["fraction_e"] for r in curve])
    groups = sorted(window["groups"],key=lambda g:g["fraction_e"])
    new_f = np.array([g["fraction_e"] for g in groups])
    np.testing.assert_allclose(new_f, [.80,.88,.94])
    old_f = np.array([g["control"] for g in old])
    # Dodge only the shared settings; all population colors at a setting use its same x.
    overlap = np.isin(old_f,new_f)
    old_x, new_x = old_f+.0025*overlap, new_f-.0025

    total.plot(f,[r["D_E_over_N"] for r in curve],color=E)
    estimate_points(total,old_x,[g["estimate"]["D_E_by_N"] for g in old],E,historical=True)
    estimate_points(total,new_x,[g["metrics"]["D_E_over_N"] for g in groups],E)
    fstar = theory["D_E_over_N_peak_fraction"]
    top = max(r["D_E_over_N"] for r in curve)
    total.plot(fstar,top,marker="D",ms=4.5,mfc="white",mec=E,ls="none",zorder=3)
    total.annotate(rf"$f_E^*={fstar:.3f}$",xy=(fstar,top),xytext=(0,11),textcoords="offset points",
                   fontsize=7,color=E,ha="center")
    total.margins(y=.23)
    total.set_ylabel(r"Total E dimension / $N$,  $D_E/N$")

    for k, (color,style,name) in enumerate(((E,"-","E"),(I,"--","I"))):
        pair.plot(f,[r["pair_energy"][k] for r in curve],color=color,ls=style)
        key = "P_"+name*2
        estimate_points(pair,old_x,[g["estimate"][key] for g in old],color,historical=True)
        estimate_points(pair,new_x,[g["metrics"][key] for g in groups],color)
        fixed.plot(f,[r["readout_PR"][k] for r in curve],color=color,ls=style)
        estimate_points(fixed,new_f,[g["metrics"][name+"_readout"] for g in groups],color)

        delta = window["contrasts"][name+"_readout_80_88"]
        y = delta["mean"]
        if y is not None:
            lo,hi = delta["ci95"] if name=="E" else delta["ci_family"]
            change.errorbar(k+.06,y,yerr=[[y-lo],[hi-y]],fmt="s",ms=4.4,capsize=3,
                            elinewidth=1.2, color=color,zorder=4)
        change.plot(k-.07,delta["theory"],"D",mfc="white",mec=color,ms=5,zorder=3)
    pair.set_ylabel(r"Distinct-pair energy,  $P_{aa}$")
    fixed.set_ylabel("100-cell readout PR")
    change.set_ylabel("Change in 100-cell readout PR")
    change.set_xlabel(r"Readout population  ($f_E$: 0.80 to 0.88)")
    change.set(xlim=(-.5,1.5),xticks=[0,1],xticklabels=["E","I"])
    change.axhline(0,color=JOINT,linewidth=.8,zorder=1)
    change.margins(y=.22)

    for ax in (total,pair,fixed):
        ax.set(xlim=(.59,.951),xlabel=r"Excitatory fraction $f_E$",xticks=[.6,.7,.8,.88,.94],
               xticklabels=["0.60","0.70","0.80","0.88","0.94"])
    for tag,ax in zip("abcd",(total,pair,fixed,change)):
        panel_label(ax,tag)
        ax.grid(axis="y",color=LIGHT,linewidth=.45)
    handles = [
        Line2D([],[],color=E,label="E theory"),
        Line2D([],[],color=I,ls="--",label="I theory"),
        Line2D([],[],marker="o",ls="none",mfc="white",mec=JOINT,ms=4.2,label="Nine-fraction sweep"),
        Line2D([],[],marker="s",ls="none",color=JOINT,ms=4.2,label=cohort_label)]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.53,.015),ncol=4,
               columnspacing=1.45,handletextpad=.42)
    return fig


def main():
    source = json.loads((HERE/"summary.json").read_text())
    theory = json.loads((HERE.parent/"readout_theory/predictions.json").read_text())
    fig = make_figure(theory,historical_groups(),source["windows"]["T8000_gap0"])
    save_figure(fig,HERE/"composition_readout",formats=("pdf","png"))


if __name__ == "__main__":
    main()
