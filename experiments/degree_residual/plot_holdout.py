"""Compare prospective cell-level and population-level predictions on the same graphs."""
from pathlib import Path
import json
import os
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/ei-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'figures'))
from paper_style import E,I,JOINT,LIGHT,MUTED,apply_style,panel_label,save_figure


def main():
    apply_style()
    rows=json.loads((HERE/'holdout_scores.json').read_text())['networks']
    fig,(a,b)=plt.subplots(1,2,figsize=(7.8,3.65))
    fig.subplots_adjust(left=.105,right=.985,bottom=.24,top=.91,wspace=.32)
    for letter,ax in zip('ab',(a,b)):
        panel_label(ax,letter)
        ax.grid(axis='y',ls=':',lw=.65,color=LIGHT)
        ax.set_xticks([0,1,2])
        ax.set(xlabel=r'Degree dispersion $\chi$',xlim=(-.19,2.19))
    def points(axis, offset, marker, color, values, style, filled):
        x=np.arange(3)+offset
        vals=np.array(values)
        axis.scatter(x[:,None]+np.array([-.024,0,.024]),vals,s=8,color=color,alpha=.35,edgecolor='none')
        axis.errorbar(x,vals.mean(axis=1),yerr=vals.std(axis=1,ddof=1)/np.sqrt(3),marker=marker,
                      color=color,mfc=color if filled else 'white',ls=style,elinewidth=.8,
                      capsize=2,ms=4.5,lw=1.3)
    for typ,color,coffset in [(0,E,-.035),(1,I,.035)]:
        for name,offset,line,filled in [('shared',-.033,'--',False),('joint',.033,'-',True)]:
            vals=[[np.sqrt(r[name]['local'][typ]['gain_mse']) for r in rows if r['chi']==chi] for chi in (0,1,2)]
            points(a,coffset+offset,'o' if typ==0 else 's',color,vals,line,filled)
    a.set(ylabel='Single-cell mean gain RMSE')
    legend=[Line2D([],[],color=c,marker=m,ls='none',label=t,markersize=4.5)
            for c,m,t in [(E,'o','E cells'),(I,'s','I cells')]]
    legend.extend(Line2D([],[],color=JOINT,ls=style,label=label) for style,label in [('--','Shared variance'),('-','Cell-specific variance')])
    a.legend(handles=legend,ncol=2,loc='lower center',bbox_to_anchor=(.5,-.46),fontsize=7.5,columnspacing=.8)
    gaussian=[next(r['gaussian']['dimensions'][2] for r in rows if r['chi']==c) for c in (0,1,2)]
    b.plot([0,1,2],gaussian,'-',color=JOINT,label='Gaussian theory')
    for name,marker,color,offset,line,filled in [('observed','o',JOINT,-.075,'none',True),
                                                     ('shared','s',MUTED,0,'--',False),
                                                     ('joint','^',MUTED,.075,':',True)]:
        vals=[[(r if name=='observed' else r[name])['dimensions'][2] for r in rows if r['chi']==chi] for chi in (0,1,2)]
        points(b,offset,marker,color,vals,line,filled)
    b.set(yscale='log',ylabel='Joint dimension per neuron')
    b.set_yticks([.02,.04,.08,.16],labels=['0.02','0.04','0.08','0.16'])
    b.minorticks_off()
    handles=[Line2D([],[],color=JOINT,label='Gaussian theory'),Line2D([],[],color=JOINT,ls='none',marker='o',label='Measured',ms=4.5),
             Line2D([],[],color=MUTED,ls='--',marker='s',mfc='white',label='Shared variance',ms=4.5),
             Line2D([],[],color=MUTED,ls=':',marker='^',label='Cell-specific variance',ms=4.5)]
    b.legend(handles=handles,ncol=2,loc='lower center',bbox_to_anchor=(.5,-.46),fontsize=7.5,columnspacing=.8)
    fig.subplots_adjust(bottom=.32)
    save_figure(fig,HERE/'joint_variance',formats=('pdf','png'))


if __name__=='__main__':
    main()
