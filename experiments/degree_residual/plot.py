"""Draw the degree residuals using only the saved numerical summaries."""
import json
import os
from pathlib import Path
import sys
os.environ.setdefault("MPLCONFIGDIR","/tmp/ei-matplotlib")
import matplotlib
matplotlib.use('Agg')
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'figures'))
from paper_style import E,I,CROSS,JOINT,MUTED,LIGHT,apply_style,panel_label,save_figure,SIZE_MARKERS
from sweep_records import draw_group,point_offsets


def main():
    apply_style()
    result = json.loads((HERE/'residuals.json').read_text())
    theory = json.loads((HERE.parent/'degree/theory_curve.json').read_text())['rows']
    groups = result['means_and_network_se']
    measured = result['experiments']
    focus = sorted([g for g in groups if g['N']==2000 and g['cohort']=='dense'],key=lambda r:r['chi'])
    x = np.sqrt([r['chi'] for r in focus])
    fig,((a,b),(c,d)) = plt.subplots(2,2,figsize=(8.3,6.0))
    fig.subplots_adjust(left=.092,right=.988,bottom=.105,top=.955,wspace=.30,hspace=.47)
    for letter,ax in zip('abcd',(a,b,c,d)):
        panel_label(ax,letter)
        ax.grid(axis='y',ls=':',lw=.6,color=LIGHT)
    a.plot(np.sqrt([r['chi'] for r in theory]),[r['prediction']['dimension_fraction'] for r in theory],color=JOINT,label='Theory')
    keys = [(np.sqrt(r['chi']),r['N'],r['cohort']) for r in groups]
    offsets = point_offsets(keys,.025)
    for g,key in zip(groups,keys):
        rows = [r for r in measured if all(g[k]==r[k] for k in ('cohort','chi','N'))]
        draw_group(a,key[0]+offsets[key],[r['d'] for r in rows],SIZE_MARKERS[g['N']],MUTED,g['cohort'],jitter=.002)
    a.set(yscale='log',ylabel='Joint dimension per neuron')
    ticks = [.02,.05,.10,.20]
    a.set_yticks(ticks,labels=['0.02','0.05','0.10','0.20'])
    a.minorticks_off()
    handles = [Line2D([],[],color=JOINT,label='Theory')]
    handles += [Line2D([],[],color=MUTED,marker=m,ls='none',ms=4,label=f'{n:,}') for n,m in SIZE_MARKERS.items()]
    a.legend(handles=handles,ncol=2,loc='upper right',columnspacing=.8,handlelength=1.5,fontsize=7)
    for ax in (a,b,c):
        tick = [0,.05,.2,.5,1,1.5,2]
        ax.set_xticks(np.sqrt(tick),[f'{k:g}' for k in tick])
        ax.set(xlabel=r'Degree dispersion $\chi$',xlim=(-.055,np.sqrt(2)+.045))
    b.axhline(0,color=MUTED,lw=.7)
    def series(ax,key,color,style,label):
        m,s = np.array([r['primary'][key] for r in focus]).T
        ax.errorbar(x,m,yerr=s,ls=style,color=color,marker='.',ms=3.8,lw=1.4,elinewidth=.7,capsize=1.4,label=label)
    series(b,'log_d_error',JOINT,'-','Dimension residual')
    series(b,'rate_numerator',MUTED,'--','Rate-variance term')
    # Reconstruct individual total energy contributions before taking the SE.
    energy = [[r['primary']['log_d_error']-r['primary']['rate_numerator'] for r in measured
               if r['cohort']=='dense' and r['N']==2000 and r['chi']==g['chi']] for g in focus]
    b.errorbar(x,np.mean(energy,axis=1),yerr=np.std(energy,axis=1,ddof=1)/np.sqrt(3),
               color=CROSS,ls=':',marker='.',ms=3.8,lw=1.4,elinewidth=.7,capsize=1.4,label='Covariance-energy term')
    b.set(ylabel='Contribution to log PR error')
    b.legend(loc='upper left',fontsize=7)
    c.axhline(0,color=MUTED,lw=.7)
    series(c,'ee_off',E,'-','E–E distinct pairs')
    series(c,'ei_both',CROSS,'--','2 × E–I')
    series(c,'ii',I,'-.','I–I')
    series(c,'ee_diag',E,':','E diagonal')
    c.legend(loc='upper left',fontsize=7)
    c.set(ylabel='Energy contribution to log PR error')
    holdout=json.loads((HERE/'holdout_scores.json').read_text())['networks']
    spec=[('original',1000),('original',2000),('dense',2000),('original',4000)]
    high=[next(g for g in groups if g['cohort']==cohort and g['N']==n and g['chi']==2) for cohort,n in spec]
    fresh=[r['secondary_Gaussian_degree_residual'] for r in holdout if r['chi']==2]
    high.insert(3,{'primary':{metric:[np.mean([r[metric] for r in fresh]),
                                         np.std([r[metric] for r in fresh],ddof=1)/np.sqrt(3)] for metric in fresh[0]}})
    d.axhline(0,color=MUTED,lw=.7)
    for i,(name,color,marker,label) in enumerate([('log_d_error',JOINT,'o','Log PR error'),('rate_numerator',MUTED,'s','Rate-variance term'),('ee',E,'^','E–E term')]):
        m,se=np.array([g['primary'][name] for g in high]).T
        d.errorbar(np.arange(5)+.14*(i-1),m,yerr=se,ls='none',marker=marker,ms=4.3,
                   elinewidth=1,capsize=2,color=color,label=label)
    d.set(xticks=range(5),xticklabels=['1,000\noriginal','2,000\noriginal','2,000\ndense','2,000\nheld out','4,000\noriginal'],
          ylabel=r'Contribution at $\chi=2$',xlabel='Network size and cohort',xlim=(-.45,4.45))
    d.legend(loc='upper right',fontsize=7,handlelength=1.)
    save_figure(fig,HERE/'degree_residual',formats=('pdf','png'))


if __name__=='__main__':
    main()
