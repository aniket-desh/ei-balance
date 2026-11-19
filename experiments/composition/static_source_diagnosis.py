"""Exact fixed-network static source accounting for selected retained records.

This is a diagnostic identity, not a parameter-only closure. Dependence of the
measured rate means on J is retained. Row tiles keep extra memory modest.
"""
from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'recovered/bundle/ei_full/ei_balance_handoff/src'))
from eicavity.network import make_network,NetworkConfig


def analyze(runpath):
    meta=json.loads(runpath.read_text());name=meta['name'];p=HERE/f'{name}_block_local.npz'
    if meta['status']!='complete':return
    c=NetworkConfig(**meta['job']['config'])
    if p.exists():
        local=np.load(p);m=local['mean_rate'].mean(0);mu=local['mean_x'].mean(0);source=str(p.relative_to(ROOT))
    else:
        selected=next((e.get('neuron_statistics_file')for e in meta.get('estimators',[])if e['block_duration']==500.),None)
        if selected is None:return
        local=np.load(ROOT/selected);m=local['mean_rate'];mu=local['mean_x'];source=selected
    J,drive,x,labels,info=make_network(c,np.array(meta['job']['mean_input']));j0=np.array(c.mean_connectivity);f=np.bincount(labels,minlength=2)/c.n;G=np.array(c.variance_profile);W=G*f[None,:]
    m2=np.array([np.mean(m[labels==a]**2)for a in [0,1]]);mbar=np.array([m[labels==a].mean()for a in [0,1]])
    target=W@m2;eta=[];directdiag=[];allmu_residual=[];rows=[]
    for a in [0,1]:
        idx=np.flatnonzero(labels==a);ea=[];da=[];res=[]
        for first in range(0,len(idx),64):
            ii=idx[first:first+64];dJ=J[ii]-j0[a,labels][None,:]/np.sqrt(c.n)
            e=dJ@m;di=np.square(dJ)@(m*m)
            ea.extend(e.tolist());da.extend(di.tolist())
            res.extend((mu[ii]-(J[ii]@m+drive[ii])).tolist())
        ea=np.array(ea);da=np.array(da);res=np.array(res)
        diag=float(da.mean());off=float(np.mean(ea*ea)-diag);mean=float(ea.mean());var=float(ea.var())
        rows.append({'population':a,'measured_mean_rate':float(mbar[a]),'measured_source_mean_rate_squared_by_presynaptic_population':m2.tolist(),'leading_GFm2_using_measured_rates':float(target[a]),'exact_diagonal_source':diag,'exact_offdiagonal_source':off,'mean_eta':mean,'mean_eta_squared':mean*mean,'exact_centered_static_variance_from_Jm':var,'measured_input_mean_static_variance':float(mu[labels==a].var()),'offdiagonal_minus_centering':off-mean*mean,'diagonal_minus_GFm2':diag-target[a],'state_mean_minus_Jrate_mean_plus_drive_rms':float(np.sqrt(np.mean(res*res))),'identity_residual':var-(diag+off-mean*mean)})
    out={'neuron_moment_source':source,'name':name,'scope':'Exactfixed-networkdecompositionatmeasuredratemeans; noindependenceassumption. Source-to-labeldependenceandquenchedcross-termsareincluded, notpredicted. Sampled-stateaveragesneednotexactlyobeycontinuousmeanidentityduetofiniterecordandmidpoint/samplingdiscretization.','config':vars(c),'rows':rows}
    (HERE/f'{name}_static_source.json').write_text(json.dumps(out,indent=2)+'\n');return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path,nargs='+');a=p.parse_args()
    for path in a.run:
        r=analyze(path)
        if r:print(json.dumps(r,indent=2),flush=True)
