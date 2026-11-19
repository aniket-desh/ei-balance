"""Independent block-moment diagnostics for retained prespecified trajectories.

No pair covariance enters these Gaussian-law diagnostic substitutions. The
script uses one BLAS thread and one block at a time. Raw arrays remain in
runtime/tmp; compact per-neuron block moments are saved for reproducibility.
"""
from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse,json,sys,itertools
import numpy as np
from scipy.special import ndtr

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'recovered/bundle/ei_full/ei_balance_handoff/src'))
from eicavity.single_site import centered_moment, raw_moment


def analyze(runpath):
    meta=json.loads(runpath.read_text());c=meta['job']['config'];name=meta['name']
    if meta['status']!='complete':return
    raw=ROOT/'next_round/runtime/tmp'/f'{name}_trajectory.npy'
    saved=HERE/f'{name}_block_local.npz'
    N=c['n'];ne=round(N*c['fraction_e']);groups=[slice(0,ne),slice(ne,N)]
    B=500.;ns=round(B/c['sample_dt'])
    if saved.exists():
        a=np.load(saved);xb=a['mean_x'];x2b=a['mean_x_squared'];rb=a['mean_rate'];r2b=a['mean_rate_squared'];gb=a['mean_gain'];nb=len(xb)
    elif any(e.get('neuron_statistics_file') for e in meta.get('estimators',[])if e['block_duration']==500.):
        selected=next(e['neuron_statistics_file']for e in meta['estimators']if e['block_duration']==500.)
        a=np.load(ROOT/selected);xb=a['block_mean_x'];x2b=a['block_q_x']+xb*xb;rb=a['block_mean_rate'];r2b=a['block_variance_rate']+rb*rb;gb=a['block_mean_gain'];nb=len(xb)
    elif raw.exists():
        x=np.load(raw,mmap_mode='r');nb=len(x)//ns
        xb=[];x2b=[];rb=[];r2b=[];gb=[]
        for k in range(nb):
            z=np.asarray(x[k*ns:(k+1)*ns],float);r=2*ndtr(z);g=np.sqrt(2/np.pi)*np.exp(-.5*z*z)
            xb.append(z.mean(0));x2b.append(np.square(z).mean(0));rb.append(r.mean(0));r2b.append(np.square(r).mean(0));gb.append(g.mean(0))
        xb,x2b,rb,r2b,gb=map(np.array,[xb,x2b,rb,r2b,gb])
        np.savez_compressed(saved,mean_x=xb,mean_x_squared=x2b,mean_rate=rb,mean_rate_squared=r2b,mean_gain=gb,block_duration=B,sample_dt=c['sample_dt'])
    else:return
    mu=xb.mean(0);q=x2b.mean(0)-mu*mu;m=rb.mean(0);v=r2b.mean(0)-m*m;g=gb.mean(0)
    result={'name':name,'config':c,'scope':'Prespecified selected trajectories; cross-block estimates exact only for stationary independent blocks. Gaussianized substitutions are calibrated local diagnostics, not new predictions.','block_duration':B,'duration':nb*B,'groups':[]}
    W=np.asarray(c['variance_profile'])*np.array([ne/N,1-ne/N])[None,:]
    BB=np.asarray(c['mean_connectivity'])*np.array([ne/N,1-ne/N])[None,:]
    meanrate=np.array([m[a].mean()for a in groups]);hh=np.array([mu[a].mean()for a in groups])
    residual=hh-np.array(meta['job']['mean_input'])-np.sqrt(N)*BB@(meanrate-.3)
    result['coherent_balance_inferred_centered_input_mean']=residual.tolist()
    result['coherent_balance_caveat']='Inferred using the mean dynamical equation; includes finite-record endpoint derivative. It is a diagnostic, not an independent validation of that identity.'
    pairs=list(itertools.combinations(range(nb),2)); pp=np.array(pairs); index={z:i for i,z in enumerate(pairs)}
    vp=.5*((r2b-rb*rb)[pp[:,0]]+(r2b-rb*rb)[pp[:,1]])+.5*np.square(rb[pp[:,0]]-rb[pp[:,1]])
    qmatch=[]
    for aa,bb,cc,dd in itertools.combinations(range(nb),4):
        qmatch.extend([(index[(aa,bb)],index[(cc,dd)]),(index[(aa,cc)],index[(bb,dd)]),(index[(aa,dd)],index[(bb,cc)])])
    qp,qqp=np.array(qmatch).T
    diagenergy=[]
    for ix in groups:
        gram=vp[:,ix]@vp[:,ix].T/N;diagenergy.append(float(gram[qp,qqp].mean()))
    est=next(e for e in meta['estimators']if e['block_duration']==500.); primary=next(g for pre in est['prefixes']if pre['duration']==nb*B for g in pre['gaps']if g['empty_blocks']==0)
    ed=sum(diagenergy);nv=primary['nu'];etot=primary['energy']
    result['diagonal_energy_four_window']={'by_population_total_N_normalization':diagenergy,'total':ed,'d_uncorrelated':nv*nv/ed,'correlation_energy_penalty_factor':etot/ed,'d_reconstructed':nv*nv/ed/(etot/ed),'primary_d':primary['d'],'scope':'Samequadruplepairingsasprimarygap0; ratiosremainfinite-recordestimates.'}
    for pop,ix in enumerate(groups):
        D=mu[ix].var();qq=q[ix].mean();h=mu[ix].mean();S=1+qq
        A_gauss_pop=2/(np.pi*np.sqrt(S*(S+2*D)))*np.exp(-h*h/(S+2*D))
        A_gauss_ind=np.mean(2/(np.pi*(1+q[ix]))*np.exp(-mu[ix]**2/(1+q[ix])))
        A_gauss_common=np.mean(2/(np.pi*S)*np.exp(-mu[ix]**2/S))
        vn_gauss_ind=centered_moment(mu[ix],q[ix],q[ix]).mean()
        m2_measured=np.mean(m[ix]**2); m2_gauss_ind=np.mean((2*ndtr(mu[ix]/np.sqrt(1+q[ix])))**2); m2_gauss_common=np.mean((2*ndtr(mu[ix]/np.sqrt(S)))**2); m2_gauss_pop=raw_moment(h,D+qq,D)
        row={'mean_rate_squared_measured':float(m2_measured),'mean_rate_squared_Gaussian_individual_mu_q':float(m2_gauss_ind),'mean_rate_squared_Gaussian_individual_mu_common_q':float(m2_gauss_common),'mean_rate_squared_Gaussian_population_h_Delta_q':float(m2_gauss_pop),'static_mu_standardized_skewness':float(np.mean((mu[ix]-h)**3)/D**1.5),'static_mu_excess_kurtosis':float(np.mean((mu[ix]-h)**4)/D**2-3),'population':pop,'h':float(h),'Delta_own_record':float(D),'q_own_record':float(qq),'total_input_variance':float(D+qq),'A_full_record':float(np.square(g[ix]).mean()),'A_Gaussian_individual_mu_q':float(A_gauss_ind),'A_Gaussian_individual_mu_common_q':float(A_gauss_common),'A_Gaussian_population_h_Delta_q':float(A_gauss_pop),'nu_measured_own_record':float(v[ix].mean()),'nu_Gaussian_individual_mu_q':float(vn_gauss_ind),'cross_block':[]}
        for gap in [0,1,2]:
            ii,jj=np.triu_indices(nb,k=gap+1)
            xx=np.mean(xb[ii,ix]*xb[jj,ix],axis=0)
            mm=np.mean(rb[ii,ix]*rb[jj,ix],axis=0)
            As=np.mean(gb[ii,ix]*gb[jj,ix])
            Dcross=xx.mean()-np.mean(xb[ii,ix].mean(1)*xb[jj,ix].mean(1))
            qc=x2b[:,ix].mean()-xx.mean();vc=r2b[:,ix].mean()-mm.mean()
            row['cross_block'].append({'empty_blocks':gap,'pairs':len(ii),'A_cross':float(As),'Delta_cross':float(Dcross),'q_cross':float(qc),'nu_cross':float(vc)})
        result['groups'].append(row)
    dest=HERE/f'{name}_local_addon.json';dest.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path,nargs='+');a=p.parse_args()
    for path in a.run:
        r=analyze(path)
        if r:print(json.dumps({'name':r['name'],'groups':r['groups']},indent=2),flush=True)
