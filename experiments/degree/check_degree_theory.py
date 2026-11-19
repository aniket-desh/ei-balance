"""Exact finite-pool moment identities and deterministic quadrature refinement."""
from pathlib import Path
from itertools import product
import sys,json,time,math
import numpy as np
from scipy.stats import binom,betabinom

sys.path.insert(0,str(Path(__file__).resolve().parent))
from predict_degree import predict,chi_effective,OUT


def count_pmf(m,p,chi):
    k=np.arange(m+1)
    if chi<=1:
        trials=round(chi*m)
        base=(m-trials)*p
        integer=int(np.floor(base));fraction=base-integer
        pmf=(1-fraction)*binom.pmf(k-integer,trials,p)
        pmf+=fraction*binom.pmf(k-integer-1,trials,p)
    else:
        concentration=(m-chi)/(chi-1)
        pmf=betabinom.pmf(k,m,p*concentration,(1-p)*concentration)
    return pmf


def finite_pool_check(m=8,p=.3,chi=0.):
    pmf=count_pmf(m,p,chi)
    degrees=np.arange(m+1)
    mean=degrees@pmf
    var=(degrees-mean)**2@pmf
    ce=var/(m*p*(1-p))
    masks=np.array(list(product([0.,1.],repeat=m)))
    prob=np.array([pmf[int(a.sum())]/math.comb(m,int(a.sum())) for a in masks])
    neuron_means=np.linspace(.1,1.4,m)**1.3
    centered=masks-p
    cov=(centered.T*prob)@centered
    expected=p*(1-p)*((m-ce)/(m-1)*np.eye(m)+(ce-1)/(m-1)*np.ones((m,m)))
    measured=(centered@neuron_means)**2@prob
    predicted=m*p*(1-p)*((m-ce)/(m-1)*np.var(neuron_means)+ce*np.mean(neuron_means)**2)
    return {'M':m,'p':p,'chi_requested':chi,'chi_effective':float(ce),
            'probability_sum_error':float(prob.sum()-1),
            'mean_count_error':float(mean-m*p),
            'edge_covariance_max_error':float(np.max(np.abs(cov-expected))),
            'input_variance_measured':float(measured),
            'input_variance_predicted':float(predicted),
            'input_variance_relative_error':float(measured/predicted-1)}


def main():
    checks=[finite_pool_check(chi=c) for c in [0.,.15,.5,1.,2.,5.]]
    assert max(x['edge_covariance_max_error'] for x in checks)<1e-13
    baseline=json.loads((OUT/'predictions.json').read_text())
    refinement=[]
    for chi in [0.,.5,1.,2.]:
        tic=time.monotonic()
        row,_=predict(chi,dt=.04,nh=160,horizon=150.,cutoff=20.)
        old=next(r for r in baseline['rows'] if r['chi']==chi)
        refinement.append({'chi':chi,'refined':row,
                           'dimension_relative_difference':float(row['prediction']['dimension_fraction']/old['prediction']['dimension_fraction']-1),
                           'energy_relative_difference':float(row['prediction']['covariance_energy']/old['prediction']['covariance_energy']-1),
                           'block_relative_differences':(np.array(row['block_energies'])/np.array(old['block_energies'])-1).tolist(),
                           'elapsed_seconds':time.monotonic()-tic})
        print(json.dumps({'chi':chi,'d_refined':row['prediction']['dimension_fraction'],
                          'relative_difference':refinement[-1]['dimension_relative_difference']}),flush=True)
    out={'finite_pool_exact_enumeration':checks,'refinement':refinement,
         'status':'Exact finite-pool identities pass; refinements compare deterministic conditional closure only.'}
    (OUT/'theory_checks.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
