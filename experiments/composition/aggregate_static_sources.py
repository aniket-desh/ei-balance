"""Keep mean static-source shifts distinct from connectivity-to-connectivity spread."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import t
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=json.loads((ROOT/'recovered/bundle/diagnosis/theory/summary.json').read_text())


def stats(x):
    x=np.asarray(x,float);n=len(x);m=x.mean(0)
    if n>1:
        sd=x.std(0,ddof=1);se=sd/np.sqrt(n);q=t.ppf(.975,n-1)*se;ci=np.stack([m-q,m+q],axis=-1).tolist()
        return dict(n=n,mean=m.tolist(),sd=sd.tolist(),se=se.tolist(),ci95_t=ci,covariance=np.cov(x,rowvar=False,ddof=1).tolist())
    return dict(n=n,mean=m.tolist(),sd=None,se=None,ci95_t=None,covariance=None)


def main():
    rows=[]
    for p in sorted(HERE.glob('*_static_source.json')):
        d=json.loads(p.read_text());c=d['config']
        for z in d['rows']:
            a=z['population'];D0=BASE['static_variances'][a]
            v=[z['leading_GFm2_using_measured_rates']-D0,z['diagonal_minus_GFm2'],z['offdiagonal_minus_centering'],z['exact_centered_static_variance_from_Jm']-D0]
            np.testing.assert_allclose(sum(v[:3]),v[3],rtol=1e-12,atol=1e-12)
            rows.append(dict(name=d['name'],N=c['n'],fraction_e=c['fraction_e'],seed=c['seed'],population=a,components=v,mean_state_static_variance_minus_Jm_static_variance=z['measured_input_mean_static_variance']-z['exact_centered_static_variance_from_Jm'],mean_equation_rms=z['state_mean_minus_Jrate_mean_plus_drive_rms']))
    out={'scope':'Empiricalsamplemeansandnetworkspreadofexactstatic-sourceidentities, notpredictedensemblecorrection. Onlynetworksmomentfilesavailableincluded: prespecifiedseed3001initialpointselections+allv2extensions. Thisisnotfull40networkcohort.','component_order':['presynaptic_mean_rate_distribution_shift','realized_squared_coupling_weighting_shift','centered_cross_source_term','sum_static_variance_shift'],'groups':{},'per_network':rows}
    for N,fe,a in sorted({(r['N'],r['fraction_e'],r['population'])for r in rows}):
        rs=[r for r in rows if (r['N'],r['fraction_e'],r['population'])==(N,fe,a)]
        z=stats([r['components']for r in rs]);z['seeds']=[r['seed']for r in rs];z['leading_static_variance']=BASE['static_variances'][a];z['max_mean_equation_rms']=max(r['mean_equation_rms']for r in rs);z['max_static_identity_sampling_difference']=max(abs(r['mean_state_static_variance_minus_Jm_static_variance'])for r in rs)
        out['groups'][f'f{fe}_N{N}_pop{a}']=z
    (HERE/'static_source_summary.writing').write_text(json.dumps(out,indent=2)+'\n'); (HERE/'static_source_summary.writing').replace(HERE/'static_source_summary.json');print(json.dumps({k:dict(n=v['n'],mean=v['mean'],se=v['se'])for k,v in out['groups'].items()},indent=2))


if __name__=='__main__':main()
