"""Quantify diagnostic Gaussian marginal substitutions separately from predictions."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import t
HERE=Path(__file__).resolve().parent


def stat(x):
    x=np.asarray(x,float);n=len(x);m=x.mean(0)
    if n>1:
        sd=x.std(0,ddof=1);se=sd/np.sqrt(n);ci=np.stack([m-t.ppf(.975,n-1)*se,m+t.ppf(.975,n-1)*se],axis=-1).tolist()
        return dict(n=n,mean=m.tolist(),sd=sd.tolist(),se=se.tolist(),ci95_t=ci)
    return dict(n=n,mean=m.tolist(),sd=None,se=None,ci95_t=None)


def main():
    rows=[]
    for p in sorted(HERE.glob('*_local_addon.json')):
        d=json.loads(p.read_text());c=d['config']
        for z in d['groups']:
            primary=next(w for w in z['cross_block']if w['empty_blocks']==0)
            value=dict(A_Gaussian_individual_mu_q_error=z['A_Gaussian_individual_mu_q']/z['A_full_record']-1,A_Gaussian_empirical_mu_common_q_error=z['A_Gaussian_individual_mu_common_q']/z['A_full_record']-1,A_Gaussian_population_h_Delta_q_error=z['A_Gaussian_population_h_Delta_q']/z['A_full_record']-1,nu_Gaussian_individual_mu_q_error=z['nu_Gaussian_individual_mu_q']/z['nu_measured_own_record']-1,m2_Gaussian_individual_mu_q_error=z['mean_rate_squared_Gaussian_individual_mu_q']/z['mean_rate_squared_measured']-1,m2_Gaussian_population_h_Delta_q_error=z['mean_rate_squared_Gaussian_population_h_Delta_q']/z['mean_rate_squared_measured']-1,A_cross_block_vs_square_full_mean=primary['A_cross']/z['A_full_record']-1,q_cross_block_vs_own_record=primary['q_cross']/z['q_own_record']-1,Delta_cross_block_vs_own_record=primary['Delta_cross']/z['Delta_own_record']-1)
            value['A_population_to_empirical_static_label_sector']=value['A_Gaussian_population_h_Delta_q_error']-value['A_Gaussian_empirical_mu_common_q_error'];value['A_common_to_individual_q_sector']=value['A_Gaussian_empirical_mu_common_q_error']-value['A_Gaussian_individual_mu_q_error'];value['A_individual_temporal_marginal_sector']=value['A_Gaussian_individual_mu_q_error']
            np.testing.assert_allclose(value['A_population_to_empirical_static_label_sector']+value['A_common_to_individual_q_sector']+value['A_individual_temporal_marginal_sector'],value['A_Gaussian_population_h_Delta_q_error'],atol=1e-14)
            rows.append(dict(name=d['name'],N=c['n'],fraction_e=c['fraction_e'],seed=c['seed'],population=z['population'],values=value))
    out={'scope':'Same-recordmomentcalibrations, notparameter-onlypredictionsorindependentheldouttests. GaussianmarginalchecksdoNOTvalidatefulltemporalGaussianprocessresponseorcollectivepairclosure. Sample: selectedseed3001initialpointsplusallv2extensionswithcompactmoments.','groups':{},'per_network':rows}
    for N,fe,a in sorted({(r['N'],r['fraction_e'],r['population'])for r in rows}):
        rs=[r for r in rows if (r['N'],r['fraction_e'],r['population'])==(N,fe,a)]
        out['groups'][f'f{fe}_N{N}_pop{a}']={'seeds':[r['seed']for r in rs],'metrics':{k:stat([r['values'][k]for r in rs])for k in rs[0]['values']}}
    (HERE/'local_calibration_summary.writing').write_text(json.dumps(out,indent=2)+'\n'); (HERE/'local_calibration_summary.writing').replace(HERE/'local_calibration_summary.json')
    print(json.dumps({k:{name:m['mean']for name,m in x['metrics'].items()}for k,x in out['groups'].items()},indent=2))


if __name__=='__main__':main()
