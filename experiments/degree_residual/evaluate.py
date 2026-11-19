"""Score frozen graph-only predictions on the independent degree recordings."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, t as student_t
from threadpoolctl import threadpool_limits

from residuals import terms
from predict import (HERE,EXP,SOURCE,F,CHI,SEEDS,hermite_power,centered_moment,kernel_basis,
                     Site,population_dimension)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_checks():
    # Compare the local Hermite expansion against the model's unrelated Price quadrature.
    checks = []
    for chiq in CHI:
        _,q,_,filtered = kernel_basis(chiq)
        for qi in [q[0]*.5,q[1]*1.6]:
            mu = np.linspace(-12,7,87)
            coeff = hermite_power(mu,np.full_like(mu,qi))
            for rho in (0,.19,.55,.98,1.):
                approx = coeff@(rho**np.arange(1,coeff.shape[1]+1))
                checks.append(float(np.max(abs(approx-centered_moment(mu,qi,qi*rho)))))
            assert (filtered>0).all() and (filtered<=1).all()
    assert max(checks)<3e-5
    # Exact homogeneous chi=0 control: using repeated identical cells should reproduce
    # the existing population formula, including its three dimension normalizations.
    chi0 = next(r for r in SOURCE['rows'] if r['chi']==0)
    ts,q,shape,_ = kernel_basis(0)
    labels = np.repeat([0,1],[8,2])
    mu = np.asarray(chi0['mean_inputs'])[labels]
    pair = population_dimension(Site(mu,q[labels],labels,ts,shape))
    predicted = [*pair['within_population_dimension_fractions'],pair['dimension_fraction']]
    expected = [*chi0['prediction']['within_population_dimension_fractions'],chi0['prediction']['dimension_fraction']]
    error = float(np.max(abs(np.array(predicted)-expected)))
    assert error<5e-6
    return {'max_abs_Hermite_vs_Price_covariance':max(checks),'homogeneous_E_I_joint_PR_max_absolute_error':error}


def score(path):
    record = json.loads(path.read_text())
    config = record['job']['config']
    chi,seed = config['degree_dispersion'],config['seed']
    assert record['status']=='complete' and seed in SEEDS and chi in CHI and config['n']==2000
    pred_path = HERE/'predictions'/f'chi{chi:g}_seed{seed}.json'
    prediction = json.loads(pred_path.read_text())
    assert prediction['graph_sha256']==record['connectivity_sha256']
    est = next(e for e in record['estimators'] if e['block_duration']==500)
    pre = next(p for p in est['prefixes'] if p['duration']==8000)
    main = next(g for g in pre['gaps'] if g['empty_blocks']==0)
    filename = path.with_name(path.stem+'_B500_neuron_statistics.npz')
    ref = next(row for row in SOURCE['rows'] if row['chi']==chi)
    dims = [*(F*np.array(main['nu_by_population'])**2/np.diag(main['population_energy'])),main['d']]
    row = {'name':record['name'],'chi':chi,'seed':seed,'record_sha256':sha(path),'capsule_sha256':sha(filename),
           'graph_sha256':record['connectivity_sha256'],'dimensions':dims,'local_observed':[],
           'secondary_Gaussian_degree_residual':terms(main,ref)}
    with np.load(filename) as truth, np.load(pred_path.with_suffix('.npz')) as predicted:
        labels = predicted['labels']
        np.testing.assert_array_equal(truth['labels'],labels)
        observed_q = truth['q_x']
        observed_gain = truth['mean_gain']
        for a in range(2):
            mask = labels==a
            row['local_observed'].append({'mean_q':float(np.mean(observed_q[mask])),
                                         'sd_q':float(np.std(observed_q[mask])),
                                         'gain_second':main['ingredient_cross_block'][a]['A_cross_block_timeaveraged_gain']})
        for mode in ('gaussian','shared','joint'):
            local = []
            d = ([*ref['prediction']['within_population_dimension_fractions'],ref['prediction']['dimension_fraction']]
                 if mode=='gaussian' else prediction[mode].get('dimensions'))
            for a in range(2):
                mask = labels==a
                if mode=='gaussian':
                    mu = ref['mean_inputs'][a]
                    q = np.array(ref['dynamic_variances'][a])
                    # Without graph information the class mean is the MSE-optimal constant
                    # predictor of gain under the Gaussian conditional-disorder law.
                    var = q+ref['static_variances'][a]
                    gain = np.sqrt(2/np.pi/(1+var))*np.exp(-mu**2/(2*(1+var)))
                    a2 = ref['alpha_second'][a]
                else:
                    mu = predicted[mode+'_mu'][mask]
                    q = predicted[mode+'_q'][mask]
                    gain = predicted[mode+'_gain'][mask]
                    a2 = prediction[mode]['gain_second'][a]
                obs = row['local_observed'][a]
                correlation = None if np.ptp(q)<1e-12 else float(spearmanr(q,observed_q[mask]).statistic)
                local.append({'gain_mse':float(np.mean((observed_gain[mask]-gain)**2)),
                              'q_mse':float(np.mean((observed_q[mask]-q)**2)),
                              'mu_mse':float(np.mean((truth['mean_x'][mask]-mu)**2)),
                              'gain_second_error':float(a2-obs['gain_second']),
                              'q_spearman':correlation,'q_predicted_sd':float(np.std(q))})
            row[mode]={'dimensions':d,'dimension_log_errors':None if d is None else np.log(np.asarray(d)/dims).tolist(),
                       'local':local}
    return row


def summarize(rows):
    out = {}
    for chi in CHI:
        take = [r for r in rows if r['chi']==chi]
        assert len(take)==3
        entry = {'measured_dimensions':np.mean([r['dimensions'] for r in take],axis=0).tolist()}
        for m in ['gaussian','shared','joint']:
            dims = [r[m]['dimensions'] for r in take if r[m]['dimensions'] is not None]
            logs = [r[m]['dimension_log_errors'] for r in take if r[m]['dimensions'] is not None]
            entry[m]={'accepted_PR':len(dims),'predicted_dimensions_mean':np.mean(dims,axis=0).tolist(),
                      'dimension_MALE':np.mean(np.abs(logs),axis=0).tolist()}
            for var in ['gain_mse','mu_mse','q_mse','gain_second_error','q_predicted_sd','q_spearman']:
                values=[[k[var] for k in r[m]['local']] for r in take]
                if all(v is not None for subrow in values for v in subrow):
                    entry[m][var+'_mean']=np.mean(values,axis=0).tolist()
        out[chi]=entry
    return out


def contrasts(rows):
    # Average over the three prespecified dispersions *within* each matched seed.
    def extract(r,mode,metric):
        if metric=='abs_log_dimension':
            return np.abs(r[mode]['dimension_log_errors'])
        return np.array([pop[metric] for pop in r[mode]['local']])
    out={}
    for m in ('shared','gaussian'):
        entry={}
        for metric in ('gain_mse','q_mse','abs_log_dimension'):
            vals = []
            for seed in SEEDS:
                take = [r for r in rows if r['seed']==seed]
                vals.append(np.mean([extract(r,'joint',metric)-extract(r,m,metric) for r in take],axis=0))
            mean=np.mean(vals,axis=0)
            half=student_t.ppf(.975,2)*np.std(vals,axis=0,ddof=1)/np.sqrt(3)
            entry[metric]={'matched_seed_differences':np.array(vals).tolist(),'mean':mean.tolist(),
                           't95_low':(mean-half).tolist(),'t95_high':(mean+half).tolist()}
        out['joint_minus_'+m]=entry
    return out


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=EXP/'degree_holdout/runs')
    args=parser.parse_args()
    manifest=json.loads((HERE/'prediction_hashes.json').read_text())
    assert manifest['protocol_sha256']==sha(HERE/'PROTOCOL.md') and manifest['prediction_script_sha256']==sha(HERE/'predict.py')
    for name,digest in manifest['files'].items():
        assert digest==sha(HERE/'predictions'/name)
    with threadpool_limits(1):
        checks=numerical_checks()
        paths=[]
        for p in args.runs.glob('*.json'):
            data=json.loads(p.read_text())
            if data.get('job',{}).get('config',{}).get('seed') in SEEDS:
                paths.append(p)
        assert len(paths)==9, f'Expected nine prospective networks, found {len(paths)}'
        rows=[score(p) for p in sorted(paths)]
    assert {(r['chi'],r['seed']) for r in rows}=={(chi,seed) for chi in CHI for seed in SEEDS}
    result={'prediction_manifest_sha256':sha(HERE/'prediction_hashes.json'),'numerical_checks':checks,
            'networks':rows,'by_chi':summarize(rows),'paired_contrasts':contrasts(rows)}
    (HERE/'holdout_scores.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
