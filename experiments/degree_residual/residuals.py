"""Exact per-network log dimension residuals and held-half local gain diagnostics."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
F = np.array([.8,.2])
SOURCE = EXP/'degree/theory_curve.json'
THEORY = {row['chi']:row for row in json.loads(SOURCE.read_text())['rows']}


def terms(measure, theory):
    """Integrate d log(E)=sum_b dE_b/E along the straight path between energies."""
    observed = np.asarray(measure['population_energy'])
    expected = np.asarray(theory['block_energies'])
    e0 = expected.sum()
    delta = (observed.sum()-e0)/e0
    weight = (np.log1p(delta)/delta if delta else 1.)/e0
    matrix = -(observed-expected)*weight
    parts = {'rate_numerator':float(2*np.log(measure['nu']/theory['nu'])),
             'ee':float(matrix[0,0]),'ei_both':float(matrix[0,1]+matrix[1,0]),'ii':float(matrix[1,1])}
    residual = float(np.log(measure['d']/theory['prediction']['dimension_fraction']))
    np.testing.assert_allclose(sum(parts.values()),residual,atol=1e-12)
    np.testing.assert_allclose(observed.sum(),measure['energy'],atol=1e-12)
    np.testing.assert_allclose(observed,observed.T,atol=1e-12)
    if 'diagonal_population_energy' in measure:
        diag = np.asarray(measure['diagonal_population_energy'])
        diag0 = F*np.array(theory['prediction']['variance_seconds'])
        parts['ee_diag'],parts['ii_diag'] = (-(diag-diag0)*weight).tolist()
        parts['ee_off'] = parts['ee']-parts['ee_diag']
        parts['ii_off'] = parts['ii']-parts['ii_diag']
    return {'log_d_error':residual,**parts}


def halves(path):
    with np.load(path) as z:
        labels = z['labels']
        mu = z['block_mean_x']
        second = z['block_q_x']+mu**2
        gain = z['block_mean_gain']
    m1,m2 = np.split(mu,2)
    s1,s2 = np.split(second,2)
    g1,g2 = np.split(gain,2)
    out = []
    for a in range(2):
        mask = labels==a
        losses,correlations = [],[]
        for m,s,g,other_m,other_s in ((m1,s1,g2,m2,s2),(m2,s2,g1,m1,s1)):
            h = m[:,mask].mean(0)
            q = s[:,mask].mean(0)-h*h
            other_q = other_s[:,mask].mean(0)-other_m[:,mask].mean(0)**2
            correlations.append(spearmanr(h,other_q).statistic)
            held = g[:,mask].mean(0)
            pred = lambda v: np.sqrt(2/np.pi/(1+v))*np.exp(-h*h/(2*(1+v)))
            losses.append([np.mean((pred(v)-held)**2) for v in (q.mean(),q)])
        shared,local = np.sqrt(np.mean(losses,axis=0))
        out.append({'gain_rmse_shared_q':float(shared),'gain_rmse_cell_q':float(local),
                    'opposite_half_mean_q_spearman':float(np.mean(correlations))})
    return out


def make():
    rows = []
    for cohort, paths in [('dense',(EXP/'dense_sweeps/runs').glob('dense_degree_*.json')),
                          ('original',(EXP/'runtime/runs').glob('degree_*.json'))]:
        for path in sorted(paths):
            run = json.loads(path.read_text())
            cfg = run['job']['config']
            assert run['status']=='complete' and run['job']['experiment']=='degree'
            chi = cfg['degree_dispersion']
            theory = THEORY[chi]
            est = next(e for e in run['estimators'] if e['block_duration']==500)
            row = {'name':run['name'],'measurement':str(path.relative_to(EXP.parent)),
                   'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cohort':cohort,
                   'chi':chi,'N':cfg['n'],'seed':cfg['seed']}
            for key,duration,gap in [('primary',8000,0),('short',4000,0),('gap',8000,1)]:
                prefix = next(p for p in est['prefixes'] if p['duration']==duration)
                m = next(g for g in prefix['gaps'] if g['empty_blocks']==gap)
                row[key] = terms(m,theory)
                if key=='primary':
                    row['d'] = m['d']
                    if 'ingredient_cross_block' in m:
                        metrics = {}
                        for var,field,pred in [('gain2','A_cross_block_timeaveraged_gain','alpha_second'),
                                               ('q','q_mean_free_input','dynamic_variances'),
                                               ('delta','Delta_cross_block_inputmeans','static_variances')]:
                            arr = np.array([pop[field] for pop in m['ingredient_cross_block']])
                            metrics[var+'_error'] = (arr-np.array(theory[pred])).tolist()
                        row['local']=metrics
            if cohort=='dense':
                capsule = path.with_name(path.stem+'_B500_neuron_statistics.npz')
                row['split_half']=halves(capsule)
                row['capsule_sha256']=hashlib.sha256(capsule.read_bytes()).hexdigest()
            rows.append(row)
    assert len(rows)==69 and len({r['name'] for r in rows})==69
    groups = defaultdict(list)
    for row in rows:
        groups[(row['cohort'],row['N'],row['chi'])].append(row)
    summaries = []
    for (cohort,n,chi),rset in sorted(groups.items()):
        out = {'cohort':cohort,'N':n,'chi':chi,'seeds':[r['seed'] for r in rset]}
        for key in ('primary','short','gap'):
            out[key]={}
            for metric in sorted(set.intersection(*(set(r[key]) for r in rset))):
                vals = np.array([r[key][metric] for r in rset])
                out[key][metric] = [float(vals.mean()),float(vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else None]
        summaries.append(out)
    associations=[]
    for seed in (5101,5102,5103):
        group = sorted((r for r in rows if r['N']==2000 and r['seed']==seed),key=lambda r:r['chi'])
        assert len(group)==12
        for var in ('gain2_error','delta_error','q_error'):
            for a in range(2):
                associations.append({'seed':seed,'class':'EI'[a],'local_discrepancy':var,
                                     'spearman_versus_ee_energy_contribution':float(spearmanr(
                                         [r['local'][var][a] for r in group],[r['primary']['ee'] for r in group]).statistic)})
    return {'theory_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'experiments':rows,
            'means_and_network_se':summaries,'within_seed_associations_over_chi':associations}


if __name__=='__main__':
    result = make()
    (HERE/'residuals.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
