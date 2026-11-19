"""Summarize measured draws, retaining seed-level contrasts and finite-record controls."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent


def metrics(measurement, counts):
    n = counts.sum()
    fractions = counts / n
    energy = np.asarray(measurement['population_energy'])
    diagonal = np.asarray(measurement['diagonal_population_energy'])
    nu = np.asarray(measurement['nu_by_population'])
    np.testing.assert_allclose(energy, energy.T, rtol=2e-12, atol=1e-13)
    np.testing.assert_allclose([nu @ fractions, energy.sum()], [measurement['nu'], measurement['energy']], atol=1e-13)
    if np.isfinite(measurement['d']):
        np.testing.assert_allclose(measurement['d'] * energy.sum(), (nu @ fractions)**2, atol=1e-13)
    with np.errstate(divide='ignore', invalid='ignore'):
        within = fractions * nu**2 / np.diag(energy)
    # B and Ddiag have denominator total N. Remove the same-neuron terms before
    # dividing by the actual number of distinct ordered pairs in this population.
    pair = n**2 * (np.diag(energy) - diagonal) / (counts * (counts - 1))
    values = dict(d=measurement['d'], d_E=within[0], d_I=within[1], D_E_by_N=fractions[0]*within[0],D_I_by_N=fractions[1]*within[1],
                  P_EE=pair[0],P_II=pair[1],P_EI=n**2*energy[0,1]/counts.prod(),nu=measurement['nu'],nu_E=nu[0],nu_I=nu[1],
                  B_EE=energy[0,0],B_EI=energy[0,1],B_II=energy[1,1],energy=energy.sum(),
                  diagonal_E=diagonal[0],diagonal_I=diagonal[1])
    return {key:float(value) if np.isfinite(value) else None for key, value in values.items()}


def estimate(values):
    if not values or any(x is None for x in values):
        return {'mean':None,'se':None}
    se = float(np.std(values,ddof=1) / np.sqrt(len(values))) if len(values)>1 else None
    return {'mean':float(np.mean(values)), 'se':se}


def paired_estimate(differences, expected):
    stats = estimate(list(differences.values()))
    ci = None
    if len(differences)==expected and stats['se'] is not None:
        margin = float(t.ppf(.975,expected-1))*stats['se']
        ci = [stats['mean']-margin,stats['mean']+margin]
    sign = 'pending' if ci is None else ('positive' if ci[0]>0 else 'negative' if ci[1]<0 else 'inconclusive')
    return {'difference_by_seed':differences,**stats,'pointwise_95pct_t_interval':ci,'interval_sign':sign}


def window_signs(estimates):
    primary = estimates['T8000_B500_gap0']
    p = primary['mean']
    return {window:None if primary['pointwise_95pct_t_interval'] is None or row['pointwise_95pct_t_interval'] is None
            else bool(np.sign(row['mean'])!=np.sign(p)) for window,row in estimates.items()}


def interior_curve(groups, jobs):
    parent = HERE.parent
    archived = json.loads((parent/'runtime/campaign_jobs.json').read_text())
    old = {j['config']['degree_dispersion'] for j in archived if j['experiment']=='degree'}
    sites = sorted({j['config']['degree_dispersion'] for j in jobs if j['experiment']=='degree' and j['config']['n']==2000}-old)
    theory = json.loads((parent/'degree/theory_curve.json').read_text())
    log_d = {r['chi']:float(np.log(r['prediction']['dimension_fraction'])) for r in theory['rows']}
    target = np.array([log_d[x] for x in sites])
    secant = log_d[0] + (log_d[2]-log_d[0])*np.array(sites)/2
    seeds = sorted({j['config']['seed'] for j in jobs if j['experiment']=='degree'})
    analysis = {}
    for duration in (4000,8000):
        for gap in (0,1):
            key = f'T{duration}_B500_gap{gap}'
            differences, errors = {}, {}
            for seed in seeds:
                values = {x:r['windows'][key]['d'] for x in sites for r in groups.get(('degree',2000,x),[]) if r['seed']==seed}
                missing = [x for x in sites if values.get(x) is None or values[x]<=0]
                if missing:
                    errors[seed] = {'missing_or_nonpositive_at_chi':missing}
                    continue
                y = np.log([values[x] for x in sites])
                a = float(np.sqrt(np.mean((y-target)**2)))
                b = float(np.sqrt(np.mean((y-secant)**2)))
                errors[seed] = {'rmse_theory':a,'rmse_secant':b}
                differences[seed] = a-b
            analysis[key] = {**paired_estimate(differences,len(seeds)), 'per_seed_errors':errors}
    return {'scope':'Supplemental N2000 curve comparison. Each previously unsimulated chi has equal weight. Negative differences favor theory.',
            'chi':sites,'fixed_log_prediction':target.tolist(),'endpoint_secant_in_log_d':secant.tolist(),
            'windows':analysis,'mean_sign_changes_vs_primary':window_signs(analysis)}


def main():
    manifest = HERE/'jobs.json'
    jobs = json.loads(manifest.read_text())
    records = []
    pending = {}
    grouped = defaultdict(list)
    for job in jobs:
        name = job['name']
        path = HERE/'runs'/(name+'.json')
        if not path.exists():
            pending[name] = 'missing'
            continue
        record = json.loads(path.read_text())
        saved_job = dict(record['job'])
        saved_job.pop('native_library',None)
        if saved_job != job or record['name'] != name:
            raise ValueError(f'Manifest mismatch in {name}')
        if record['status'] != 'complete':
            pending[name] = record['status']
            continue
        c = job['config']
        f = np.asarray(record['network_info']['fractions'])
        counts = np.rint(c['n']*f).astype(int)
        np.testing.assert_array_equal(counts, [round(c['n']*c['fraction_e']),c['n']-round(c['n']*c['fraction_e'])])
        control = c['degree_dispersion'] if job['experiment']=='degree' else c['fraction_e']
        windows = {}
        warnings = []
        for estimator in record['estimators']:
            block = int(estimator['block_duration'])
            if estimator['nblocks'] != round(c['duration']/block):
                raise ValueError(f'Unexpected number of blocks in {name}')
            capsule = HERE/'runs'/Path(estimator['neuron_statistics_file']).name
            if not capsule.is_file():
                warnings.append(f'missing neuron statistics for B={block}')
            for prefix in estimator['prefixes']:
                for gap in prefix['gaps']:
                    window = f"T{int(prefix['duration'])}_B{block}_gap{gap['empty_blocks']}"
                    values = metrics(gap,counts)
                    if any(v is None or v<=0 for v in values.values()):
                        warnings.append(f'nonpositive or undefined estimate in {window}')
                    windows[window] = values
        primary = windows['T8000_B500_gap0']
        info = record['network_info']
        degree = None
        if job['experiment']=='degree':
            p = np.asarray(info['probabilities'])
            finite_chi = np.asarray(info['degree_empirical_variances']) / (counts[None,:]*p*(1-p))
            degree = {'expected':info['degree_expected_dispersion'],'realized':finite_chi.tolist()}
        row = {'name':name,'experiment':job['experiment'],'N':c['n'],'control':control,'seed':c['seed'],
               'population_counts':counts.tolist(),'degree_dispersion_by_block':degree,'primary':primary,
               'windows':windows,'flags':warnings,'connectivity_sha256':record['connectivity_sha256'],
               'record_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'final_drift_rms':record['final_drift_rms'],
               'engine_source_sha256':record['engine_source_sha256'],'action_backend':record['action_backend'],
               'native_library_sha256':record.get('native_library_sha256'),
               'simulation_seconds':record['simulation_seconds'],'elapsed_seconds':record['elapsed_seconds']}
        records.append(row)
        grouped[(row['experiment'],c['n'],control)].append(row)
    groups = [{'experiment':exp,'N':n,'control':x,'seeds':[r['seed'] for r in rows],
               'estimate':{k:estimate([r['primary'][k] for r in rows]) for k in rows[0]['primary']}}
              for (exp,n,x),rows in sorted(grouped.items())]
    contrasts = {}
    for exp,lo,hi,key,seeds in [('degree',0,2,'d',range(5101,5104)),('composition',.8,.88,'P_EE',range(6101,6104)),
                                 ('composition',.8,.88,'P_II',range(6101,6104)),('composition',.8,.88,'D_E_by_N',range(6101,6104)),
                                 ('composition',.88,.94,'D_E_by_N',range(6101,6104))]:
        by_seed = [{r['seed']:r for r in grouped.get((exp,2000,x),[])} for x in (lo,hi)]
        estimates = {}
        for duration in (4000,8000):
            for gap in (0,1):
                window = f'T{duration}_B500_gap{gap}'
                differences = {}
                for seed in seeds:
                    if all(seed in data for data in by_seed):
                        a,b = [data[seed]['windows'][window][key] for data in by_seed]
                        differences[seed] = b-a if None not in (a,b) else None
                estimates[window] = paired_estimate(differences,len(seeds))
        contrasts[f'{exp}_{key}_{hi:g}_minus_{lo:g}'] = {**estimates,'mean_sign_changes_vs_primary':window_signs(estimates)}
    summary = {'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(), 'expected':len(jobs),
               'complete':len(records),'pending_or_failed':pending,'status_counts':{'complete':len(records),**Counter(pending.values())},
               'records':records, 'groups':groups, 'paired_N2000_contrasts':contrasts,
               'composition_scope':'Three-seed pilot at N=2000, with additional N=4000 measurements at selected anchors.',
               'supplemental_degree_interior':interior_curve(grouped,jobs)}
    with TemporaryDirectory(dir=HERE, prefix='.summary-') as stage:
        work = Path(stage)/'summary.json'
        work.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        work.replace(HERE/'summary.json')
    print(f'{len(records)} / {len(jobs)} complete, {len(pending)} missing, running, or failed')


if __name__ == '__main__':
    main()
