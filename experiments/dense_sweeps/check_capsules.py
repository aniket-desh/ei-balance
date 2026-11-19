"""Reconstruct the trace and diagonal-energy estimates from saved neuron statistics."""

from itertools import combinations
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

for variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np

HERE = Path(__file__).resolve().parent


def check(estimator):
    path = HERE/'runs'/Path(estimator['neuron_statistics_file']).name
    with np.load(path, allow_pickle=False) as data:
        means = data['block_mean_rate']
        variances = data['block_variance_rate']
        labels = data['labels']
    pairs = list(combinations(range(len(means)),2))
    index = {pair:i for i,pair in enumerate(pairs)}
    # This is the per-neuron diagonal of a mean-difference covariance
    # constructed using blocks a and b. Independent pairs supply its square.
    diagonals = np.array([.5*(variances[a]+variances[b]+(means[a]-means[b])**2) for a,b in pairs])
    max_nu = max_diagonal = 0.
    for population in (0,1):
        per_pair = diagonals[:,labels==population]
        nu = per_pair.mean(axis=1)
        products = per_pair @ per_pair.T / len(labels)
        for prefix in estimator['prefixes']:
            n = round(prefix['duration']/estimator['block_duration'])
            for trial in prefix['gaps']:
                gap = trial['empty_blocks']
                chosen = [i for i,(a,b) in enumerate(pairs) if b<n and b-a>gap]
                contributions = []
                for a,b,c,d in combinations(range(n),4):
                    if min(b-a,c-b,d-c)>gap:
                        contributions.extend((products[index[a,b],index[c,d]],products[index[a,c],index[b,d]],
                                              products[index[a,d],index[b,c]]))
                if len(contributions)!=3*trial['quadruples']:
                    raise ValueError('The number of disjoint window matchings differs.')
                max_nu = max(max_nu,abs(float(nu[chosen].mean())-trial['nu_by_population'][population]))
                max_diagonal = max(max_diagonal,abs(float(np.mean(contributions))-trial['diagonal_population_energy'][population]))
    if max(max_nu,max_diagonal)>2e-12:
        raise ValueError(f'{path.name}: trace/diagonal discrepancies {max_nu:g}, {max_diagonal:g}')
    return {'capsule':path.name,'nu_max_abs_error':max_nu,'diagonal_max_abs_error':max_diagonal}


def main():
    jobs = json.loads((HERE/'jobs.json').read_text())
    cases = {}
    missing = []
    for job in jobs:
        record = HERE/'runs'/(job['name']+'.json')
        r = json.loads(record.read_text()) if record.is_file() else {}
        if r.get('status')!='complete':
            missing.append(job['name'])
        else:
            cases[job['name']] = [check(e) for e in r['estimators']]
    with TemporaryDirectory(dir=HERE,prefix='.checks-') as stage:
        target = Path(stage)/'capsule_checks.json'
        target.write_text(json.dumps({'verified':cases,'missing_or_incomplete':missing},indent=2)+'\n')
        target.replace(HERE/target.name)
    print(f'{len(cases)} / {len(jobs)} completed cases checked in all available time windows.')


if __name__ == '__main__':
    main()
