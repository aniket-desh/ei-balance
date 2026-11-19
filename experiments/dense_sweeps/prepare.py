"""Generate prospective degree and composition manifests from the archived protocol."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
CHIS = (0, .025, .05, .1, .2, .3, .5, .75, 1, 1.25, 1.5, 2)
FRACTIONS = (.60, .65, .70, .75, .80, .84, .88, .91, .94)
LARGE_CHIS = (.1, .75, 1.5)
LARGE_FRACTIONS = (.70, .88, .94)


def main():
    archived = json.loads((ROUND / 'runtime/campaign_jobs.json').read_text())
    reference = next(j for j in archived if j['experiment'] == 'composition' and j['config']['fraction_e'] == .8)
    theory = json.loads((ROUND / 'degree/theory_curve.json').read_text())
    h_chi = {row['chi']: row['mean_inputs'] for row in theory['rows']}
    jobs = []
    for n, degrees, fractions in [(2000, CHIS, FRACTIONS), (4000, LARGE_CHIS, LARGE_FRACTIONS)]:
        for replicate in range(3):
            for experiment, grid in [('degree', degrees), ('composition', fractions)]:
                for value in grid:
                    f = .8 if experiment == 'degree' else value
                    seed = (5101 if experiment == 'degree' else 6101) + replicate
                    config = dict(reference['config'], n=n, seed=seed, fraction_e=f)
                    if experiment == 'degree':
                        config.update(ensemble='degree', degree_dispersion=value,
                                      variance_profile=theory['G'], mean_connectivity=theory['J0'])
                        mean_input = h_chi[value]
                    else:
                        # Scale presynaptic columns so G diag(f) and J0 diag(f) stay fixed.
                        for key, source in [('variance_profile', 'G'), ('mean_connectivity', 'J0')]:
                            config[key] = [[row[0] * (.8 / f), row[1] * (.2 / (1 - f))] for row in theory[source]]
                        mean_input = reference['mean_input']
                    jobs.append({'name': f'dense_{experiment}_{value:g}_N{n}_seed{seed}', 'experiment': experiment,
                                 'config': config, 'mean_input': mean_input, 'blocks': [500.,1000.] if n==4000 and replicate==0 else [500.],
                                 'prefixes': [4000.,8000.], 'retain_temporary_trajectory': False,
                                 'shard': 'local' if n == 2000 and replicate == 0 else 'remote'})
    assert len({j['name'] for j in jobs}) == len(jobs) == 81
    path = HERE / 'jobs.json'
    data = json.dumps(jobs, indent=2) + '\n'
    path.write_text(data)
    print(f'{len(jobs)} networks, manifest SHA-256 {hashlib.sha256(data.encode()).hexdigest()}')


if __name__ == '__main__':
    main()
