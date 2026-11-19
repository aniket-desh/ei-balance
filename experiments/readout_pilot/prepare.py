"""Fix the four connectivity seeds and all uniformly sampled readouts before analysis."""

import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SUBSET_SEED = 20261119
SIZE = 100
DRAWS = 8


def main():
    dense = json.loads((HERE.parent/'dense_sweeps/jobs.json').read_text())
    jobs = []
    for seed in (6101,6104,6102,6103):
        for fraction in (.8,.88,.94):
            base = next(j for j in dense if j['experiment']=='composition' and j['config']['n']==2000
                        and j['config']['seed']==6101 and j['config']['fraction_e']==fraction)
            config = dict(base['config'],seed=seed)
            n = config['n']
            ne = round(fraction*n)
            subsets = []
            for pop,(offset,count) in enumerate(((0,ne),(ne,n-ne))):
                readouts = []
                for draw in range(DRAWS):
                    rng = np.random.default_rng(np.random.SeedSequence([SUBSET_SEED,seed,pop,draw]))
                    # Fixed priorities on class-relative IDs couple the readout choices across fractions.
                    priorities = rng.random(n)
                    ids = np.sort(np.argsort(priorities[:count],kind='stable')[:SIZE]+offset)
                    readouts.append(ids.tolist())
                subsets.append(readouts)
            jobs.append({'name':f'readout_f{fraction:g}_N{n}_seed{seed}','config':config,'mean_input':base['mean_input'],
                         'full_ensemble_record':f'dense_composition_{fraction:g}_N{n}_seed{seed}' if seed!=6104 else None,
                         'subsets':subsets})
    manifest = {'network_seeds':[6101,6102,6103,6104],'fractions':[.8,.88,.94],'subset_seed':SUBSET_SEED,
                'readout_size':SIZE,'draws_per_class':DRAWS,'jobs':jobs}
    (HERE/'protocol.json').write_text(json.dumps(manifest,separators=(',',':'))+'\n')


if __name__=='__main__':
    main()
