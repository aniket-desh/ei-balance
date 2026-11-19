"""Average the eight readouts within each network before computing seed-level intervals."""

from itertools import combinations
import json
from pathlib import Path
import sys
import hashlib
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import t

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'recovered/bundle/ei_full/ei_balance_handoff/src'))
from eicavity.network import NetworkConfig, make_network
FRACTIONS=(.8,.88,.94)
WINDOWS=('T8000_gap0','T8000_gap1','T4000_gap0','T4000_gap1')
METRICS=('PR','normalized_PR','pair_energy')


def by_seed(values):
    numbers=list(values.values()); n=len(numbers)
    if not n or any(x is None for x in numbers):return {'values_by_seed':values,'mean':None,'se':None,'ci95':None}
    mean=float(np.mean(numbers));se=float(np.std(numbers,ddof=1)/np.sqrt(n)) if n>1 else None
    interval=None
    if n==4:
        width=t.ppf(.975,n-1)*se;interval=[float(mean-width),float(mean+width)]
    return {'values_by_seed':values,'mean':mean,'se':se,'ci95':interval}


def source_check(job,obs,blocks):
    name=job['name']; dense=job['full_ensemble_record']; dirs=HERE.parent/'dense_sweeps/runs' if dense else HERE/'runs'
    record=dirs/((dense or name)+'.json')
    status={'source':str(record.relative_to(HERE.parent)),'full_case_role':'dense cohort' if dense else 'new seed 6104'}
    if not record.exists():return {**status,'comparison':'full record pending'},{}
    full=json.loads(record.read_text())
    if full['status']!='complete':return {**status,'comparison':'full record pending'},{}
    if full['connectivity_sha256']!=obs['connectivity_sha256'] or full['job']['config']!=job['config']:
        raise ValueError(f'Full network and readout graph or config disagree in {name}')
    est=next(e for e in full['estimators'] if e['block_duration']==500)
    c=job['config']; n=c['n'];ne=round(n*c['fraction_e']);counts=np.array([ne,n-ne]); full_metrics={}
    for prefix in est['prefixes']:
        for window in prefix['gaps']:
            B=np.array(window['population_energy']); Ddiag=np.array(window['diagonal_population_energy']);
            pair=n*n*(np.diag(B)-Ddiag)/(counts*(counts-1))
            full_metrics[f"T{int(prefix['duration'])}_gap{window['empty_blocks']}"]={'P_EE':float(pair[0]),'P_II':float(pair[1])}
    caps=dirs/Path(est['neuron_statistics_file']).name
    status.update(graph_match=True,native_binary_match=full.get('native_library_sha256')==obs['native_library_sha256'])
    if not caps.exists():return {**status,'comparison':'full population capsule pending'},full_metrics
    with np.load(caps) as original, np.load(blocks) as pilot:
        ids=pilot['indices'];k=ids.shape[-1]; diagonal=(np.arange(1,k+1)*np.arange(2,k+2))//2-1
        original_mean=original['block_mean_rate'][:,ids].transpose(1,2,0,3)
        original_var=original['block_variance_rate'][:,ids].transpose(1,2,0,3)
        error_mean=float(np.max(abs(original_mean-pilot['block_mean_rate'])))
        error_var=float(np.max(abs(original_var-pilot['block_cov_lower'][...,diagonal])))
    status.update(max_block_mean_difference=error_mean,max_block_variance_difference=error_var,
                  comparison='per-cell block moments match' if max(error_mean,error_var)<1e-10 else 'same graph, different numerical trajectory')
    return status,full_metrics


def edge_counts(job,expected_sha):
    J,_,_,labels,info=make_network(NetworkConfig(**job['config']),np.array(job['mean_input']))
    if hashlib.sha256(J).hexdigest()!=expected_sha:raise ValueError('Graph checksum differs while counting incoming edges.')
    cells=[slice(0,int((labels==0).sum())),slice(int((labels==0).sum()),len(labels))]
    rows=[]
    for a,receiver in enumerate(cells):
        for b,sender in enumerate(cells):
            size=sender.stop-sender.start; degree=np.count_nonzero(J[receiver,sender],axis=1)
            p=float(info['probabilities'][a,b]);rows.append({'target':('E','I')[a],'source':('E','I')[b],
                'source_pool_size':size,'expected_missing_inputs':size*(1-p),'mean_missing_inputs':float((size-degree).mean()),
                'mean_degree':float(degree.mean()),'degree_variance':float(degree.var()),'total_edges':int(degree.sum())})
    return rows


def main():
    manifest=json.loads((HERE/'protocol.json').read_text()); rows=[];missing=[]
    for j in manifest['jobs']:
        path=HERE/'measurements'/(j['name']+'.json')
        if not path.is_file():missing.append(j['name']);continue
        data=json.loads(path.read_text());ids=np.array(data['subset_indices'])
        if j['config']!=data['config'] or not np.array_equal(ids,j['subsets']):raise ValueError(f'Protocol mismatch in {path.name}')
        f=j['config']['fraction_e']; ne=round(j['config']['n']*f); pools=[ne,j['config']['n']-ne]
        row={'name':j['name'],'fraction_e':f,'seed':j['config']['seed'],'population_sizes':pools,'windows':{},'mean_pairwise_subset_overlap':{}}
        row['source_comparison'],row['full_population']=source_check(j,data,HERE/'measurements'/data['block_statistics'])
        row['incoming_edge_counts']=edge_counts(j,data['connectivity_sha256'])
        for pop,letter in enumerate(('E','I')):
            row['mean_pairwise_subset_overlap'][letter]=float(np.mean([len(np.intersect1d(a,b))/ids.shape[-1] for a,b in combinations(ids[pop],2)]))
            for window in WINDOWS:
                entry=row['windows'].setdefault(window,{}).setdefault(letter,{})
                for metric in METRICS:
                    values=[x[window][metric] for x in data['readouts'][pop]]
                    if any(v is None or not np.isfinite(v) for v in values):
                        entry[metric]={'draws':values,'mean':None,'sd':None,'min':None,'max':None}
                    else:
                        entry[metric]={'draws':values,'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1)),'min':min(values),'max':max(values)}
        rows.append(row)
    groups=[]; contrasts=[]; full_contrasts=[]
    for window in WINDOWS:
        for pop in ('E','I'):
            for metric in METRICS:
                observations={}
                for f in FRACTIONS:
                    observations[f]={r['seed']:r['windows'][window][pop][metric]['mean'] for r in rows if r['fraction_e']==f}
                    groups.append({'fraction_e':f,'window':window,'population':pop,'metric':metric,**by_seed(observations[f])})
                for a,b in zip(FRACTIONS[:-1],FRACTIONS[1:]):
                    delta={s:observations[b][s]-observations[a][s] for s in observations[b].keys()&observations[a].keys()
                           if observations[b][s] is not None and observations[a][s] is not None}
                    contrasts.append({'lower':a,'upper':b,'window':window,'population':pop,'metric':metric,**by_seed(delta)})
        for metric in ('P_EE','P_II'):
            by_fraction={f:{r['seed']:r['full_population'][window][metric] for r in rows if r['fraction_e']==f and window in r['full_population']} for f in FRACTIONS}
            for a,b in zip(FRACTIONS[:-1],FRACTIONS[1:]):
                delta={s:by_fraction[b][s]-by_fraction[a][s] for s in by_fraction[b].keys()&by_fraction[a].keys()}
                full_contrasts.append({'lower':a,'upper':b,'window':window,'metric':metric,**by_seed(delta)})
    output={'complete':len(rows),'expected':len(manifest['jobs']),'missing':missing,'readout_size':manifest['readout_size'],
            'draws_per_network_per_class':manifest['draws_per_class'],'uncertainty':'Mean of eight actual readout PRs within each seed, then pointwise two-sided t interval across four network seed labels.',
            'records':rows,'groups':groups,'contrasts':contrasts,'full_population_contrasts':full_contrasts}
    with TemporaryDirectory(dir=HERE,prefix='.summary-') as stage:
        dest=Path(stage)/'summary.json';dest.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n');dest.replace(HERE/'summary.json')
    print(f'{len(rows)} / 12 composition networks, each with 8 E and 8 I readouts')


if __name__=='__main__': main()
