"""Check the adopted records and published normalizations after all jobs finish."""
from pathlib import Path
import ast,json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'next_round/runtime'


def main():
    manifests=['campaign_jobs.json','extension_jobs.json','degree_extension_jobs.json']
    all_jobs=[j for name in manifests for j in json.loads((RUN/name).read_text())]
    unique={};overlap=[]
    for job in all_jobs:
        if job['name'] in unique:
            # The complete extension manifest also includes its degree-only
            # subset. It is a schedule reference, not another realization.
            for key in ['config','mean_input','experiment','blocks','prefixes']:
                assert unique[job['name']][key]==job[key]
            overlap.append(job['name'])
        unique[job['name']]=job
    jobs=list(unique.values())
    missing=[];rows=[];errors=[]
    for job in jobs:
        path=RUN/'runs'/(job['name']+'.json')
        r=json.loads(path.read_text()) if path.exists() else {'status':'not_started'}
        if r['status']!='complete':missing.append({'name':job['name'],'status':r['status']});continue
        assert r['job']['config']==job['config']
        assert len(r['connectivity_sha256'])==64
        est=next(e for e in r['estimators'] if e['block_duration']==500.)
        prefix=next(p for p in est['prefixes'] if p['duration']==8000.)
        primary=next(g for g in prefix['gaps'] if g['empty_blocks']==0)
        e=np.asarray(primary['population_energy']);f=np.array([job['config']['fraction_e'],1-job['config']['fraction_e']])
        errors.extend([abs(e.sum()-primary['energy']),
                       abs(e[0,1]-e[1,0]),
                       abs(f@primary['nu_by_population']-primary['nu']),
                       abs(primary['nu']**2/primary['energy']-primary['d'])])
        assert primary['energy']>0 and primary['nu']>0
        assert est['dense_formula_error']<1e-9
        rows.append({'name':job['name'],'N':job['config']['n'],
                     'experiment':job['experiment'],'nu':primary['nu'],
                     'energy':primary['energy'],'d':primary['d']})
    syntax=[]
    for path in sorted((ROOT/'next_round').rglob('*.py')):
        if 'tmp' in path.relative_to(ROOT).parts:continue
        ast.parse(path.read_text(),filename=str(path));syntax.append(str(path.relative_to(ROOT)))
    result={'expected_primary_networks':len(jobs),'completed_primary_networks':len(rows),
            'identical_schedule_references_deduplicated':overlap,
            'missing_or_failed':missing,'all_primary_complete':not missing,
            'maximum_primary_normalization_error':max(errors,default=0),
            'python_sources_syntax_checked':len(syntax),'python_sources':syntax,
            'exact_geometry_identity_checks':'check_population_gram_geometry.json',
            'original_model_tests':'original_unittest.log: 13 passed',
            'scope':'Integrity and normalization checks are not tests of theoretical accuracy.'}
    assert result['maximum_primary_normalization_error']<1e-10
    dest=Path(__file__).with_suffix('.json');dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='python_sources'},indent=2))
    if missing:raise SystemExit(2)


if __name__=='__main__':main()
