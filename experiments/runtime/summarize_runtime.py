"""Write exact completion/optimization facts, without converting pilots to claims."""
from pathlib import Path
import json,hashlib,time
import numpy as np
HERE=Path(__file__).resolve().parent

def read(path):return json.loads(path.read_text())
def primary(result):return next(e for e in result['estimators'] if e['block_duration']==500.)['prefixes'][-1]['gaps'][0]

def local_at_duration(result,duration):
 est=next(e for e in result['estimators'] if e['block_duration']==500.)
 filename=est.get('neuron_statistics_file')
 if not filename:return est['population_local_stats']
 arrays=np.load(HERE.parents[1]/filename)
 k=round(duration/500.);labels=arrays['labels'];mu=arrays['block_mean_x'][:k].mean(0)
 q=(arrays['block_q_x'][:k]+arrays['block_mean_x'][:k]**2).mean(0)-mu**2
 gain=arrays['block_mean_gain'][:k].mean(0);m=arrays['block_mean_rate'][:k].mean(0)
 var=(arrays['block_variance_rate'][:k]+arrays['block_mean_rate'][:k]**2).mean(0)-m**2
 return [{'q_temporal_input':float(q[labels==a].mean()),'A_timeaveraged_gain_squared':float(np.square(gain[labels==a]).mean()),'nu_temporal_rate':float(var[labels==a].mean())} for a in range(2)]

def summarize():
 initial=read(HERE/'campaign_jobs.json');extension=read(HERE/'extension_jobs.json');checks=read(HERE/'integration_check_jobs.json')
 alljobs=initial+extension;records=[];missing=[]
 for job in alljobs:
  path=HERE/'runs'/(job['name']+'.json')
  record=read(path) if path.exists() else {'status':'not_started'}
  if record.get('status')!='complete':missing.append({'name':job['name'],'status':record.get('status')})
  else:records.append(record)
 dt=[]
 for job in checks:
  path=HERE/'runs'/(job['name']+'.json')
  if not path.exists() or read(path).get('status')!='complete':continue
  small=read(path);baseline_name=job.get('baseline_name',job['name'].removeprefix('dtcheck_').removesuffix('_dt0025'));base=read(HERE/'runs'/(baseline_name+'.json'))
  y=primary(small);duration=small['estimators'][0]['prefixes'][-1]['duration'];x=next(z for z in base['estimators'][0]['prefixes'] if z['duration']==duration)['gaps'][0]
  same=base['connectivity_sha256']==small['connectivity_sha256'];assert same
  local0=local_at_duration(base,duration);local1=local_at_duration(small,duration)
  dt.append({'case':baseline_name,'compared_duration':duration,'same_connectivity_SHA256':same,
   'baseline_full_record':{k:primary(base)[k] for k in ['nu','energy','d']},'baseline':{k:x[k] for k in ['nu','energy','d']},'half_dt':{k:y[k] for k in ['nu','energy','d']},
   'relative_changes':{k:y[k]/x[k]-1 for k in ['nu','energy','d']},
   'population_relative_changes':[{k:local1[a][k]/local0[a][k]-1 for k in ['q_temporal_input','A_timeaveraged_gain_squared','nu_temporal_rate']} for a in range(2)],
   'interpretation':'Stationary-statistic check on same connectivity; finite-record chaotic sampling and timestep changes both contribute. No pointwise long-trajectory agreement asserted.'})
 tangents=[]
 for path in sorted((HERE/'checks').glob('*_tangent.json')):
  record=read(path);assert record['duration']==2000. and record['burn']==500. and len(record['eight_duration_blocks'])==8 and np.isfinite(record['eight_duration_blocks']).all()
  baseline=read(HERE/'runs'/(record['case_name']+'.json'));assert record['connectivity_sha256']==baseline['connectivity_sha256']
  tangents.append({'case':record['case_name'],'largest_finite_time_exponent':record['largest_finite_time_exponent'],'minimum_250time_block':min(record['eight_duration_blocks']),'max_rss_bytes':record.get('max_rss_bytes')})
 diagnostics=[]
 diagnostic_manifest=HERE/'outlier_diagnostic_jobs.json'
 expected_diagnostics=read(diagnostic_manifest) if diagnostic_manifest.exists() else []
 for job in expected_diagnostics:
  path=HERE/'runs'/(job['name']+'.json')
  if not path.exists() or read(path).get('status')!='complete':continue
  record=read(path);est=record['estimators'][0]
  baseline=read(HERE/'runs'/(job['baseline_name']+'.json'));assert record['connectivity_sha256']==baseline['connectivity_sha256']
  row={'name':job['name'],'dt':job['config']['dt'],'same_connectivity_SHA256':True,'prefixes':[]}
  for prefix in est['prefixes']:
   primary_value=prefix['gaps'][0];local=local_at_duration(record,prefix['duration'])
   row['prefixes'].append({'duration':prefix['duration'],**{k:primary_value[k] for k in ['nu','energy','d']},'population_local':local})
  if job['config']['dt']==.025:
   old=read(HERE/'runs'/'dtcheck_composition_f80_N1000_seed3001_dt0025.json')
   new8=next(v for v in row['prefixes'] if v['duration']==8000.)
   row['prior_halfstep_prefix_reproduction_absolute_difference']={k:abs(new8[k]-primary(old)[k]) for k in ['nu','energy','d']}
   assert max(row['prior_halfstep_prefix_reproduction_absolute_difference'].values())<1e-10
  diagnostics.append(row)
 audits=[]
 for fn in ['resource_audit.json','extension_first_jobs_resource_audit.json','extension_remaining_jobs_resource_audit.json','integration_check_jobs_resource_audit.json','outlier_diagnostic_jobs_resource_audit.json']:
  path=HERE/fn
  if path.exists():
   r=read(path);audits.append({'file':fn,'workers':r['workers'],'maximum_observed_aggregate_RSS_bytes':r['maximum_observed_aggregate_RSS_bytes'],'budget_bytes':r['memory_budget_bytes']})
 bench=read(HERE/'optimization_benchmark.json')
 out={'status':'complete' if not missing and len(dt)==len(checks) and len(tangents)==4 and len(diagnostics)==len(expected_diagnostics) else 'in_progress',
      'updated_unix':time.time(),'primary_cases_expected':len(alljobs),'primary_cases_completed':len(records),'missing_primary':missing,
      'time_step_checks_completed':len(dt),'time_step_checks_expected':len(checks),'time_step_checks':dt,
      'tangent_checks_completed':len(tangents),'tangent_checks_expected':4,'tangent_checks':tangents,
      'outlier_diagnostic_cases_expected':len(expected_diagnostics),'outlier_diagnostic_cases_completed':len(diagnostics),'outlier_diagnostics':diagnostics,
      'resource_audits':audits,'maximum_observed_network_phase_aggregate_RSS_bytes':max(r['maximum_observed_aggregate_RSS_bytes'] for r in audits),
      'dense_to_sparse_speedups_singleworker':[{ 'N':c['N'],'fe':c['fe'],'factor':c['seconds_per_midpoint_step']['dense1']/c['seconds_per_midpoint_step']['sparse_complement']} for c in bench['cases']],
      'maximum_dense_action_error':max(c['max_action_error'] for c in bench['cases']),
      'network_backends_actual':'v1SciPy_sparsethenv2compiledconstantCSR. Largesecondwave2workers. V3/V4arevalidatedlower-memoryportableexports,notactualprimarycampaignbackend.',
      'portable_export_validation':'portable_export_validation.json','portable_macOS_execution_tested':False,
      'all_listed_outcomes_retained':True,'primary_sampling_decisions':'campaign_protocol.json,extension_protocol.json,degree_extension_protocol.json',
      'minimum_observed_primary_nu':min(primary(r)['nu'] for r in records) if records else None,
      'minimum_observed_primary_energy':min(primary(r)['energy'] for r in records) if records else None,
      'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'engine.py',HERE/'engine_v2.py',HERE/'engine_v3.py',HERE/'engine_v4.py',HERE/'constant_csr.cpp',HERE/'native_action.py',HERE/'run_campaign.py']}}
 if records:out['elapsed_primary_campaign_seconds']=max(r.get('finished_unix',0) for r in records)-min(r['started_unix'] for r in records)
 if len(dt)==len(checks):out['maximum_abs_dtcheck_relative_change']={k:max(abs(d['relative_changes'][k]) for d in dt) for k in ['nu','energy','d']}
 if len(diagnostics)==3:
  baseline=read(HERE/'runs/composition_f80_N1000_seed3001.json')
  native05=next(r for r in diagnostics if r['dt']==.05)
  p8=next(v for v in native05['prefixes'] if v['duration']==8000.)
  vals=[r['prefixes'][-1]['d'] for r in diagnostics]
  out['outlier_interpretation']={
   'same_dt_native_vs_SciPy_baseline_T8000_relative_d_change':p8['d']/primary(baseline)['d']-1,
   'T16000_dimension_range':[min(vals),max(vals)],
   'T16000_max_to_min_relative_spread':max(vals)/min(vals)-1,
   'verdict':'No converged invariant-covariance estimate established for this atypical N1000 network. Arithmetic/phase, record length and step dependence are entangled. Do not attribute the24%change solely to timestep or claim full numerical validation.',
   'scope_decision':'Retain all primary and diagnostic records; no further campaign; flag the outlier and finiteN1000precision limitation.'}
 session_evidence=HERE/'original_execution_session_completion.json'
 if session_evidence.exists():out['original_execution_session_completion']=read(session_evidence)
 path=HERE/'runtime_completion_summary.json';path.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:out[k] for k in ['status','primary_cases_expected','primary_cases_completed','time_step_checks_completed','tangent_checks_completed','maximum_observed_network_phase_aggregate_RSS_bytes']},indent=2))
 return out

if __name__=='__main__':summarize()
