"""Serial campaign phases ensure at most four network workers in total."""
from pathlib import Path
import time,json,subprocess,sys,hashlib
HERE=Path(__file__).resolve().parent

def atomic(path,data):
 p=path.with_suffix('.writing');p.write_text(json.dumps(data,indent=2)+'\n');p.replace(path)

def run_phase(label,manifest,workers):
 command=[sys.executable,str(HERE/'engine_v2.py'),'--jobs',str(HERE/manifest),'--workers',str(workers)]
 atomic(HERE/'continuation_status.json',{'phase':label,'started_unix':time.time(),'command':command})
 with (HERE/(label+'.log')).open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':
 initial=json.loads((HERE/'campaign_jobs.json').read_text())
 while True:
  results=[]
  for job in initial:
   p=HERE/'runs'/(job['name']+'.json')
   results.append(json.loads(p.read_text()) if p.exists() else {'status':'not_started'})
  if all(r['status']=='complete' for r in results):break
  if any(r['status']=='failed' for r in results):raise RuntimeError('Initial campaign failure; retained for review before continuation.')
  time.sleep(2)
 run_phase('extension_first','extension_first_jobs.json',4)
 large=[json.loads((HERE/'runs'/(j['name']+'.json')).read_text()) for j in json.loads((HERE/'extension_first_jobs.json').read_text()) if j['config']['n']==4000]
 rss=max(r['max_rss_bytes'] for r in large)
 audit=json.loads((HERE/'extension_first_jobs_resource_audit.json').read_text());budget=audit['memory_budget_bytes']
 workers=max(1,min(4,int(budget/(1.15*rss))))
 atomic(HERE/'large_network_memory_gate.json',{'two_N4000_measured_peak_worker_RSS_bytes':rss,'first_wave_maximum_observed_aggregate_RSS_bytes':audit['maximum_observed_aggregate_RSS_bytes'],'budget_bytes':budget,'remaining_large_workers':workers,'decision_uses_memory_only':True})
 run_phase('extension_remaining','extension_remaining_jobs.json',workers)
 run_phase('integration_checks','integration_check_jobs.json',4)
 with (HERE/'tangent_checks.log').open('w') as stream:subprocess.run([sys.executable,str(HERE/'tangent_checks.py')],stdout=stream,stderr=subprocess.STDOUT,check=True)
 atomic(HERE/'continuation_status.json',{'phase':'complete','finished_unix':time.time(),'all_extensions_and_selected_validation_checks_complete':True})
