"""Continue after the unchanged first wave; apply verified memory-only v3 fix."""
from pathlib import Path
import time,json,subprocess,sys,hashlib,psutil
HERE=Path(__file__).resolve().parent

def atomic(path,data):
 p=path.with_suffix('.writing');p.write_text(json.dumps(data,indent=2)+'\n');p.replace(path)

def run_phase(label,manifest,workers):
 command=[sys.executable,str(HERE/'engine_v3.py'),'--jobs',str(HERE/manifest),'--workers',str(workers)]
 atomic(HERE/'continuation_status.json',{'phase':label,'started_unix':time.time(),'command':command,'memory_optimization':'v3 verified identical statistics, lower resident maps'})
 with (HERE/(label+'.log')).open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':
 first=json.loads((HERE/'extension_first_jobs.json').read_text())
 while True:
  results=[]
  for job in first:
   p=HERE/'runs'/(job['name']+'.json');results.append(json.loads(p.read_text()) if p.exists() else {'status':'not_started'})
  if all(r['status']=='complete' for r in results):break
  if any(r['status']=='failed' for r in results):raise RuntimeError('First extension failure retained for review.')
  time.sleep(2)
 validation=json.loads((HERE/'memory_v3_validation.json').read_text());assert validation['all_numeric_fields_bitwise_equal']
 assert hashlib.sha256((HERE/'engine_v3.py').read_bytes()).hexdigest()==validation['v3_source_sha256']
 budget=min(4*1024**3,int(.4*psutil.virtual_memory().available))
 conservative_worker=int(validation['max_RSS_bytes']*1.3)
 workers=max(1,min(4,budget//conservative_worker))
 atomic(HERE/'large_network_memory_gate.json',{'measured_v3_N4000_analysis_RSS_bytes':validation['max_RSS_bytes'],'safety_multiplier':1.3,'budget_bytes':budget,'remaining_large_workers':workers,'decision_uses_memory_only':True,'v3_exact_statistic_match_fields':validation['numeric_fields_checked']})
 run_phase('extension_remaining','extension_remaining_jobs.json',workers)
 run_phase('integration_checks','integration_check_jobs.json',4)
 with (HERE/'tangent_checks.log').open('w') as stream:subprocess.run([sys.executable,str(HERE/'tangent_checks.py')],stdout=stream,stderr=subprocess.STDOUT,check=True)
 atomic(HERE/'continuation_status.json',{'phase':'complete','finished_unix':time.time(),'all_extensions_and_selected_validation_checks_complete':True})
