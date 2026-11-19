"""Portable launcher; default SciPy backend requires no C++ compiler.

Usage: python run_campaign.py --jobs campaign_jobs.json --workers 4
Use --native to compile the optional local constant-weight sparse kernel.
"""
import os
for _thread_name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']:
 os.environ[_thread_name]='1'
from pathlib import Path
import argparse,json,subprocess,sys,shutil
import psutil
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,default=HERE/'campaign_jobs.json');p.add_argument('--workers',type=int,default=4);p.add_argument('--native',action='store_true');p.add_argument('--rerun-complete',action='store_true');args=p.parse_args()
 source=args.jobs if args.jobs.is_absolute() else Path.cwd()/args.jobs
 jobs=json.loads(source.read_text());library=None
 if args.native:
  from native_action import build
  try:library,_=build()
  except (RuntimeError,subprocess.CalledProcessError) as exc:print('Native build unavailable; using SciPy:',exc,file=sys.stderr)
 selected=[]
 for job in jobs:
  result=HERE/'runs'/(job['name']+'.json')
  if result.exists() and not args.rerun_complete and json.loads(result.read_text()).get('status')=='complete':continue
  job.pop('native_library',None)
  if library is not None:job['native_library']=str(library)
  selected.append(job)
 if not selected:print('Every selected job already has a complete saved result.');return
 maxn=max(j['config']['n'] for j in selected)
 estimated_one=192*1024**2+24*maxn**2+32*maxn*2000
 if sys.platform=='darwin':estimated_one=max(estimated_one,384*1024**2+64*maxn**2)
 memory_budget=min(4*1024**3,int(.4*psutil.virtual_memory().available))
 if estimated_one>memory_budget:
  raise SystemExit('One worker exceeds the conservative memory budget. Close other memory-heavy apps or choose a smaller network before running.')
 disk_budget=min(20*1024**3,shutil.disk_usage(HERE).free//2)
 per_job_disk=max(int(j['config']['duration']/j['config']['sample_dt'])*j['config']['n']*4+int(j['config']['duration']/min(j.get('blocks',[500.])))*j['config']['n']**2*8 for j in selected)
 retained_disk=sum(int(j['config']['duration']/j['config']['sample_dt'])*j['config']['n']*4 for j in selected if j.get('retain_temporary_trajectory'))
 disk_workers=(disk_budget-retained_disk)//per_job_disk
 if disk_workers<1:raise SystemExit('Requested recordings exceed the conservative scratch-disk budget. Reduce retained trajectories, duration, or network size.')
 args.workers=min(args.workers,int(disk_workers))
 (HERE/'tmp').mkdir(exist_ok=True)
 planned=HERE/'tmp'/'local_selected_jobs.json';planned.write_text(json.dumps(selected,indent=2)+'\n')
 subprocess.run([sys.executable,str(HERE/'engine_v4.py'),'--jobs',str(planned),'--workers',str(args.workers)],check=True,cwd=HERE.parents[1])

if __name__=='__main__':main()
