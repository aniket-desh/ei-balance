"""Replay fixed graphs with two local workers. Only the new seed gets a full covariance analysis."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent/'runtime'
sys.path.insert(0,str(RUNTIME))
for variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[variable]='1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--one', help=argparse.SUPPRESS)
    parser.add_argument('--library', help=argparse.SUPPRESS)
    args=parser.parse_args()
    jobs=json.loads((HERE/'protocol.json').read_text())['jobs']
    for directory in ('runs','tmp','measurements'):
        (HERE/directory).mkdir(exist_ok=True)
    if args.one:
        import engine_v4 as engine
        engine.HERE=HERE
        selected=next(j for j in jobs if j['name']==args.one)
        result=HERE/'runs'/(args.one+'.json')
        traj=HERE/'tmp'/(args.one+'_trajectory.npy')
        if not (result.exists() and traj.exists() and json.loads(result.read_text())['status']=='complete'):
            job={'name':args.one,'config':selected['config'],'mean_input':selected['mean_input'],
                 'blocks':[500.] if selected['full_ensemble_record'] is None else [], 'prefixes':[4000.,8000.],
                 'retain_temporary_trajectory':True,'native_library':args.library}
            engine.run_one(job)
        from measure import measure_case
        measure_case(selected,traj)
        traj.unlink()
        print(args.one+' complete',flush=True)
        return
    # A second supervisor in this checkout would exceed the shared two-worker cap.
    campaign_lock=(HERE/'tmp'/'supervisor.lock').open('a')
    try:
        fcntl.flock(campaign_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('The readout pilot is already running in this directory.')
    # Reuse the Mac binary that produced the 6101 dense recordings when present.
    reference = HERE.parent/'dense_sweeps/runs/dense_composition_0.8_N2000_seed6101.json'
    original=json.loads(reference.read_text()) if reference.exists() else {}
    candidate=Path(original.get('job',{}).get('native_library',''))
    if candidate.is_file() and hashlib.sha256(candidate.read_bytes()).hexdigest()==original.get('native_library_sha256'):
        library=candidate
    else:
        from native_action import build
        library,_=build(HERE/'tmp'/'kernel')
    pending=[j for j in jobs if not (HERE/'measurements'/(j['name']+'.json')).is_file()]
    active={}; high=0; failed=[]; start=time.time()
    try:
        while pending or active:
            while pending and len(active)<2:
                name=pending.pop(0)['name']
                proc=subprocess.Popen([sys.executable,__file__,'--one',name,'--library',str(library)])
                active[proc]=name;print('started '+name,flush=True)
            rss=psutil.Process().memory_info().rss
            for proc,name in list(active.items()):
                code=proc.poll()
                if code is not None:
                    if code: failed.append(name)
                    del active[proc]
                else:
                    try: rss+=psutil.Process(proc.pid).memory_info().rss
                    except psutil.NoSuchProcess: pass
            high=max(high,rss)
            if rss>4*1024**3: raise MemoryError('Readout worker RSS exceeded the 4 GiB cap.')
            if active: time.sleep(1)
    finally:
        for proc in active:
            if proc.poll() is None:proc.terminate()
            proc.wait()
        audit={'start_unix':start,'end_unix':time.time(),'max_rss_bytes':high,'cap_bytes':4*1024**3,
               'worker_limit':2,'failed':failed,'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()}
        (HERE/'runs'/f'audit_{os.getpid()}.json').write_text(json.dumps(audit,indent=2)+'\n')
    if failed: raise SystemExit(f'Failed readouts: {failed}')


if __name__=='__main__': main()
