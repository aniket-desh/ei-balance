"""Capture compact diagnostics as the fixed prospective campaign completes.

This watcher runs no simulations, uses one BLAS thread, and skips existing
outputs. It stops when all forty planned composition jobs are complete or at
its wall limit. The runtime runner owns all heavy compute and resource gates.
"""
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,time
from static_source_diagnosis import analyze as static_analyze
from selected_local_addon import analyze as local_analyze
from aggregate_static_sources import main as static_aggregate
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    started=time.monotonic();dest=HERE/'diagnostic_watcher_status.json'
    while True:
        total=complete=new=0
        for path in sorted((ROOT/'next_round/runtime/runs').glob('composition*.json')):
            d=json.loads(path.read_text());total+=1
            if d['status']!='complete':continue
            complete+=1;name=d['name']
            if not(HERE/f'{name}_static_source.json').exists():
                r=static_analyze(path)
                if r:new+=1;print(json.dumps({'static_complete':name}),flush=True)
            if not(HERE/f'{name}_local_addon.json').exists():
                # v1 onlyhasretainedseed3001records; v2compactcapsulescovereveryextension.
                r=local_analyze(path)
                if r:new+=1;print(json.dumps({'local_complete':name}),flush=True)
        if new:static_aggregate()
        status={'status':'running','complete_composition_jobs':complete,'started_composition_jobs':total,'new_diagnostic_files_this_pass':new,'elapsed_seconds':time.monotonic()-started,'no_simulations_run':True,'worker_threads':1}
        if complete==40:status['status']='complete'
        if time.monotonic()-started>1800:status['status']='stopped_at_wall_limit'
        dest.write_text(json.dumps(status,indent=2)+'\n')
        if status['status']!='running':break
        time.sleep(20)


if __name__=='__main__':main()
