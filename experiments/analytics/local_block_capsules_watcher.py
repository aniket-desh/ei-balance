"""Capture compact block local moments from verified completed simulation prefixes.

No dynamics are run. POSIX read-only mappings retain a trajectory after the
simulation worker unlinks its path. Only prefixes published after flush are
read. All summary arrays are updated atomically. Single-thread, chunked math.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import json,time,mmap,resource
from pathlib import Path
import numpy as np
from scipy.special import ndtr
ROOT=Path(__file__).resolve().parents[2]
RUNTIME=ROOT/'next_round/runtime'
OUT=Path(__file__).with_name('local_block_capsules');OUT.mkdir(exist_ok=True)
B=500.;CHUNK=128

def atomic_json(path,data):
    temp=path.with_suffix('.writing');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def save_capsule(name,arrays,job,verified):
    path=OUT/f'{name}.npz';temp=path.with_suffix('.writing')
    with temp.open('wb') as handle:
        np.savez_compressed(handle,**{k:np.stack(v) for k,v in arrays.items()},
            block_duration=np.array(B),sample_dt=np.array(job['config']['sample_dt']),
            verified_duration_at_capture=np.array(verified),
            fraction_e=np.array(job['config']['fraction_e']),n=np.array(job['config']['n']))
    temp.replace(path)

def capture_block(trajectory,k,ns):
    N=trajectory.shape[1]
    totals={key:np.zeros(N) for key in ['mean_x','second_x','mean_rate','second_rate','mean_gain','second_gain']}
    for start in range(k*ns,(k+1)*ns,CHUNK):
        stop=min(start+CHUNK,(k+1)*ns)
        x=np.array(trajectory[start:stop],dtype=np.float64)
        totals['mean_x']+=x.sum(0);totals['second_x']+=np.einsum('ij,ij->j',x,x)
        rates=2*ndtr(x)
        totals['mean_rate']+=rates.sum(0);totals['second_rate']+=np.einsum('ij,ij->j',rates,rates)
        del rates
        gains=np.sqrt(2/np.pi)*np.exp(-.5*x*x)
        totals['mean_gain']+=gains.sum(0);totals['second_gain']+=np.einsum('ij,ij->j',gains,gains)
        del x,gains
    try:trajectory._mmap.madvise(mmap.MADV_DONTNEED)
    except (AttributeError,ValueError):pass
    return {key:value/ns for key,value in totals.items()}

def main():
    jobs=json.loads((RUNTIME/'campaign_jobs.json').read_text())
    # Cover all still available v1 trajectories, retaining explicit missingness.
    states={j['name']:{'job':j,'mapped':None,'arrays':None,'status':'not_attached',
                          'blocks':0,'source_missing_observed':False} for j in jobs}
    started=time.time()
    while True:
        complete=0
        for name,state in states.items():
            progress=RUNTIME/'runs'/f'{name}.json'
            try:record=json.loads(progress.read_text())
            except (FileNotFoundError,json.JSONDecodeError):continue
            status=record.get('status');job=state['job'];nblocks=round(job['config']['duration']/B)
            if status in ['complete','failed']:complete+=1
            path=RUNTIME/'tmp'/f'{name}_trajectory.npy'
            if state['mapped'] is None and state['blocks']<nblocks:
                try:
                    state['mapped']=np.load(path,mmap_mode='r')
                    state['status']='attached'
                    if state['arrays'] is None:
                        state['arrays']={k:[] for k in ['mean_x','second_x','mean_rate','second_rate','mean_gain','second_gain']}
                    print(json.dumps({'event':'attached','name':name,'progress_status':status}),flush=True)
                except FileNotFoundError:
                    state['source_missing_observed']=True
                    if status in ['complete','failed']:state['status']='unavailable_before_attachment'
                    continue
                except (ValueError,OSError):continue
            if state['mapped'] is None:continue
            # recorded_duration is published only after traj.flush(). Analyzing
            # or complete also implies the full trajectory was written.
            verified=float(record.get('recorded_duration',0))
            if status in ['analyzing','complete']:verified=float(job['config']['duration'])
            ready=min(nblocks,int(verified//B));ns=round(B/job['config']['sample_dt'])
            changed=False
            while state['blocks']<ready:
                block=capture_block(state['mapped'],state['blocks'],ns)
                for key,value in block.items():state['arrays'][key].append(value)
                state['blocks']+=1;changed=True
            if changed:
                save_capsule(name,state['arrays'],job,verified)
                print(json.dumps({'event':'captured','name':name,'blocks':state['blocks'],'verified_duration':verified}),flush=True)
            if state['blocks']==nblocks:
                state['status']='captured_complete'
                state['mapped']._mmap.close();state['mapped']=None
        report={'started_unix':started,'updated_unix':time.time(),'campaign_jobs':len(jobs),
                'campaign_finished':complete,'block_duration':B,'chunk_samples':CHUNK,
                'verified_prefix_only':True,'new_simulations':0,
                'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'networks':{name:{'experiment':s['job']['experiment'],'status':s['status'],
                    'blocks':s['blocks'],'expected_blocks':round(s['job']['config']['duration']/B),
                    'source_missing_observed':s['source_missing_observed']} for name,s in states.items()}}
        atomic_json(OUT/'coverage.json',report)
        if complete==len(jobs):break
        time.sleep(2)
    print(json.dumps({'event':'finished','captured':sum(s['blocks']>0 for s in states.values()),
                      'full':sum(s['status']=='captured_complete' for s in states.values())}),flush=True)
if __name__=='__main__':main()
