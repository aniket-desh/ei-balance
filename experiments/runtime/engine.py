"""Portable memory-bounded EI simulation and exact streamed block estimators.

All dynamics are float64. Sparse complements plus a two-population rank term
represent exactly the original strict-Dale matrix. No connectivity is pruned.
"""
import os
for _name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_name]='1'
from pathlib import Path
import sys,time,json,hashlib,itertools,mmap,resource
import numpy as np
from scipy.special import ndtr
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'recovered/bundle/ei_full/ei_balance_handoff/src'))
from eicavity.network import NetworkConfig, make_network
HERE=Path(__file__).resolve().parent


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    p=path.with_suffix('.writing');p.write_text(json.dumps(data,indent=2)+'\n');p.replace(path)


def jsonable(obj):
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,np.generic):return obj.item()
    if isinstance(obj,dict):return {k:jsonable(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [jsonable(v) for v in obj]
    return obj


def sparse_complement(J,labels,info):
    S=J.copy();B=np.zeros((2,2));N=len(labels)
    for a in range(2):
        ia=np.flatnonzero(labels==a)
        for b in range(2):
            if info['probabilities'][a,b]>.5:
                ib=np.flatnonzero(labels==b)
                value=info['nonzero_weights_scaled'][a,b]*(1 if b==0 else -1)/np.sqrt(N)
                S[np.ix_(ia,ib)]-=value;B[a,b]=value
    return csr_matrix(S),B


class SparseAction:
    def __init__(self,S,B,ne):self.S=S;self.B=B;self.ne=ne
    def __call__(self,r):
        if r.ndim==1:
            v=self.B@np.array([r[:self.ne].sum(),r[self.ne:].sum()])
            out=self.S@r;out[:self.ne]+=v[0];out[self.ne:]+=v[1];return out
        v=self.B@np.stack([r[:self.ne].sum(0),r[self.ne:].sum(0)])
        out=self.S@r;out[:self.ne]+=v[0];out[self.ne:]+=v[1];return out


def step(x,drive,action,dt):
    mid=x+.5*dt*(-x+action(2*ndtr(x))+drive)
    return x+dt*(-mid+action(2*ndtr(mid))+drive)


def discard_map(array):
    try:array._mmap.madvise(mmap.MADV_DONTNEED)
    except (AttributeError,ValueError):pass


def bench():
    from scipy.linalg.blas import dgemv
    out={'threadpools':None,'cases':[]}
    from threadpoolctl import threadpool_info
    out['threadpools']=threadpool_info()
    for N,fe in [(1000,.8),(2000,.6),(4000,.8)]:
        f=np.array([fe,1-fe]);scale=np.array([.8,.2])/f
        config=NetworkConfig(n=N,seed=99001,fraction_e=fe,
             variance_profile=(np.array([[16,9],[25,12]])*scale).tolist(),
             mean_connectivity=(np.array([[1,-6],[2,-10]])*scale).tolist())
        J,drive,x,labels,info=make_network(config,np.array([-2.6712782365825896,-3.2189749212858354]))
        S,B=sparse_complement(J,labels,info);action=SparseAction(S,B,sum(labels==0))
        rng=np.random.default_rng(99002);vectors=rng.normal(size=(N,4))
        error=float(np.max(np.abs(J@vectors-action(vectors))))
        xx=x.copy();yy=x.copy()
        for _ in range(50):xx=step(xx,drive,lambda v:J@v,.05);yy=step(yy,drive,action,.05)
        trajectory_error=float(np.max(np.abs(xx-yy)))
        timings={}
        for name,fn in [('dense1',lambda v:J@v),('sparse_complement',action)]:
            t=time.monotonic();z=x.copy()
            for _ in range(300):z=step(z,drive,fn,.05)
            timings[name]=(time.monotonic()-t)/300
        if N==2000:
            for threads in [2,4]:
                with threadpool_limits(limits=threads):
                    t=time.monotonic();z=x.copy()
                    for _ in range(300):z=step(z,drive,lambda v:J@v,.05)
                    timings[f'dense{threads}']=(time.monotonic()-t)/300
        row={'N':N,'fe':fe,'nnz_fraction':S.nnz/N**2,'dense_bytes':J.nbytes,
             'sparse_bytes':S.data.nbytes+S.indices.nbytes+S.indptr.nbytes,
             'max_action_error':error,'max_50step_trajectory_error':trajectory_error,
             'seconds_per_midpoint_step':timings,
             'estimated_T8000_sim_seconds':timings['sparse_complement']*170000}
        out['cases'].append(row);print(json.dumps(row),flush=True)
    atomic(HERE/'optimization_benchmark.json',out)


def analyze(traj,labels,sample_dt=.25,B=500.,prefixes=(4000.,8000.),name='network'):
    """Exact four-distinct-window average; covariance arrays streamed via disk.

    Population energies are divided by total N and include diagonal terms.
    Total E = E_EE + 2 E_EI + E_II. Every matching over four windows is retained.
    """
    start=time.monotonic();ns=round(B/sample_dt);nb=traj.shape[0]//ns;N=traj.shape[1]
    groups=[np.flatnonzero(labels==a) for a in range(2)];Na=np.array([len(g) for g in groups])
    means=np.empty((nb,N));diag=np.empty((nb,N));Cpath=HERE/'tmp'/f'{name}_B{int(B)}_cov.npy'
    cs=np.lib.format.open_memmap(Cpath,mode='w+',dtype=np.float64,shape=(nb,N,N))
    cross=np.zeros((2,2,nb,nb));pairs=list(itertools.combinations(range(nb),2));pp=np.array(pairs)
    # First pass: block rate means, then compact difference vectors.
    for k in range(nb):
        r=2*ndtr(np.asarray(traj[k*ns:(k+1)*ns],float));means[k]=r.mean(0)
    D=means[pp[:,0]]-means[pp[:,1]]
    Q=np.empty((2,2,nb,len(pairs)));Z=np.empty((nb,N,2));A=np.empty((nb,2,2))
    U=np.zeros((N,2));U[groups[0],0]=1/np.sqrt(Na[0]);U[groups[1],1]=1/np.sqrt(Na[1])
    local={'mean_x':np.zeros(N),'second_x':np.zeros(N),'mean_rate':np.zeros(N),
           'second_rate':np.zeros(N),'mean_gain':np.zeros(N)}
    gain2time=np.zeros(N);last_drift=None
    for k in range(nb):
        x=np.asarray(traj[k*ns:(k+1)*ns],float);r=2*ndtr(x);C=r.T@r/ns-np.outer(means[k],means[k]);diag[k]=np.diag(C)
        cs[k]=C;cs.flush();Z[k]=C@U;A[k]=U.T@Z[k]
        for a in range(2):
            for b in range(2):
                Ca=C[np.ix_(groups[a],groups[b])]
                Q[a,b,k]=np.einsum('pn,pn->p',D[:,groups[a]]@Ca,D[:,groups[b]])/N
        local['mean_x']+=x.sum(0);local['second_x']+=np.square(x).sum(0)
        local['mean_rate']+=r.sum(0);local['second_rate']+=np.square(r).sum(0)
        gain=np.sqrt(2/np.pi)*np.exp(-.5*x*x);local['mean_gain']+=gain.sum(0);gain2time+=np.square(gain).sum(0)
        del C,r,x,gain;discard_map(cs)
    nt=nb*ns
    for key in local:local[key]/=nt
    vm=local['second_rate']-np.square(local['mean_rate']);qx=local['second_x']-np.square(local['mean_x'])
    poplocal=[]
    for a,g in enumerate(groups):
        gm=local['mean_gain'][g];mu=local['mean_x'][g]
        poplocal.append({'population':int(a),'count':len(g),'mean_rate':float(local['mean_rate'][g].mean()),
             'h':float(mu.mean()),'Delta_input_means':float(mu.var()),'q_temporal_input':float(qx[g].mean()),
             'nu_temporal_rate':float(vm[g].mean()),'mean_v_squared':float(np.square(vm[g]).mean()),
             'A_timeaveraged_gain_squared':float(np.square(gm).mean()),
             'A4_timeaveraged_gain_fourth':float(np.power(gm,4).mean()),
             'instantaneous_gain_squared':float((gain2time[g]/nt).mean()),
             'variance_temporal_q_across_neurons':float(qx[g].var())})
    # Conditional local-law curve, all neurons; bins are diagnostic empirical labels.
    conditional=[]
    for a,g in enumerate(groups):
        order=g[np.argsort(local['mean_x'][g])]
        for b,ix in enumerate(np.array_split(order,12)):
            conditional.append({'population':a,'bin':b,'count':len(ix),
                'mu':float(local['mean_x'][ix].mean()),'temporal_input_q':float(qx[ix].mean()),
                'mean_rate':float(local['mean_rate'][ix].mean()),'rate_variance':float(vm[ix].mean()),
                'mean_gain':float(local['mean_gain'][ix].mean()),
                'mean_gain_squared':float(np.square(local['mean_gain'][ix]).mean())})
    # Independently specified deterministic neuron sample for temporal-kernel checks.
    from scipy.fft import rfft,irfft,next_fast_len
    lag_values=np.array([0,1,2,4,8,16,32,64,128,256]);kernels=[]
    for a,g in enumerate(groups):
        selected=g[np.linspace(0,len(g)-1,min(96,len(g)),dtype=int)]
        acx=np.zeros(len(lag_values));acr=np.zeros(len(lag_values));nf=next_fast_len(2*nt)
        for first in range(0,len(selected),16):
            xx=np.array(traj[:nt,selected[first:first+16]],dtype=float)
            rr=2*ndtr(xx)
            xx-=xx.mean(0);rr-=rr.mean(0)
            fx=rfft(xx,n=nf,axis=0);fr=rfft(rr,n=nf,axis=0)
            acx+=irfft(fx*np.conj(fx),n=nf,axis=0)[lag_values].sum(1)/(nt-lag_values)
            acr+=irfft(fr*np.conj(fr),n=nf,axis=0)[lag_values].sum(1)/(nt-lag_values)
        kernels.append({'population':a,'sampled_neurons':selected.tolist(),'lags':(lag_values*sample_dt).tolist(),
            'input_centered_kernel':(acx/len(selected)).tolist(),'rate_centered_kernel':(acr/len(selected)).tolist(),
            'scope':'Fixed-index neuron subsample, whole-record centering, diagnostic not exact unbiased finite-window estimate.'})
    # Chunk spatial rows: no nb*N*N resident covariance tensor.
    for a,ga in enumerate(groups):
        for b,gb in enumerate(groups):
            for first in range(0,len(ga),32):
                rows=ga[first:first+32]
                chunk=np.asarray(cs[:,rows[:,None],gb[None,:]],float).reshape(nb,-1)
                cross[a,b]+=chunk@chunk.T/N
                del chunk
            discard_map(cs)
    DD=[D[:,g]@D[:,g].T for g in groups];index={p:i for i,p in enumerate(pairs)}
    nu_pair=np.stack([.5*(diag[pp[:,0]][:,g].mean(1)+diag[pp[:,1]][:,g].mean(1))+.5*np.square(D[:,g]).mean(1) for g in groups],axis=1)
    Zdiff=.5*(Z[pp[:,0]]+Z[pp[:,1]])+.5*D[:,:,None]*(D@U)[:,None,:]
    Adiff=np.einsum('na,pnb->pab',U,Zdiff)
    zflat=Zdiff.reshape(len(pairs),-1);aflat=Adiff.reshape(len(pairs),-1)
    ecoh=(2*(zflat@zflat.T)-(aflat@aflat.T))/N
    def pairenergy(p,q):
        aa,bb=pairs[p];cc,dd=pairs[q]
        e=.25*(cross[:,:,aa,cc]+cross[:,:,aa,dd]+cross[:,:,bb,cc]+cross[:,:,bb,dd]+Q[:,:,cc,p]+Q[:,:,dd,p]+Q[:,:,aa,q]+Q[:,:,bb,q])
        for a in range(2):
            for b in range(2):e[a,b]+=.25*DD[a][p,q]*DD[b][p,q]/N
        return e
    p,q=index[(0,1)],index[(2,3)]
    # Literal dense covariance check on first four windows validates contractions.
    dense0=.5*(np.array(cs[0])+np.array(cs[1]))+.5*np.outer(D[p],D[p]);dense1=.5*(np.array(cs[2])+np.array(cs[3]))+.5*np.outer(D[q],D[q])
    dense=np.array([[np.sum(dense0[np.ix_(ga,gb)]*dense1[np.ix_(ga,gb)])/N for gb in groups] for ga in groups])
    denseerror=float(np.max(np.abs(pairenergy(p,q)-dense)))
    np.testing.assert_allclose(pairenergy(p,q),dense,rtol=5e-11,atol=5e-12)
    del dense0,dense1;discard_map(cs)
    result={'block_duration':B,'nblocks':nb,'N':N,'population_energy_normalization':'sum entries in a,b divided by total N; off-diagonal E-I block counted twice in total',
            'population_local_stats':poplocal,'conditional_local_bins':conditional,'temporal_kernels':kernels,'dense_formula_error':denseerror,'prefixes':[]}
    for duration in prefixes:
        k=round(duration/B)
        if k>nb or k<4:continue
        row={'duration':duration,'gaps':[]}
        for gap in [0,1]:
            qs=[q for q in itertools.combinations(range(k),4) if min(np.diff(q))>=gap+1]
            if not qs:continue
            energies=[];coh=[]
            for aa,bb,cc,dd in qs:
                for m in [((aa,bb),(cc,dd)),((aa,cc),(bb,dd)),((aa,dd),(bb,cc))]:
                    p,q=index[m[0]],index[m[1]];energies.append(pairenergy(p,q));coh.append(ecoh[p,q])
            energy=np.mean(energies,axis=0);total=energy.sum()
            ii=[i for i,(aa,bb) in enumerate(pairs) if bb<k and bb-aa>=gap+1]
            nu=nu_pair[ii].mean(0);nutotal=float(Na@nu/N)
            ix=np.triu_indices(k,k=gap+1);classic=np.array([[cross[a,b][ix].mean() for b in range(2)] for a in range(2)])
            row['gaps'].append({'empty_blocks':gap,'quadruples':len(qs),'population_energy':energy.tolist(),
                'energy':float(total),'nu_by_population':nu.tolist(),'nu':nutotal,'d':nutotal**2/total,
                'classic_energy':float(classic.sum()),'classic_population_energy':classic.tolist(),
                'coherent_projection_energy':float(np.mean(coh)),
                'coherent_projection_fraction':float(np.mean(coh)/total)})
        result['prefixes'].append(row)
    result['elapsed_seconds']=time.monotonic()-start
    del cs;Cpath.unlink(missing_ok=True)
    return result


def run_one(job):
    started=time.monotonic();name=job['name'];path=HERE/'runs'/f'{name}.json';c=NetworkConfig(**job['config'])
    dest=HERE/'tmp'/f'{name}_trajectory.npy'
    r={'name':name,'job':job,'status':'running','started_unix':time.time()};atomic(path,r)
    try:
        J,drive,x,labels,info=make_network(c,np.array(job['mean_input']));S,B=sparse_complement(J,labels,info)
        action=SparseAction(S,B,sum(labels==0));r['connectivity_sha256']=hashlib.sha256(J.tobytes()).hexdigest();del J
        r['network_info']=jsonable(info);r['sparse_nnz']=S.nnz
        nt=round(c.duration/c.sample_dt);traj=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float32,shape=(nt,c.n))
        nb=round(c.burn/c.dt);ns=round(c.duration/c.dt);stride=round(c.sample_dt/c.dt);sample=0
        for i in range(nb+ns):
            x=step(x,drive,action,c.dt)
            if i>=nb and (i-nb+1)%stride==0:
                traj[sample]=x;sample+=1
                if sample%8000==0:
                    traj.flush();r.update(recorded_duration=sample*c.sample_dt,elapsed_seconds=time.monotonic()-started);atomic(path,r)
        r['simulation_seconds']=time.monotonic()-started
        drift=-x+action(2*ndtr(x))+drive
        r['final_drift_rms']=float(np.sqrt(np.mean(drift*drift)))
        r['state_description']='Persistent temporal fluctuations unless vanishing variance and drift; Lyapunov sign not measured by this runner.'
        del S,action,drive,x
        r['estimators']=[]
        for block in job.get('blocks',[500.]):
            r['estimators'].append(analyze(traj,labels,c.sample_dt,block,job.get('prefixes',[4000.,8000.]),name))
            r.update(status='analyzing',elapsed_seconds=time.monotonic()-started);atomic(path,r)
        r.update(status='complete',elapsed_seconds=time.monotonic()-started,finished_unix=time.time(),max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        atomic(path,r);del traj
        if not job.get('retain_temporary_trajectory',False):dest.unlink(missing_ok=True)
        return {'name':name,'status':'complete','elapsed_seconds':r['elapsed_seconds'],'simulation_seconds':r['simulation_seconds'],'max_rss_bytes':r['max_rss_bytes']}
    except BaseException as exc:
        r.update(status='failed',exception=repr(exc),elapsed_seconds=time.monotonic()-started);atomic(path,r);raise


if __name__=='__main__':
    import argparse,multiprocessing as mp,concurrent.futures as cf
    p=argparse.ArgumentParser();p.add_argument('--benchmark',action='store_true');p.add_argument('--jobs',type=Path);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if args.benchmark:bench()
    elif args.jobs:
        jobs=json.loads(args.jobs.read_text())
        import psutil
        available=psutil.virtual_memory().available
        memory_budget=min(4*1024**3,int(.4*available))
        maxn=max(j['config']['n'] for j in jobs)
        # Float64 dense generation, covariance work arrays, row chunks and FFT batches.
        estimated_worker_peak=192*1024**2+24*maxn**2+32*maxn*2000
        workers=max(1,min(args.workers,4,max(1,(os.cpu_count() or 1)-2),memory_budget//estimated_worker_peak))
        audit={'platform':sys.platform,'cpu_count':os.cpu_count(),'available_memory_bytes':available,
            'memory_budget_bytes':memory_budget,'estimated_worker_peak_bytes':estimated_worker_peak,
            'workers':workers,'blas_threads_per_worker':1,'multiprocessing_start_method':'spawn',
            'temporary_disk_ceiling_bytes':20*1024**3,'maximum_observed_aggregate_RSS_bytes':0,
            'temporary_raw_trajectories_are_not_deliverables':True}
        atomic(HERE/'resource_audit.json',audit)
        with cf.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
            futures={pool.submit(run_one,j):j['name'] for j in jobs}
            pending=set(futures)
            while pending:
                done,pending=cf.wait(pending,timeout=2,return_when=cf.FIRST_COMPLETED)
                processes=psutil.Process().children(recursive=True)
                rss=sum(p.memory_info().rss for p in processes if p.is_running())+psutil.Process().memory_info().rss
                audit['maximum_observed_aggregate_RSS_bytes']=max(audit['maximum_observed_aggregate_RSS_bytes'],rss)
                audit['updated_unix']=time.time();audit['remaining_jobs']=len(pending)
                atomic(HERE/'resource_audit.json',audit)
                if rss>memory_budget:
                    audit['stopped_for_memory_limit']=True;atomic(HERE/'resource_audit.json',audit)
                    for process in processes:process.terminate()
                    raise MemoryError('Configured aggregate memory limit reached; completed records retained.')
                for f in done:print(json.dumps(f.result()),flush=True)
