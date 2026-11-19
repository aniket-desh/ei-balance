"""Selected midpoint tangent checks; not a phase diagram or basin estimate."""
from pathlib import Path
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']:os.environ[name]='1'
import sys,time,json,hashlib,resource,multiprocessing as mp,concurrent.futures as cf
import numpy as np
from scipy.special import ndtr
import engine_v2 as engine
from native_action import ConstantAction,build
HERE=Path(__file__).resolve().parent

def one(job):
 started=time.monotonic();c=engine.NetworkConfig(**job['config']);c.duration=2000.
 J,drive,x,labels,info=engine.make_network(c,np.array(job['mean_input']));S,B=engine.sparse_complement(J,labels,info)
 action=ConstantAction(S,B,labels,info,Path(job['native_library']));jhash=hashlib.sha256(J.tobytes()).hexdigest();del J
 rng=np.random.default_rng(c.seed+2000003);v=rng.normal(size=c.n);v/=np.linalg.norm(v)
 nb=round(c.burn/c.dt);ns=round(c.duration/c.dt);lyap=np.empty(ns);k=0
 sumr=np.zeros(c.n);sumr2=np.zeros(c.n);samples=0;stride=round(c.sample_dt/c.dt)
 for step in range(nb+ns):
  rx=2*ndtr(x);gx=np.sqrt(2/np.pi)*np.exp(-.5*x*x)
  mid=x+.5*c.dt*(-x+action(rx)+drive)
  vmid=v+.5*c.dt*(-v+action(gx*v))
  rm=2*ndtr(mid);gm=np.sqrt(2/np.pi)*np.exp(-.5*mid*mid)
  vn=v+c.dt*(-vmid+action(gm*vmid))
  x+=c.dt*(-mid+action(rm)+drive)
  norm=np.linalg.norm(vn);v=vn/norm
  if step>=nb:
   lyap[k]=np.log(norm)/c.dt;k+=1
   if (step-nb)%stride==0:sumr+=rx;sumr2+=rx*rx;samples+=1
 vr=sumr2/samples-(sumr/samples)**2
 out={'rate_variance_by_population':[float(vr[labels==a].mean()) for a in range(2)],'rate_mean_by_population':[float((sumr/samples)[labels==a].mean()) for a in range(2)],'case_name':job['name'],'config':vars(c),'connectivity_sha256':jhash,'tangent_initial_seed':c.seed+2000003,
      'largest_finite_time_exponent':float(lyap.mean()),'eight_duration_blocks':[float(a.mean()) for a in np.array_split(lyap,8)],
      'duration':c.duration,'burn':c.burn,'elapsed_seconds':time.monotonic()-started,
      'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
      'scope':'One tangent and one initial condition. A positive finite-time exponent supports the fluctuating chaotic branch; no attractor-exhaustiveness or basin-probability claim.'}
 engine.atomic(HERE/'checks'/(job['name']+'_tangent.json'),out)
 return out

if __name__=='__main__':
 cases=json.loads((HERE/'campaign_jobs.json').read_text());selected=[];lib,_=build()
 for job in cases:
  c=job['config'];take=(job['experiment']=='composition' and c['n']==2000 and c['seed']==3001) or(job['experiment']=='degree' and c['n']==2000 and c['seed']==4001 and c['degree_dispersion'] in [0,2])
  if take:job['native_library']=str(lib);selected.append(job)
 engine.atomic(HERE/'tangent_check_protocol.json',{'selection':'FourrepresentativeN2000casesselectedbeforetangentoutcomes','cases':[j['name'] for j in selected],'duration':2000.,'burn':500.,'dt':.05,'workers':4,'no_simulation_selection_on_Lyapunov':True})
 with cf.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
  for out in pool.map(one,selected):print(json.dumps(out),flush=True)
