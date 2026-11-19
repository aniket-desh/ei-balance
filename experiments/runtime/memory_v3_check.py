import os
for x in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[x]='1'
from pathlib import Path
import json,time,resource,sys,hashlib
import numpy as np
import engine_v3
HERE=Path(__file__).resolve().parent
name='composition_f80_N4000_seed3001';baseline=json.loads((HERE/'runs'/(name+'.json')).read_text())['estimators'][0]
x=np.load(HERE/'tmp'/(name+'_trajectory.npy'),mmap_mode='r');labels=np.r_[np.zeros(3200,int),np.ones(800,int)]
start=time.monotonic();result=engine_v3.analyze(x,labels,name='memorycheck_'+name)
checks={}
def cmp(a,b,path):
 if isinstance(a,dict):
  for k in a:
   if k not in ['elapsed_seconds','neuron_statistics_file']:cmp(a[k],b[k],path+'/'+k)
 elif isinstance(a,list):
  for i,(u,v) in enumerate(zip(a,b)):cmp(u,v,path+f'/{i}')
 elif isinstance(a,(int,float)):
  checks[path]=abs(a-b)
 else:assert a==b,(path,a,b)
cmp(baseline,result,'')
out={'case':name,'numeric_fields_checked':len(checks),'maximum_absolute_numeric_difference':max(checks.values()),'all_numeric_fields_bitwise_equal':all(v==0 for v in checks.values()),'max_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'elapsed_seconds':time.monotonic()-start,'v3_source_sha256':hashlib.sha256(Path(engine_v3.__file__).read_bytes()).hexdigest(),'changes':'Release raw memmap pages after first means and FFT passes; in-place covariance scaling/subtraction; release written covariance pages immediately.'}
engine_v3.atomic(HERE/'memory_v3_validation.json',out);print(json.dumps(out,indent=2));assert out['maximum_absolute_numeric_difference']<1e-12
