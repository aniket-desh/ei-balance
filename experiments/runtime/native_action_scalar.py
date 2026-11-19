"""Optional constant-block sparse action with a portable SciPy fallback.

Compile only on explicit request; no fast-math or multithreading. The native
kernel exploits four constant edge weights without loading one value per edge.
"""
import ctypes,subprocess,shutil,sys,hashlib,json,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def build():
 compiler=shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
 if compiler is None:raise RuntimeError('No C++ compiler; use documented SciPy fallback.')
 library=HERE/'tmp'/('constant_csr.dylib' if sys.platform=='darwin' else 'constant_csr.so')
 flags=['-O3','-std=c++11','-fPIC','-shared']
 command=[compiler,*flags,str(HERE/'constant_csr.cpp'),'-o',str(library)]
 subprocess.run(command,check=True,capture_output=True,text=True)
 return library,command

class ConstantAction:
 def __init__(self,S,B,labels,info,library):
  self.n=len(labels);self.ne=int(np.sum(labels==0));self.ptr=np.asarray(S.indptr,dtype=np.int32);self.ix=np.asarray(S.indices,dtype=np.int32)
  self.split=np.array([self.ptr[i]+np.searchsorted(self.ix[self.ptr[i]:self.ptr[i+1]],self.ne) for i in range(self.n)],dtype=np.int32)
  weights=np.asarray(info['nonzero_weights_scaled'])*np.array([1.,-1.])[None,:]/np.sqrt(self.n)
  self.weights=np.asarray(np.where(info['probabilities']>.5,-weights,weights),dtype=np.float64,order='C')
  self.rank=np.asarray(B,dtype=np.float64,order='C');self.lib=ctypes.CDLL(str(library));self.f=self.lib.ei_action
  self.f.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*7;self.f.restype=None
  self.fixed=[a.ctypes.data for a in [self.ptr,self.ix,self.split,self.weights,self.rank]]
 def __call__(self,x):
  assert x.dtype==np.float64 and x.flags.c_contiguous and x.shape==(self.n,)
  y=np.empty(self.n);self.f(self.n,self.ne,*self.fixed,x.ctypes.data,y.ctypes.data);return y

if __name__=='__main__':
 import engine
 lib,command=build();out={'build_command':command,'fast_math':False,'cases':[]}
 for n,fe in [(1000,.8),(2000,.6),(4000,.8),(4000,.6)]:
  scale=np.array([.8,.2])/np.array([fe,1-fe])
  c=engine.NetworkConfig(n=n,seed=9923,fraction_e=fe,variance_profile=(np.array([[16,9],[25,12]])*scale).tolist(),mean_connectivity=(np.array([[1,-6],[2,-10]])*scale).tolist())
  J,drive,x,labels,info=engine.make_network(c,np.array([-2.6712782365825896,-3.2189749212858354]));S,B=engine.sparse_complement(J,labels,info);sci=engine.SparseAction(S,B,sum(labels==0));native=ConstantAction(S,B,labels,info,lib)
  rng=np.random.default_rng(9924);v=rng.normal(size=n);err=float(np.max(np.abs(J@v-native(v))))
  a=x.copy();b=x.copy()
  for _ in range(50):a=engine.step(a,drive,sci,.05);b=engine.step(b,drive,native,.05)
  trajerr=float(np.max(np.abs(a-b)));timings={}
  for name,fn in [('scipy_sparse',sci),('constant_native',native)]:
   t=time.monotonic();z=x.copy()
   for _ in range(500):z=engine.step(z,drive,fn,.05)
   timings[name]=(time.monotonic()-t)/500
  row={'N':n,'fe':fe,'max_action_error':err,'trajectory50error':trajerr,'seconds_per_midpoint_step':timings,'estimated_T8000_seconds':timings['constant_native']*170000}
  out['cases'].append(row);print(json.dumps(row),flush=True)
 engine.atomic(HERE/'native_optimization_benchmark.json',out)
