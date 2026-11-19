"""Exact finite-count cumulant diagnostic, not a PR correction."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import betabinom,binom
HERE=Path(__file__).resolve().parent

def cumulants(M,p,chi):
 if chi<=1:
  L=round(chi*M);base=(M-L)*p;frac=base-np.floor(base)
  variance=L*p*(1-p)+frac*(1-frac)
  third=L*p*(1-p)*(1-2*p)+frac*(1-frac)*(1-2*frac)
  return variance,third
 s=(M-chi)/(chi-1)
 return M*p*(1-p)*chi,M*p*(1-p)*(1-2*p)*(s+M)*(s+2*M)/((s+1)*(s+2))

if __name__=='__main__':
 checks=[]
 for M in [17,80,800,3200]:
  for p in [1/17,.8]:
   for chi in [0.,.5,1.,1.25,2.,3.5]:
    k=np.arange(M+1)
    if chi<=1:
     L=round(chi*M);base=(M-L)*p;integer=int(np.floor(base));f=base-integer
     prob=(1-f)*binom.pmf(k-integer,L,p)+f*binom.pmf(k-integer-1,L,p)
    else:
     s=(M-chi)/(chi-1);prob=betabinom.pmf(k,M,p*s,(1-p)*s)
    prob/=prob.sum();v=float(np.sum((k-M*p)**2*prob));t=float(np.sum((k-M*p)**3*prob));vv,tt=cumulants(M,p,chi)
    assert abs(v-vv)<1e-6 and abs(t-tt)<2e-5
    checks.append({'M':M,'p':p,'chi':chi,'variance_absolute_error':abs(v-vv),'third_cumulant_absolute_error':abs(t-tt)})
 cases=[];G=np.array([[16.,9.],[25.,12.]]);J0=np.array([[1.,-6.],[2.,-10.]]);probs=J0**2/(J0**2+G);weights=J0/probs;f=np.array([.8,.2]);m=.3
 for N in [1000,2000,4000]:
  for chi in [0.,.5,1.,2.]:
   v=np.zeros(2);t=np.zeros(2)
   for a in range(2):
    for b in range(2):
     vv,tt=cumulants(round(N*f[b]),probs[a,b],chi)
     v[a]+=(weights[a,b]*m)**2*vv/N;t[a]+=(weights[a,b]*m)**3*tt/N**1.5
   cases.append({'N':N,'chi':chi,'mean_rate_degree_component_variance':v.tolist(),'mean_rate_degree_component_third_cumulant':t.tolist(),'standardized_skew':np.divide(t,v**1.5,out=np.zeros_like(t),where=v>0).tolist()})
 out={'scope':'Exact cumulants of implemented degree-count generator, and of incoming constant-mean-rate contribution. Not full conditional input skew, response correction, or PR repair.','checks':checks,'cases':cases}
 (HERE/'degree_count_cumulant_checks.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'checks':len(checks),'max_variance_error':max(c['variance_absolute_error'] for c in checks),'max_third_cumulant_error':max(c['third_cumulant_absolute_error'] for c in checks)},indent=2))
