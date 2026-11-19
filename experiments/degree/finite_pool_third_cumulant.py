"""Exact exchangeable third-cumulant audit for degree-controlled rows.

This is a finite-pool identity, not a closure of the finite-network PR theory.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
from pathlib import Path
from itertools import product
import math,json
import numpy as np
from check_degree_theory import count_pmf

HERE=Path(__file__).resolve().parent

def audit(m,p,chi):
    pmf=count_pmf(m,p,chi);k=np.arange(m+1)
    p2=float((k*(k-1))@pmf/(m*(m-1)))
    p3=float((k*(k-1)*(k-2))@pmf/(m*(m-1)*(m-2)))
    c111=p*(1-p)*(1-2*p)
    c112=(1-2*p)*(p2-p*p)
    c123=p3-3*p*p2+2*p**3
    masks=np.array(list(product([0.,1.],repeat=m)))
    probs=np.array([pmf[int(a.sum())]/math.comb(m,int(a.sum())) for a in masks])
    rows=[]
    centered=np.linspace(.1,1.2,m)**2
    centered-=centered.mean()
    for label,values in [('constant',np.full(m,.3)),('noncentered',np.linspace(.1,1.2,m)),('centered',centered)]:
        s1=values.sum();s2=np.square(values).sum();s3=(values**3).sum()
        exact=float(((masks-p)@values)**3@probs)
        predicted=(c111-3*c112+2*c123)*s3+3*(c112-c123)*s1*s2+c123*s1**3
        rows.append({'source':label,'third_moment_enumerated':exact,'third_moment_formula':float(predicted),'absolute_error':float(exact-predicted)})
    return {'M':m,'p':p,'chi':chi,'kappa111':c111,'kappa112':c112,'kappa123':c123,
            'centered_source_cubic_coefficient':c111-3*c112+2*c123,'rows':rows}

def main():
    cases=[audit(8,.3,c) for c in [0.,.15,.5,1.,2.,5.]]
    cases.append(audit(8,.375,0.)) # integerfixedcount3
    assert max(abs(r['absolute_error']) for c in cases for r in c['rows'])<1e-12
    fixed=cases[-1];factor=8**2/((8-1)*(8-2))
    assert abs(fixed['centered_source_cubic_coefficient']/fixed['kappa111']-factor)<1e-12
    result={'scope':'Exact finite-pool moment identities only. Leading one-site degree closure does not determine the finite-N skewness corrections; replacing a Bernoulli static variance by chi is insufficient for a finite-size correction.',
            'cases':cases,'fixed_integer_count_centered_source_factor':factor,
            'status':'Exact enumeration and fixed-count cancellation pass.'}
    (HERE/'finite_pool_third_cumulant.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
