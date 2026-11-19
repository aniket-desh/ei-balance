"""Exact row-covariance algebra and small independent-row Monte Carlo check."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys,json,time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'recovered/bundle/ei_full/ei_balance_handoff/src'))
from eicavity.degree import degree_counts

def covariance(G,chi,sizes):
    N=sum(sizes);S=np.zeros((N,N));offset=0
    for g,x,m in zip(G,chi,sizes):
        S[offset:offset+m,offset:offset+m]=g/N*((m-x)/(m-1)*np.eye(m)+(x-1)/(m-1)*np.ones((m,m)))
        offset+=m
    return S

def expansion(C,G_a,G_c,chi_a,chi_c,sizes):
    N=sum(sizes);bounds=np.r_[0,np.cumsum(sizes)]
    total=0.;pieces=np.zeros(4)
    for b,m in enumerate(sizes):
        ub=np.zeros(N);ub[bounds[b]:bounds[b+1]]=1/np.sqrt(m)
        s=G_a[b]/N*(m-chi_a[b])/(m-1)
        t=G_a[b]/N*m*(chi_a[b]-1)/(m-1)
        for d,l in enumerate(sizes):
            ud=np.zeros(N);ud[bounds[d]:bounds[d+1]]=1/np.sqrt(l)
            v=G_c[d]/N*(l-chi_c[d])/(l-1)
            w=G_c[d]/N*l*(chi_c[d]-1)/(l-1)
            sl_b=slice(bounds[b],bounds[b+1]);sl_d=slice(bounds[d],bounds[d+1])
            pieces += [s*v*np.sum(C[sl_b,sl_d]**2),
                       t*v*np.sum((C.T@ub)[sl_d]**2),
                       s*w*np.sum((C@ud)[sl_b]**2),
                       t*w*(ub@C@ud)**2]
    return float(pieces.sum()),pieces.tolist()

def sample_rows(rng,ns,sizes,ps,G,chi):
    rows=[];effective=[];N=sum(sizes)
    for m,p,g,x in zip(sizes,ps,G,chi):
        counts,eff=degree_counts(rng,ns,m,p,x)
        # Sorting iid continuous keys generates a uniformly random subset.
        rank=np.argsort(np.argsort(rng.random((ns,m)),axis=1),axis=1)
        a=rank<counts[:,None]
        rows.append(np.sqrt(g/(N*p*(1-p)))*(a-p));effective.append(eff)
    return np.concatenate(rows,axis=1),effective

def main():
    tic=time.monotonic();rng=np.random.default_rng(734899)
    sizes=[20,20];N=sum(sizes);G_a=[3.,5.];G_c=[4.,2.]
    A=rng.normal(size=(N,N))/np.sqrt(N)
    C=A@A.T+.5*np.eye(N)
    monte=[]
    for chi in [0.,.5,1.,2.]:
        x,ca=sample_rows(rng,50000,sizes,[.25,.4],G_a,[chi,chi])
        y,cc=sample_rows(rng,50000,sizes,[.25,.4],G_c,[chi,chi])
        Sa=covariance(G_a,ca,sizes);Sc=covariance(G_c,cc,sizes)
        exact=float(np.trace(Sa@C@Sc@C.T))
        expanded,pieces=expansion(C,G_a,G_c,ca,cc,sizes)
        np.testing.assert_allclose(expanded,exact,rtol=1e-13,atol=1e-13)
        Y=np.einsum('ij,ij->i',x@C,y);sq=Y*Y
        obs=float(sq.mean());se=float(sq.std(ddof=1)/np.sqrt(len(sq)))
        monte.append({'chi_requested':chi,'chi_effective':ca,'exact_source':exact,
                      'expanded_source':expanded,'source_pieces':pieces,
                      'monte_carlo_source':obs,'MC_SE':se,'z_error':(obs-exact)/se})
    scaling=[]
    for n in [40,80,160,320,640]:
        u=np.ones(n)/np.sqrt(n)
        for label,amp in [('bounded_coherent',1.),('sqrtN_coherent',np.sqrt(n))]:
            C=np.eye(n)+amp*np.outer(u,u)
            Cu=C@u;energy=np.sum(C*C)
            ratio=(n/(n-1))**2*(1-(2*np.sum(Cu*Cu)-(u@C@u)**2)/energy)
            S=(np.eye(n)-np.outer(u,u))*n/(n-1)/n
            direct=np.trace(S@C@S@C.T)/(energy/n**2)
            np.testing.assert_allclose(ratio,direct,rtol=1e-13,atol=1e-13)
            scaling.append({'N':n,'case':label,'source_ratio_fixed_to_iid':float(ratio),
                            'coherent_projection_energy_fraction':float(np.sum(Cu*Cu)/energy)})
    out={'scope':'Probe rows independent of specified C; exact finite row-covariance identity, not recurrent closure theorem.',
         'monte_carlo_checks':monte,'coherent_scaling_counterexample':scaling,
         'all_algebra_checks_passed':True,'seconds':time.monotonic()-tic}
    dest=Path(__file__).with_suffix('.json');dest.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
