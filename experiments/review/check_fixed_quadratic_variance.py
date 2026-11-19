"""Exhaustive fixed-subset checks of quadratic and coherent-row variance."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import itertools,json,math
import numpy as np

def inclusion(m,k,r):
    if r>k:return 0.
    return math.prod(range(k-r+1,k+1))/math.prod(range(m-r+1,m+1))

def form_variance(S,m,k):
    p=k/m;P=np.eye(m)-np.ones((m,m))/m;B=P@S@P
    ps=[None]+[inclusion(m,k,r) for r in range(1,5)]
    c=ps[2]-2*ps[3]+ps[4];kap=p-7*ps[2]+12*ps[3]-6*ps[4]
    D=np.trace(B);T=np.sum(B*B);L=np.sum(np.diag(B)**2)
    variance=2*c*T+kap*L+(c-(p-ps[2])**2)*D*D
    V=p*(1-p)*m/(m-1)*P
    return variance,(p-ps[2])*D,B,V,p-3*ps[2]+2*ps[3]

def all_rows(m,k):
    rows=[]
    for subset in itertools.combinations(range(m),k):
        a=np.zeros(m);a[list(subset)]=1;rows.append(a)
    return np.array(rows)

def main():
    rng=np.random.default_rng(990178);checks=[]
    for m,k in [(5,2),(7,3),(8,2),(8,5)]:
        A=rng.normal(size=(m,m));S=A@A.T/m
        v,mean,B,V,c3=form_variance(S,m,k)
        aa=all_rows(m,k);z=aa-k/m
        q=np.einsum('ni,ij,nj->n',z,S,z)
        np.testing.assert_allclose([q.mean(),q.var()],[mean,v],rtol=1e-12,atol=1e-12)
        lam=rng.normal(size=m);lam-=lam.mean()
        linear=z@lam;cov=np.mean((q-q.mean())*linear)
        exactcov=c3*np.sum(lam*np.diag(B))
        np.testing.assert_allclose(cov,exactcov,rtol=1e-12,atol=1e-12)
        constv=form_variance(np.eye(m),m,k)[0]
        np.testing.assert_allclose(constv,0,rtol=0,atol=1e-12)
        checks.append({'M':m,'K':k,'subsets':len(aa),'variance_error':abs(q.var()-v),
                       'linear_quadratic_covariance_error':abs(cov-exactcov),'constant_diagonal_variance':constv})
    sizes=[5,4];counts=[2,2];t=np.array([2.,-1.3]);N=sum(sizes)
    p=np.array(counts)/np.array(sizes);meanrow=np.repeat(t*p/np.sqrt(N),sizes)
    A=rng.normal(size=(N,N));S=A@A.T/N;ell=S@meanrow
    bounds=np.r_[0,np.cumsum(sizes)];Vs=[];Bs=[];vdiag=0.;vlinear=0.;vcubic=0.;meaneq=meanrow@S@meanrow
    for b,(m,k) in enumerate(zip(sizes,counts)):
        sl=slice(bounds[b],bounds[b+1]);v,mean,B,V,c3=form_variance(S[sl,sl],m,k)
        Vs.append(V);Bs.append(B);meaneq+=t[b]**2/N*mean
        vdiag+=t[b]**4/N**2*v
        vlinear+=4*t[b]**2/N*(ell[sl]@V@ell[sl])
        lc=ell[sl]-ell[sl].mean()
        vcubic+=4*t[b]**3/N**1.5*c3*np.sum(lc*np.diag(B))
    S01=S[:sizes[0],sizes[0]:]
    vcross=4*t[0]**2*t[1]**2/N**2*np.trace(Vs[0]@S01@Vs[1]@S01.T)
    expectedvar=vdiag+vlinear+vcubic+vcross
    rows0=all_rows(sizes[0],counts[0]);rows1=all_rows(sizes[1],counts[1])
    jj=np.array([np.r_[t[0]*a,t[1]*b]/np.sqrt(N) for a in rows0 for b in rows1])
    qq=np.einsum('ni,ij,nj->n',jj,S,jj)
    np.testing.assert_allclose([qq.mean(),qq.var()],[meaneq,expectedvar],rtol=1e-12,atol=1e-12)
    out={'scope':'Exact exhaustive finite-subset identities with fixed S, independent pools; no stationary recurrent expansion asserted.',
         'single_pool':checks,'two_pool':{'networks_enumerated':len(jj),'mean_error':abs(qq.mean()-meaneq),
             'variance_error':abs(qq.var()-expectedvar),'variance':float(expectedvar),
             'sectors':{'centered_quadratic_diag':float(vdiag),'centered_quadratic_cross':float(vcross),
                       'coherent_linear':float(vlinear),'linear_quadratic_cubic_covariance':float(vcubic)}},
         'all_checks_passed':True}
    Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
