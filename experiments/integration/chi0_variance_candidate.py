"""Partial frozen-row q-variance source at nominally fixed degree.

This is an explicitly incomplete diagnostic candidate, not a finite-N repair.
In particular the omitted Cov(mu,q) term is the same order in measured data.
"""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'recovered/bundle/ei_full/ei_balance_handoff/src'))
from eicavity.pair import even_spectrum


def main():
    info=json.loads((ROOT/'next_round/degree/predictions.json').read_text())
    row=next(r for r in info['rows'] if r['chi']==0)
    G=np.array(info['G']);f=np.array(info['fractions']);F=np.diag(f);W=G@F
    one=np.load(ROOT/'next_round/degree/chi0_one_site.npz')
    omega,c=even_spectrum(one['covariance'][:,0],one['time'][1]-one['time'][0])
    keep=np.abs(omega)<=12.;omega,c=omega[keep],c[:,keep]
    H=1/(1+1j*omega);H2=np.abs(H)**2
    alpha=np.array(row['alpha_second'])**.5
    beta=-np.array(row['mean_inputs'])/(1+np.array(row['dynamic_variances']))*alpha
    B=np.diag(alpha**2)@W
    X=np.zeros((2,2),complex)
    for lo in range(0,len(omega),32):
        hi=min(len(omega),lo+32)
        product=H[lo:hi,None]*H[None,:]
        R=np.linalg.inv(np.eye(2)-product[...,None,None]*B)
        Q=np.moveaxis(c[:,lo:hi,None]*c[:,None,:]/f[:,None,None],0,-1)
        full=np.einsum('...ac,...c,...bc->...ab',R,Q,R.conj(),optimize=True)
        X+=np.einsum('x,y,xyab->ab',H2[lo:hi],H2,full,optimize=True)
    Efiltered=np.outer(f,f)*X.real
    filtered_local=c@H2
    Eoff=Efiltered-np.diag(f*filtered_local**2)
    qcoeff=2*np.einsum('ab,bc,ac->a',G,Eoff,G)
    mucoeff=np.linalg.solve(np.eye(2)-W@np.diag(alpha**2),W@(beta**2/4*qcoeff))
    result={'scope':'Partial frozen-row candidate; requires exact subset contraction and omission of other same-order sources to be justified.',
        'chi':0,'filtered_population_energy':Efiltered.tolist(),
        'filtered_offdiagonal_energy':Eoff.tolist(),'filtered_local_variance':filtered_local.tolist(),
        'alpha':alpha.tolist(),'beta':beta.tolist(),
        'N_times_q_variance_candidate':qcoeff.tolist(),
        'N_times_Delta_candidate_without_mixed_source':mucoeff.tolist(),
        'N1000_q_variance_candidate':(qcoeff/1000).tolist(),
        'N1000_Delta_candidate_without_mixed_source':(mucoeff/1000).tolist(),
        'omitted_same_order_term':'alpha*beta*Cov(mu,q), plus exact degree-row and endogenous-feedback contractions; no full correction is claimed.',
        'existing_measured_reference':'Selected degree N1000 chi0 seed4001: Var(q)~[.2076,.5473], Cov(mu,q)~[.01962,.05572], corrected Delta~[.06067,.09650]. These measurements were not fitted.'}
    destination=ROOT/'next_round/integration/chi0_variance_candidate.json'
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
