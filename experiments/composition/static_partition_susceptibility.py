"""Conditional fixed-total-variance partition susceptibility and degree tangent.

The frozen-total-variance Jacobian is an exact derivative of the Gaussian
static functional, not the complete finite-network quenched susceptibility.
The chi tangent solves the full population boundary-value problem and keeps
its static/dynamic coupling. No network datum is fitted.
"""
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'recovered/bundle/ei_full/ei_balance_handoff/src'))
from eicavity.multitype import solve_populations
from eicavity.single_site import raw_moment


def main():
    ref=json.loads((ROOT/'recovered/bundle/diagnosis/theory/summary.json').read_text())
    D,q,h,A=map(lambda k:np.array(ref[k]),['static_variances','dynamic_variances','mean_inputs','alpha_second'])
    V=D+q;W=np.array([[12.8,1.8],[20.,2.4]]);G=np.array([[16.,9.],[25.,12.]])
    J=W@np.diag(A);eig,right=np.linalg.eig(J);ii=np.argmax(eig.real);r=right[:,ii].real;r/=np.linalg.norm(r);r*=np.sign(r[0])
    leftval,left=np.linalg.eig(J.T);jj=np.argmin(np.abs(leftval-eig[ii]));ell=left[:,jj].real;ell/=ell@r
    inv=np.linalg.inv(np.eye(2)-J)
    # Independent finite differences of the moment functional at fixed h,V.
    steps=[1e-3,5e-4,1e-4];fd=[]
    for eps in steps:
        mplus=np.array([raw_moment(h[a],V[a],D[a]+eps)for a in [0,1]])
        mminus=np.array([raw_moment(h[a],V[a],D[a]-eps)for a in [0,1]])
        deriv=(mplus-mminus)/(2*eps)
        fd.append({'step':eps,'dM2_dDelta_at_fixed_total_variance':deriv.tolist(),'max_relative_error_vs_A':float(np.max(np.abs(deriv/A-1)))})
    tangent=[]
    for eps in [.01,.005]:
        minus=solve_populations(G,degree_dispersion=1-eps);plus=solve_populations(G,degree_dispersion=1+eps)
        dD=(plus.static_variances-minus.static_variances)/(2*eps);dq=(plus.dynamic_variances-minus.dynamic_variances)/(2*eps)
        tangent.append({'step':eps,'dDelta_dchi':dD.tolist(),'dq_dchi':dq.tolist(),'dtotal_dchi':(dD+dq).tolist(),'dq_dDelta_componentwise':(dq/dD).tolist(),'h_plus':plus.mean_inputs.tolist(),'h_minus':minus.mean_inputs.tolist(),'max_bvp_residual':max(plus.bvp_residual,minus.bvp_residual)})
    out={'scope':'Parameter-only conditional Gaussian-functional derivatives; frozenVJacobiannotfullquenchedtheory. FullDMFTdegree-tangentvariesVandhselfconsistently; nofittedchi.','A':A.tolist(),'fixed_V_static_J':J.tolist(),'eigenvalues':eig.real.tolist(),'critical_positive_right_mode':r.tolist(),'critical_left_mode_normalized_l_dot_r_1':ell.tolist(),'response_gap':float(1-eig[ii].real),'modal_amplification':float(1/(1-eig[ii].real)),'inverse_fixedVJacobian':inv.tolist(),'inverse_2norm':float(np.linalg.norm(inv,2)),'Price_derivative_checks':fd,'full_DMFT_chi_tangent':tangent}
    (HERE/'static_partition_susceptibility.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
