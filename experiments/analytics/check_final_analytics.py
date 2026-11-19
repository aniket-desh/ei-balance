"""Independent finite-dimensional, integral and lag constraints for final EI algebra."""
import os
for _name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[_name]='1'
import json,sys
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.linalg import solve_discrete_lyapunov
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'recovered/bundle/ei_full/ei_balance_handoff/src'))
from eicavity.single_site import centered_moment
from eicavity.pair import even_spectrum,block_covariance_energy
from eicavity.lag_measure import validate_fft_convention


def integral_checks():
    nodes,weights=np.polynomial.hermite.hermgauss(80)
    nodes=nodes*np.sqrt(2);weights=weights/np.sqrt(np.pi)
    errors=[];mean_errors=[];gain_errors=[]
    for mu,q in [(-2.,.5),(.3,1.2),(1.8,3.)]:
        vals=2*ndtr(mu+np.sqrt(q)*nodes)
        mean_errors.append(abs(weights@vals-2*ndtr(mu/np.sqrt(1+q))))
        gain=np.sqrt(2/np.pi)*np.exp(-(mu+np.sqrt(q)*nodes)**2/2)
        gain_errors.append(abs(weights@gain-np.sqrt(2/np.pi/(1+q))*np.exp(-mu*mu/(2*(1+q)))))
        for rho in [.1,.6,.95]:
            x=mu+np.sqrt(q)*nodes[:,None]
            y=mu+np.sqrt(q)*(rho*nodes[:,None]+np.sqrt(1-rho*rho)*nodes[None,:])
            direct=np.sum(weights[:,None]*weights[None,:]*(2*ndtr(x))*(2*ndtr(y)))-(weights@vals)**2
            price=float(centered_moment(mu,q,rho*q))
            errors.append(abs(direct-price))
    t=np.arange(1601)*.05;q=1.3;mu=-.7;k=q*np.exp(-t)
    covariance=centered_moment(mu,q,k)
    alpha=np.sqrt(2/np.pi/(1+q))*np.exp(-mu*mu/(2*(1+q)))
    omega,residual=even_spectrum(covariance-alpha*alpha*k,.05)
    assert max(errors)<5e-10
    assert max(mean_errors+gain_errors)<2e-12
    assert residual.min()>-1e-13
    return {'Price_vs_independent_Gauss_Hermite_max_abs':max(errors),
            'mean_formula_max_abs':max(mean_errors),'gain_formula_max_abs':max(gain_errors),
            'minimum_Hermite_residual_spectral_mass':float(residual.min()),
            'spectral_mass_sum_error':float(abs(residual.sum()-(covariance[0]-alpha*alpha*q)))}


def closure_checks():
    rng=np.random.default_rng(872939)
    residuals=[];lyap=[];hermitian=[]
    for p in [1,2,3,5]:
        f=rng.uniform(.3,1,p);f/=f.sum();F=np.diag(f)
        G=rng.uniform(.1,2,(p,p));u=rng.uniform(.1,.5,p)*(1+.7j)
        A=np.diag(u)@G@F
        G*=.6/max(abs(np.linalg.eigvals(A)))
        A=np.diag(u)@G@F
        Q=rng.uniform(.1,2,p);Qd=np.diag(Q);D=np.diag(Q/f)
        R=np.linalg.inv(np.eye(p)-A);X=R@D@R.conj().T;Psi=X-D
        GF=np.linalg.solve(np.eye(p)-G@F@np.diag(u.conj()),G)
        gf_rhs=G+G@F@np.diag(u.conj())@G+G@F@np.diag(u.conj())@GF@np.diag(u.conj())@F@G
        Geta=G@F@Qd@G.T+G@F@Psi@F@G.T
        psirhs=np.diag(u)@Geta@np.diag(u.conj())+np.diag(u)@GF.conj()@Qd+Qd@GF.T@np.diag(u.conj())
        residuals.extend([np.max(abs(gf_rhs-GF)),np.max(abs(psirhs-Psi))])
        rhs=R@D+D@R.conj().T-D
        Xindependent=solve_discrete_lyapunov(A,rhs)
        lyap.append(np.max(abs(Xindependent-X)))
        hermitian.append(np.max(abs(X-X.conj().T)))
        assert np.linalg.eigvalsh(X).min()>0
    assert max(residuals+lyap+hermitian)<1e-11
    return {'full_cavity_equation_max_abs':float(max(residuals)),
            'independent_SciPy_Lyapunov_max_abs':float(max(lyap)),
            'Hermitian_max_abs':float(max(hermitian))}


def path_series_check():
    # Integrable, nonidentical stationary spectra; fixed masses on a symmetric grid.
    omega=np.linspace(-16,16,401);dw=omega[1]-omega[0]
    H=1/(1+1j*omega)
    tau=np.array([[.7,1.3,2.],[.6,1.7,3.]])
    var=np.array([[.2,.5,.9],[.15,.3,1.1]])
    mass=2*var[...,None]*tau[...,None]/(1+(tau[...,None]*omega)**2)*dw/(2*np.pi)
    wt=np.array([.2,.5,.3]);f=np.array([.7,.3]);A=np.array([.1,.12])
    G=np.array([[3.,2.],[1.,2.5]]);B=A[:,None]*G*f[None,:]
    Q=np.einsum('czw,z,czv->cwv',mass,wt,mass)
    t=H[:,None]*H[None,:]
    R=np.linalg.inv(np.eye(2)-t[...,None,None]*B)
    direct=np.einsum('ijac,cij,ijbc,c->ab',R,Q,R.conj(),1/f,optimize=True).real
    power=np.stack([np.linalg.matrix_power(B,n) for n in range(36)])
    hn=np.stack([H**n for n in range(36)])
    moments=np.einsum('czw,nw,mw->cznm',mass,hn,hn.conj(),optimize=True)
    imag=float(abs(moments.imag).max());moments=moments.real
    second=np.einsum('z,cznm,cznm->cnm',wt,moments,moments)
    series=np.einsum('nac,mbc,cnm,c->ab',power,power,second,1/f,optimize=True)
    relative=float(abs(series-direct).max()/abs(direct).max())
    assert relative<1e-10 and imag<1e-14
    eig=float(np.linalg.eigvalsh(direct).min())
    assert eig>0 and np.min(direct)>=0
    return {'relative_path_series_vs_resolvent':relative,
            'real_moment_imaginary_error':imag,
            'minimum_integrated_X_eigenvalue':eig}


def limits():
    u=.13-.08j;q=.7;g2=2.;errs=[]
    for fe in [.1,.5,.8,.95]:
        f=np.array([fe,1-fe]);full,_=block_covariance_energy(np.full((2,2),g2),f,[u,u],[q,q])
        errs.append(abs(f@full@f-q/abs(1-g2*u)**2))
    f=np.array([.8,.2]);h=1/(1+1j*np.array([.3,.8]));u=np.array([np.prod(h),.2*np.prod(h)])
    G=np.array([[0,.6],[0,0.]])
    _,psi=block_covariance_energy(G,f,u,[.4,.7]);directed=abs(psi[0,1]-.6*u[0]*.7)
    zero,_=block_covariance_energy(np.zeros((2,2)),f,u,[.4,.7])
    err0=abs(f@zero@f-(f@np.array([.4,.7])))
    # Cumulative block energies obey a finite covariance PSD bound independently.
    rng=np.random.default_rng(123);X=rng.normal(size=(11,7));C=X@X.T;split=[slice(0,8),slice(8,None)]
    E=np.array([[np.sum(C[a,b]**2) for b in split] for a in split])/11
    cross=float(E[0,1]/np.sqrt(E[0,0]*E[1,1]));share=float(2*E[0,1]/E.sum())
    assert max(errs+[directed,err0])<1e-12 and cross<=1 and share<=.5
    return {'uniform_fraction_null_max_abs':float(max(errs)),
            'isolated_directed_edge_phase_error':float(directed),
            'uncoupled_diagonal_energy_error':float(err0),
            'FFT_lag_orientation_error':validate_fft_convention(),
            'finite_covariance_cross_overlap':cross,'finite_covariance_cross_share':share}

if __name__=='__main__':
    out={'Gaussian_integrals':integral_checks(),'population_closure':closure_checks(),
         'nonnegative_population_path_expansion':path_series_check(),'limits':limits(),
         'status':'All algebra and independent numerical constraints pass; no nonlinear cavity theorem is claimed.'}
    path=Path(__file__).with_name('analytics_checks.json');path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
