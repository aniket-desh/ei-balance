"""A testable heterogeneous extension of the fixed-connectivity covariance.

The matrix identity is exact for a specified linear residual model. Its use
for nonlinear network covariances is an asymptotic cavity approximation.
"""
import numpy as np
from scipy.linalg import solve,eigvals
from scipy.special import roots_legendre
from scipy.interpolate import CubicSpline
from .single_site import rate,centered_moment
from .pair import even_spectrum


def solve_mean_inputs(j,drive,q,initial,max_iter=60,tol=1e-9):
    """Newton solve of mu=drive+J E[phi(mu+sqrt(q) Z)]."""
    mu=np.asarray(initial,dtype=float).copy()
    n=len(mu)
    def residual(x):
        return x-drive-j@rate(x/np.sqrt(1+q))
    errors=[]
    for iteration in range(max_iter):
        error=residual(mu)
        norm=np.linalg.norm(error)/np.sqrt(n)
        errors.append(float(norm))
        if np.max(np.abs(error))<tol:
            return mu,errors
        alpha=np.sqrt(2/np.pi/(1+q))*np.exp(-mu*mu/(2*(1+q)))
        direction=solve(np.eye(n)-j*alpha[None,:],error,check_finite=False)
        damping=1.
        while damping>2**-16:
            proposal=mu-damping*direction
            if np.linalg.norm(residual(proposal))/np.sqrt(n)<norm:
                mu=proposal
                break
            damping*=.5
        else:
            raise RuntimeError('Mean-field Newton line search failed')
    raise RuntimeError('Mean-field Newton solver did not converge')


def population_basis(labels):
    n=len(labels);p=int(np.max(labels))+1
    basis=np.zeros((n,p))
    f=np.bincount(labels,minlength=p)/n
    for a in range(p):
        basis[labels==a,a]=1/np.sqrt(n*f[a])
    return basis,f


def predict_covariance(j,labels,j0,dmft,mu,n_frequency=120,cutoff=6.,dt=.06,horizon=120.):
    n=len(mu)
    t,k=dmft.kernel(dt,horizon)
    q=dmft.dynamic_variance
    alpha=np.sqrt(2/np.pi/(1+q))*np.exp(-mu*mu/(2*(1+q)))
    cov=np.array([centered_moment(m,q,k) for m in mu])
    residual=cov-alpha[:,None]**2*k
    omega,mass=even_spectrum(residual,dt)
    step=omega[1]-omega[0]
    order=np.argsort(omega)
    spectrum=CubicSpline(omega[order],mass[:,order]*2*np.pi/step,axis=-1)
    nodes,weights=roots_legendre(n_frequency)
    frequencies=(nodes+1)*cutoff/2
    weights=weights*cutoff/(2*np.pi) # includes positive/negative frequencies
    powers=spectrum(frequencies)
    min_power=float(powers.min())
    if min_power< -1e-7:
        raise ValueError(f'Residual spectrum is not positive: {min_power}')
    # Roundoff and interpolation errors in negligible high-frequency tails.
    negative_power=float(-np.sum(np.minimum(powers,0)*weights[None,:]))
    powers=np.maximum(powers,0)
    basis,f=population_basis(labels)
    b=np.sqrt(n)*np.diag(np.sqrt(f))@j0@np.diag(np.sqrt(f))
    mean=basis@b@basis.T
    bulk=j-mean
    full_abscissa=float(np.max(eigvals(alpha[:,None]*j,check_finite=False).real))
    bulk_abscissa=float(np.max(eigvals(alpha[:,None]*bulk,check_finite=False).real))
    if full_abscissa>=1:
        raise ValueError(f'Full linear equivalent is unstable: full={full_abscissa}, bulk={bulk_abscissa}')
    identity=np.eye(n)
    cfull=np.zeros((n,n))
    cbulk=np.zeros_like(cfull)
    max_small_inverse=0.
    for ix,(w,weight) in enumerate(zip(frequencies,weights)):
        s=alpha/(1+1j*w)
        l0=solve(identity-s[:,None]*bulk,identity,check_finite=False)
        v=l0@(s[:,None]*basis@b)
        small=np.eye(len(f))-basis.T@v
        gain=solve(small,basis.T@l0,check_finite=False)
        lf=l0+v@gain
        noise=powers[:,ix]
        cfull+=weight*((lf*noise[None,:])@lf.conj().T).real
        if bulk_abscissa<1:
            cbulk+=weight*((l0*noise[None,:])@l0.conj().T).real
        max_small_inverse=max(max_small_inverse,float(np.linalg.norm(np.linalg.inv(small),2)))
    if bulk_abscissa>=1:
        cbulk[:]=np.nan # no stationary covariance for this control
    return {'covariance':cfull,'bulk_covariance':cbulk,'mu':mu,'alpha':alpha,
            'full_response_abscissa':full_abscissa,'bulk_response_abscissa':bulk_abscissa,
            'bulk_is_stable':bool(bulk_abscissa<1),
            'minimum_residual_spectrum':min_power,'negative_power_clipped':negative_power,
            'max_population_inverse_norm':max_small_inverse,
            'frequency_nodes':frequencies,'frequency_weights':weights}


def covariance_summary(c):
    n=len(c);v=np.trace(c)/n;energy=np.sum(c*c)/n
    return {'mean_variance':float(v),'covariance_energy':float(energy),
            'dimension_fraction':float(v*v/energy),
            'third_moment':float(np.trace(c@c@c)/n)}
