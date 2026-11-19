"""Population-composition dependence at fixed aggregate input laws."""
import numpy as np
from .pair import even_spectrum


def energy_tensor(aggregate_variance, conditional, dt, max_frequency=12.):
    """T[a,b,c] integrates R[a,c] R[b,c]* Q[c], before division by f[c].

    The conditional spectra are spectral masses, including both integration
    measures. Keeping W=G F fixed makes this tensor independent of fractions.
    """
    w=np.asarray(aggregate_variance)
    weights=conditional['weights']
    omega,spectrum=even_spectrum(conditional['covariance'],dt)
    keep=np.abs(omega)<=max_frequency
    omega,spectrum=omega[keep],spectrum[...,keep]
    a2=conditional['alpha']**2@weights
    q=np.array([(c.T*weights)@c for c in spectrum])
    h=1/(1+1j*omega)
    p=len(a2);tensor=np.zeros((p,p,p),dtype=complex)
    for lo in range(0,len(omega),32):
        hi=min(lo+32,len(omega))
        u=(h[lo:hi,None]*h[None,:])[...,None]*a2
        r=np.linalg.inv(np.eye(p)-u[..., :,None]*w)
        source=np.moveaxis(q[:,lo:hi],0,-1)
        tensor+=np.einsum('ijac,ijbc,ijc->abc',r,r.conj(),source,optimize=True)
    if np.max(np.abs(tensor.imag))>1e-8:
        raise RuntimeError('Integrated covariance energy is not real')
    return tensor.real


def dimension_at_fraction(tensor,mean_variances,variance_seconds,fractions):
    f=np.asarray(fractions,dtype=float)
    if np.any(f<=0) or not np.isclose(f.sum(),1):
        raise ValueError('Positive normalized population fractions required')
    v=np.asarray(mean_variances)
    full=np.einsum('abc,c->ab',tensor,1/f)
    energy=f@full@f
    pair=full-np.diag(np.asarray(variance_seconds)/f)
    return {'dimension_fraction':float((f@v)**2/energy),
            'covariance_energy':float(energy),'pair_covariance_energy':pair.tolist(),
            'within_population_dimension_fractions':(v*v/(f*np.diag(full))).tolist(),
            'cross_population_energy_share':float((energy-np.sum(f*f*np.diag(full)))/energy)}


def critical_composition(left,right,slope_seconds,slope_fourths,fractions):
    """Critical geometry and the best fraction on a fixed-W control path."""
    l,r,a2,a4,f=map(np.asarray,(left,right,slope_seconds,slope_fourths,fractions))
    if not np.isclose(l@r,1):raise ValueError('Perron modes must obey l.T r = 1')
    c=(l*r)**2*a4/a2**2
    root=np.sqrt(c)
    s=np.sum(c/f)
    constant=4*np.pi**2/3-12
    return {'heterogeneity_factor':float(s),'dimension_prefactor_in_gap':float(1/(constant*s)),
            'optimal_fractions':(root/root.sum()).tolist(),
            'minimum_heterogeneity_factor':float(root.sum()**2),'coefficients':c.tolist()}
