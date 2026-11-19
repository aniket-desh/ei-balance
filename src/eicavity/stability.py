"""Largest Lyapunov exponent from the one-site two-replica operator."""
import numpy as np
from scipy.linalg import eigh_tridiagonal
from .single_site import raw_derivative


def maximum_lyapunov(dmft,du=.025,scaled_horizon=25.):
    """Continuous-time exponent; Dirichlet boundaries on a distant full line.

    The two-replica Schrödinger operator is
    -d^2/dt^2 + 1 - g^2 E[phi'(x(t)) phi'(x(0))].
    Its ground-state energy E0 gives lambda=-1+sqrt(1-E0).
    """
    gap=1-dmft.response_radius
    scale=np.sqrt(gap)
    t,k=dmft.kernel(dt=du/scale,max_time=scaled_horizon/scale)
    d,q=dmft.static_variance,dmft.dynamic_variance
    gain=sum(f*raw_derivative(h,d+q,d+k) for f,h in zip(dmft.fractions,dmft.mean_inputs))
    potential=1-dmft.g**2*gain
    full=np.r_[potential[:0:-1],potential]
    step=t[1]-t[0]
    diag=2/step**2+full[1:-1]
    off=np.full(len(diag)-1,-1/step**2)
    energies=eigh_tridiagonal(diag,off,select='i',select_range=(0,1),eigvals_only=True)
    return {'maximum_lyapunov':float(np.sqrt(1-energies[0])-1),
            'ground_energy':float(energies[0]),'translation_mode_energy':float(energies[1]),
            'step':step,'horizon':float(t[-1])}
