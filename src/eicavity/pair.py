"""Conditional two-site covariance energy and participation ratio.

The scalar implementation integrates the continuous-time cavity denominator.
The multi-population matrix identity is kept independently testable.
"""
import numpy as np


def even_spectrum(covariance, dt):
    """Two-sided spectral masses, whose sum equals zero-lag covariance.

    A finite even lag sequence has period 2*T. A returned mass is
    C(omega_k) d_omega/(2*pi), not the spectral density C(omega_k).
    """
    c=np.asarray(covariance)
    sequence=np.concatenate([c,c[..., -2:0:-1]],axis=-1)
    masses=np.fft.fft(sequence,axis=-1).real/sequence.shape[-1]
    omega=2*np.pi*np.fft.fftfreq(sequence.shape[-1],dt)
    return omega,masses


def conditional_dimension(dmft,dt=.05,max_time=None,n_hermite=96,max_frequency=16):
    t,k=dmft.kernel(dt,max_time)
    one=dmft.conditional(k,n_hermite)
    omega,p=even_spectrum(one['covariance'],dt)
    keep=np.abs(omega)<=max_frequency
    discarded=float(np.sum(np.abs(p[...,~keep]))/max(np.sum(np.abs(p)),1e-30))
    omega,p=omega[keep],p[...,keep]
    weights=dmft.fractions[:,None]*one['weights'][None,:]
    alpha2=float(np.sum(weights*one['alpha']**2))
    alpha_mean=float(np.sum(weights*one['alpha']))
    radius=dmft.g**2*alpha2
    h=1/(1+1j*omega)
    factor=1/np.abs(1-radius*h[:,None]*h[None,:])**2
    ps=p.reshape(-1,len(omega))
    wt=weights.reshape(-1)
    energy_local=np.einsum('zi,ij,zj->z',ps,factor,ps,optimize=True)
    energy=float(wt@energy_local)
    variances=one['covariance'][...,0]
    mean_variance=float(np.sum(weights*variances))
    variance_second=float(np.sum(weights*variances**2))
    avg_spectrum=np.sum(weights[...,None]*p,axis=(0,1))
    homogenized_energy=float(avg_spectrum@factor@avg_spectrum)
    mean_gain_factor=1/np.abs(1-dmft.g**2*alpha_mean**2*h[:,None]*h[None,:])**2
    mean_gain_energy=float(np.einsum('z,zi,ij,zj->',wt,ps,mean_gain_factor,ps,optimize=True))
    # A readout y_i=w_i*r_i changes endpoint weights, not recurrent gains.
    # Z-scoring therefore does not reduce to replacing every c_i by c_i/v_i
    # in the recurrent formula. The three terms below keep that distinction.
    vf=variances.reshape(-1)
    af=one['alpha'].reshape(-1)
    valid=(wt>1e-15)&(vf>1e-200)
    pn=ps[valid]/vf[valid,None]
    qnorm=(pn.T*wt[valid])@pn
    qmixed=(ps[valid].T*wt[valid])@pn
    qraw=(ps.T*wt)@ps
    response_norm=np.sum(wt[valid]*af[valid]**2/vf[valid])
    aw=dmft.g**2*response_norm*h[:,None]*h[None,:]/(1-radius*h[:,None]*h[None,:])
    corr_energy=float(np.sum(qnorm+2*aw.real*qmixed+np.abs(aw)**2*qraw))
    # For centered preactivations: c_x=K, c_xr=alpha*K, s_x=H.
    ox,kp=even_spectrum(k,dt)
    kp=kp[keep]
    ax=dmft.g**2*h[:,None]*h[None,:]/(1-radius*h[:,None]*h[None,:])
    input_energy=float(np.sum((1+2*alpha2*ax.real)*kp[:,None]*kp[None,:]+np.abs(ax)**2*qraw))
    return {
        'time':t,'kernel':k,'omega':omega,'spectral_masses':p,**one,
        'mean_variance':mean_variance,'variance_second':variance_second,
        'alpha_mean':alpha_mean,'alpha_second':alpha2,'response_radius':radius,
        'covariance_energy':energy,'offdiagonal_energy':energy-variance_second,
        'dimension_fraction':mean_variance**2/energy,
        'diagonal_dimension_fraction':mean_variance**2/variance_second,
        'homogenized_dimension_fraction':mean_variance**2/homogenized_energy,
        'mean_gain_dimension_fraction':mean_variance**2/mean_gain_energy,
        'correlation_dimension_fraction':1/corr_energy,
        'correlation_covariance_energy':corr_energy,
        'input_dimension_fraction':dmft.dynamic_variance**2/input_energy,
        'input_covariance_energy':input_energy,
        'correlation_quadrature_mass_retained':float(wt[valid].sum()),
        'min_spectral_mass':float(p.min()),'discarded_spectral_mass_fraction':discarded,
        'kernel_tail_fraction':float(k[-1]/k[0]),
    }


def block_covariance_energy(variance_profile,fractions,u,q):
    """R D R^dagger and its off-diagonal-pair contribution.

    q[a]=E[c_a(z,w1)c_a(z,w2)], u[a]=E[s_a(z,w1)s_a(z,w2)].
    This function is an algebraic identity conditional on the cavity closure.
    """
    g,f,u,q=map(np.asarray,(variance_profile,fractions,u,q))
    # C_ij(tau)=<r_i(t+tau)r_j(t)>: a directed j->i edge carries u_i,
    # not its conjugate. Conjugating R reverses both lag arguments.
    a=np.diag(u)@g@np.diag(f)
    r=np.linalg.inv(np.eye(len(f))-a)
    d=np.diag(q/f)
    full=r@d@r.conj().T
    return full,full-d
