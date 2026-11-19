"""Continuation from chaos onset and quadrature resolving narrow resonances."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtri,roots_legendre
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from .single_site import ScalarDMFT,raw_moment,raw_derivative,normal_quadrature,rate_prime,_unit_x,_unit_w
from .pair import even_spectrum


def solution_at_variance(q,mean_rates=(.3,),fractions=(1.,)):
    """Parameterize the chaotic branch by q, including q=0 at onset."""
    ms,fs=np.asarray(mean_rates),np.asarray(fractions)
    a=ndtri(ms/2)
    def quantities(d):
        v=d+q;hs=a*np.sqrt(1+v)
        m2=sum(f*raw_moment(h,v,d,m) for f,h,m in zip(fs,hs,ms))
        integral=sum(f*np.sum(_unit_w*(1-_unit_x)*raw_derivative(h,v,d+q*_unit_x))
                     for f,h in zip(fs,hs))
        return hs,m2,integral
    def equation(d):
        h,m2,integ=quantities(d)
        return 2*d*integ-m2
    d=brentq(equation,1e-10,1e6,xtol=5e-14)
    h,m2,integ=quantities(d)
    g=np.sqrt(1/(2*integ))
    radius=g*g*sum(f*raw_derivative(hi,d+q,d) for f,hi in zip(fs,h))
    return ScalarDMFT(g,ms,fs,d,q,h,float(d-g*g*m2),float(1-2*g*g*integ),float(radius),float('nan'))


def critical_prefactor(mean_rates=(.3,),fractions=(1.,)):
    d=solution_at_variance(0.,mean_rates,fractions)
    z,w=normal_quadrature(160)
    mu=d.mean_inputs[:,None]+np.sqrt(d.static_variance)*z
    alpha=rate_prime(mu)
    weights=d.fractions[:,None]*w
    a2=np.sum(weights*alpha**2)
    a4=np.sum(weights*alpha**4)
    b2=np.sum(weights*(mu*alpha)**2) # E[phi''(mu)^2]
    def integrand(x):
        if abs(x)<1e-8:
            return 1/np.pi**2
        return x*x/((1+x*x)*np.sinh(np.pi*x)**2)
    constant=8*np.pi*quad(integrand,0,20,epsabs=1e-12,epsrel=1e-12)[0]
    if not np.isclose(constant,4*np.pi**2/3-12,rtol=1e-12):
        raise RuntimeError('Critical integral disagrees with its closed form')
    return {'critical_g':d.g,'critical_static_variance':d.static_variance,
            'critical_mean_inputs':d.mean_inputs.tolist(),'alpha_second':float(a2),
            'alpha_fourth':float(a4),'curvature_second':float(b2),
            'universal_constant':constant,'dimension_prefactor_in_gap':float(a2*a2/a4/constant),
            'gap_over_q_limit':float(d.g*d.g*b2/3),
            'gap_over_relative_coupling_limit':float(d.g*d.g*b2),
            'dynamic_variance_over_relative_coupling_limit':3.,
            'static_variance_derivative_in_q':float((2*d.static_variance-1)/3),
            'coupling_derivative_in_q':float(d.g/3),
            'dimension_prefactor_in_relative_coupling':float(a2*a2/a4/constant*(d.g*d.g*b2)**1.5)}


def resolved_dimension(dmft,du=.04,scaled_horizon=60.,n_hermite=96,n_outer=180,n_inner=120,scaled_cutoff=20.):
    """Integrate the pair formula without underresolving its narrow ridge.

    For fixed w1, the cavity factor has a Lorentzian in w2 centered at
    -kappa*w1/(1+w1^2), with width 1-kappa/(1+w1^2). A tangent substitution
    removes this ridge exactly, even when its width is much smaller than
    the width of the one-site spectrum.
    """
    gap=1-dmft.response_radius
    if gap<=0:
        raise ValueError('The dynamic solution needs a positive response gap')
    scale=np.sqrt(gap)
    dt=du/scale
    t,k=dmft.kernel(dt=dt,max_time=scaled_horizon/scale)
    one=dmft.conditional(k,n_hermite)
    omega,p=even_spectrum(one['covariance'],dt)
    _,kp=even_spectrum(k,dt)
    order=np.argsort(omega)
    dw=omega[1]-omega[0]
    cs=CubicSpline(omega[order],(p[...,order]*2*np.pi/dw).reshape(-1,len(omega)),axis=-1)
    ks=CubicSpline(omega[order],kp[order]*2*np.pi/dw)
    wt=(dmft.fractions[:,None]*one['weights']).reshape(-1)
    af=one['alpha'].reshape(-1)
    vf=one['covariance'][...,0].reshape(-1)
    a2=wt@(af*af)
    kap=dmft.g**2*a2
    cutoff=scaled_cutoff*scale
    nodes,weights=roots_legendre(n_outer)
    w1=(nodes+1)*cutoff/2
    w_outer=weights*cutoff/2
    center=-kap*w1/(1+w1*w1)
    width=1-kap/(1+w1*w1)
    low=np.arctan((-cutoff-center)/width)
    high=np.arctan((cutoff-center)/width)
    zn,zw=roots_legendre(n_inner)
    theta=(high-low)[:,None]*(zn[None,:]+1)/2+low[:,None]
    w2=center[:,None]+width[:,None]*np.tan(theta)
    weight_theta=(high-low)[:,None]*zw[None,:]/2
    # F(w1,w2) dw2=(1+w2^2)/width dtheta.
    measure=2/(2*np.pi)**2*w_outer[:,None]*weight_theta*(1+w2*w2)/width[:,None]
    c1=cs(w1);c2=cs(w2)
    qraw=np.einsum('z,zi,zij->ij',wt,c1,c2,optimize=True)
    energy=float(np.sum(measure*qraw))
    valid=(wt>1e-15)&(vf>1e-200)
    qnorm=np.einsum('z,zi,zij->ij',wt[valid]/vf[valid]**2,c1[valid],c2[valid],optimize=True)
    qmixed=np.einsum('z,zi,zij->ij',wt[valid]/vf[valid],c1[valid],c2[valid],optimize=True)
    normalized_response=np.sum(wt[valid]*af[valid]**2/vf[valid])
    ratio=normalized_response/a2
    hprod=1/((1+1j*w1[:,None])*(1+1j*w2))
    zz=kap*hprod
    corr_numerator=np.abs(1-zz)**2*qnorm+2*ratio*(zz-np.abs(zz)**2).real*qmixed+ratio**2*np.abs(zz)**2*qraw
    corr_energy=float(np.sum(measure*corr_numerator))
    input_numerator=(1-np.abs(zz)**2)*ks(w1)[:,None]*ks(w2)+dmft.g**4*np.abs(hprod)**2*qraw
    input_energy=float(np.sum(measure*input_numerator))
    v=wt@vf
    return {'g':dmft.g,'q':dmft.dynamic_variance,'gap':gap,'mean_variance':float(v),
            'dimension_fraction':float(v*v/energy),'covariance_energy':energy,
            'correlation_dimension_fraction':1/corr_energy,
            'input_dimension_fraction':float(dmft.dynamic_variance**2/input_energy),
            'quadrature':{'du':du,'scaled_horizon':scaled_horizon,'n_outer':n_outer,'n_inner':n_inner,'scaled_cutoff':scaled_cutoff},
            'min_interpolated_spectrum':float(min(c1.min(),c2.min()))}
