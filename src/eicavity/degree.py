"""Dense-network degree dispersion and its conditional Gaussian one-site law.

chi=1 is independent Bernoulli degree variance. chi=0 has fixed counts up to
one rounding edge. The dynamic covariance equation is unchanged at leading
order; static disorder uses Var(m)+chi*mean(m)^2.
"""
import numpy as np
from scipy.optimize import least_squares,brentq
from scipy.special import ndtri
from .single_site import ScalarDMFT,raw_moment,raw_derivative,_unit_x,_unit_w


def degree_counts(rng,size,pool_size,probability,chi):
    """Integer counts with exact mean Mp and asymptotic dispersion chi.

    For chi<=1, randomize only L=round(chi*M) Bernoulli trials and stochastically
    round the remaining mean. For chi>1 use a beta-binomial law with exactly
    the requested degree variance. Neighbor subsets must then be uniform.
    """
    m,p=pool_size,probability
    if chi<0 or chi>=m or not 0<p<1:raise ValueError('Require 0<=chi<M and 0<p<1')
    if chi<=1:
        trials=round(chi*m)
        base=(m-trials)*p
        integer=int(np.floor(base));fraction=base-integer
        counts=integer+rng.binomial(1,fraction,size)+rng.binomial(trials,p,size)
        variance=trials*p*(1-p)+fraction*(1-fraction)
    else:
        concentration=(m-chi)/(chi-1)
        probabilities=rng.beta(p*concentration,(1-p)*concentration,size)
        counts=rng.binomial(m,probabilities)
        variance=chi*m*p*(1-p)
    return counts,float(variance/(m*p*(1-p)))


def fixed_point(g,chi,mean_rates=(.3,),fractions=(1.,)):
    ms,fs=np.asarray(mean_rates),np.asarray(fractions)
    a=ndtri(ms/2)
    def quantities(d):
        h=a*np.sqrt(1+d)
        source=sum(f*(raw_moment(hh,d,d,m)-(1-chi)*m*m)
                   for f,hh,m in zip(fs,h,ms))
        slope=sum(f*raw_derivative(hh,d,d) for f,hh in zip(fs,h))
        return h,source,slope
    if chi==0:
        d=0.
    else:
        d=brentq(lambda d:d-g*g*quantities(d)[1],0,4*g*g*(1+chi)+1,xtol=2e-13)
    h,source,slope=quantities(d)
    return d,h,g*g*slope


def solve_degree_scalar(g,chi,mean_rates=(.3,),fractions=(1.,),initial=None):
    if chi<0:raise ValueError('Degree dispersion must be nonnegative')
    ms,fs=np.asarray(mean_rates,dtype=float),np.asarray(fractions,dtype=float)
    a=ndtri(ms/2)
    def quantities(d,q):
        h=a*np.sqrt(1+d+q)
        source=sum(f*(raw_moment(hh,d+q,d,m)-(1-chi)*m*m)
                   for f,hh,m in zip(fs,h,ms))
        integral=sum(f*np.sum(_unit_w*(1-_unit_x)*raw_derivative(hh,d+q,d+q*_unit_x))
                     for f,hh in zip(fs,h))
        return h,source,integral
    if chi==0:
        d=0.
        def energy(q):return 1-2*g*g*quantities(0,q)[2]
        if energy(0)>=0:raise ValueError('No unstable homogeneous fixed point at this coupling')
        q=brentq(energy,0,4*g*g+1,xtol=1e-12)
    else:
        def equations(parameters):
            d,q=np.exp(parameters)
            h,source,integral=quantities(d,q)
            if source<=0:raise ValueError('Static covariance source is nonpositive')
            return [np.log(d/(g*g*source)),1-2*g*g*integral]
        guess=initial if initial is not None else [max(.01,chi*g*g*.25),max(.1,g*g*.25)]
        fit=least_squares(equations,np.log(guess),bounds=([-25,-25],[18,18]),
                          xtol=2e-12,ftol=2e-12,gtol=2e-12,max_nfev=500)
        d,q=np.exp(fit.x)
        if np.max(np.abs(equations(fit.x)))>1e-7 or q<1e-7:
            raise RuntimeError('No resolved positive chaotic branch')
    h,source,integral=quantities(d,q)
    radius=g*g*sum(f*raw_derivative(hh,d+q,d) for f,hh in zip(fs,h))
    fp,_,fp_radius=fixed_point(g,chi,ms,fs)
    return ScalarDMFT(g,ms,fs,d,q,h,float(d-g*g*source),float(1-2*g*g*integral),
                     float(radius),float(fp_radius))


def critical_degree(chi,mean_rate=.3):
    """Uniform-profile critical point at a fixed population mean rate."""
    a=float(ndtri(mean_rate/2))
    def quantities(d):
        h=a*np.sqrt(1+d)
        slope=raw_derivative(h,d,d)
        source=raw_moment(h,d,d,mean_rate)-(1-chi)*mean_rate**2
        return h,float(slope),float(source)
    if chi==0:d=0.
    else:
        d=brentq(lambda d:d*quantities(d)[1]-quantities(d)[2],0,1e6,xtol=1e-13)
    h,a2,source=quantities(d)
    g=1/np.sqrt(a2)
    a4=(2/np.pi)**2/np.sqrt(1+4*d)*np.exp(-2*h*h/(1+4*d))
    constant=4*np.pi**2/3-12
    # With t=d/(1+d), t P(t)-int_0^t P(s)ds = chi*m^2.
    # P'(0)=a^2*phi'(a)^2 for this positive-erf transfer.
    p0=2/np.pi*np.exp(-a*a)
    return {'chi':float(chi),'critical_g':float(g),'static_variance':float(d),
            'alpha_second':a2,'alpha_fourth':float(a4),
            'dimension_prefactor_in_gap':float(a2*a2/a4/constant),
            'small_chi_static_variance_coefficient':float(np.sqrt(2*mean_rate**2/(a*a*p0)))
             if a!=0 else None}
