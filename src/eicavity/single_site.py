"""One-site DMFT for phi(x)=1+erf(x/sqrt(2))=2 Phi(x).

All Gaussian integrals use normalized quadrature weights. No stochastic CDF
evaluation is used. The scalar solver covers identical rows of the variance
profile, including multiple populations with different prescribed mean rates.
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import least_squares, brentq
from scipy.special import ndtr, ndtri, roots_legendre, roots_hermitenorm


def rate(x):
    return 2.0 * ndtr(x)


def rate_prime(x):
    return np.sqrt(2.0 / np.pi) * np.exp(-0.5 * np.asarray(x) ** 2)


def normal_quadrature(n=80):
    z, w = roots_hermitenorm(n)
    return z, w / np.sqrt(2.0 * np.pi)


_gx, _gw = roots_legendre(64)
_unit_x, _unit_w = (_gx + 1) / 2, _gw / 2


def centered_moment(mean, total_variance, cross_covariance, validate=True):
    """Cov[phi(X),phi(Y)] for equal Gaussian marginals.

    Price's theorem gives a deterministic one-dimensional integral. Supports
    vector covariance arguments and scalar marginal parameters.
    """
    mean = np.asarray(mean)
    scale = 1 + np.asarray(total_variance)
    c = np.asarray(cross_covariance)
    if validate and np.any(np.abs(c) > np.asarray(total_variance) + 1e-9):
        raise ValueError("Cross covariance exceeds the marginal variance")
    angle = np.arcsin(np.clip(c / scale, -1, 1))
    theta = angle[..., None] * _unit_x
    value = 2 / np.pi * angle * np.sum(
        _unit_w * np.exp(-(mean**2 / scale)[..., None] / (1 + np.sin(theta))),
        axis=-1,
    )
    return value


def raw_moment(mean, total_variance, cross_covariance, mean_rate=None):
    m=rate(np.asarray(mean)/np.sqrt(1+np.asarray(total_variance))) if mean_rate is None else mean_rate
    return m**2+centered_moment(mean,total_variance,cross_covariance)


def raw_derivative(mean, total_variance, cross_covariance):
    """Derivative of raw_moment with respect to cross covariance."""
    v = 1 + np.asarray(total_variance)
    c = np.asarray(cross_covariance)
    return 2 / np.pi * np.exp(-np.asarray(mean) ** 2 / (v + c)) / np.sqrt(v*v-c*c)


@dataclass
class ScalarDMFT:
    g: float
    mean_rates: np.ndarray
    fractions: np.ndarray
    static_variance: float
    dynamic_variance: float
    mean_inputs: np.ndarray
    static_residual: float
    energy_residual: float
    response_radius: float
    fixed_point_radius: float

    def population_covariance(self, k):
        d, q = self.static_variance, self.dynamic_variance
        return np.array([
            raw_moment(h, d+q, d+np.asarray(k), m) - raw_moment(h, d+q, d, m)
            for h, m in zip(self.mean_inputs, self.mean_rates)
        ])

    def mean_covariance(self, k):
        return self.fractions @ self.population_covariance(k)

    def kernel(self, dt=0.05, max_time=None):
        """Monotone homoclinic trajectory using its first-integral quadrature.

        A direct second-order IVP drifts onto the growing tail. Instead solve
        the negative square-root branch of the conserved energy; use a dense
        interpolation of the exact Price integral.
        """
        q = self.dynamic_variance
        decay = np.sqrt(max(1-self.response_radius, 1e-12))
        if max_time is None:
            max_time = max(80.0, 20.0/decay)
        times = np.arange(int(np.ceil(max_time/dt))+1) * dt
        # The energy at k is k^2 [1/2 - g^2 int_0^1(1-x) F'(d+k*x) dx].
        def speed(k):
            kk = float(np.clip(k, 0, q))
            d, v = self.static_variance, self.static_variance+q
            integral = sum(f * np.sum(_unit_w * (1-_unit_x) * raw_derivative(h,v,d+kk*_unit_x))
                           for f,h in zip(self.fractions,self.mean_inputs))
            return kk * np.sqrt(max(1-2*self.g**2*integral,0.0))
        def initial_ode(t,y):
            return [y[1],y[0]-self.g**2*float(self.mean_covariance(y[0]))]
        def leave_turning_point(t,y):
            return y[0]-.9*q
        leave_turning_point.terminal=True
        leave_turning_point.direction=-1
        initial=solve_ivp(initial_ode,(0,times[-1]),[q,0.],events=leave_turning_point,
                          rtol=1e-11,atol=min(1e-12,q*1e-11),dense_output=True,max_step=.1/decay)
        if len(initial.t_events[0])!=1:
            raise RuntimeError("Kernel failed to leave its initial turning point")
        tc=initial.t_events[0][0]
        solution=solve_ivp(lambda t,y: [-speed(y[0])],(tc,times[-1]),[.9*q],
                           rtol=1e-10,atol=min(1e-12,q*1e-11),dense_output=True,max_step=0.1/decay)
        k=np.empty_like(times)
        before=times<=tc
        k[before]=initial.sol(times[before])[0]
        k[~before]=solution.sol(times[~before])[0]
        return times,k

    def conditional(self, kernel, n_hermite=96):
        """Quenched label, conditional mean, gain, and centered covariance."""
        z,w=normal_quadrature(n_hermite)
        q,d=self.dynamic_variance,self.static_variance
        mu=self.mean_inputs[:,None]+np.sqrt(d)*z[None,:]
        means=rate(mu/np.sqrt(1+q))
        alpha=np.sqrt(2/np.pi/(1+q))*np.exp(-mu*mu/(2*(1+q)))
        cov=np.empty((len(self.fractions),len(z),len(kernel)))
        for a in range(len(self.fractions)):
            for j in range(len(z)):
                cov[a,j]=centered_moment(mu[a,j],q,np.asarray(kernel))
        return {"z":z,"weights":w,"mu":mu,"means":means,"alpha":alpha,"covariance":cov}


def solve_scalar(g, mean_rates=(0.3,), fractions=(1.0,), initial=None):
    """Solve balance-constrained DMFT when G_ab=g^2 for all populations.

    The mean rates are selected by the balance equation. The O(1) residual
    input adjusts so that the Gaussian marginal has those mean rates.
    """
    ms,fs=np.asarray(mean_rates,dtype=float),np.asarray(fractions,dtype=float)
    if np.any(ms<=0) or np.any(ms>=2) or not np.isclose(fs.sum(),1):
        raise ValueError("Require rates in (0,2) and population fractions summing to one")
    a=ndtri(ms/2)
    def equations(log_dq):
        d,q=np.exp(log_dq)
        v=d+q
        hs=a*np.sqrt(1+v)
        m2=sum(f*raw_moment(h,v,d,m) for f,h,m in zip(fs,hs,ms))
        integral=sum(f*np.sum(_unit_w*(1-_unit_x)*raw_derivative(h,v,d+q*_unit_x))
                     for f,h in zip(fs,hs))
        return [np.log(d/(g*g*m2)),1-2*g*g*integral]
    # The nonzero fluctuation solution is separate from the fixed-point root.
    # Division by q^2 in the energy equation removes the automatic q=0 root.
    x0=np.log(initial if initial is not None else [g*g*float(fs@ms),max(0.1,g*g*0.2)])
    opt=least_squares(equations,x0,bounds=([-15,-18],[16,16]),xtol=2e-12,ftol=2e-12,gtol=2e-12,max_nfev=500)
    d,q=np.exp(opt.x)
    hs=a*np.sqrt(1+d+q)
    residual=equations(opt.x)
    def fixed_residual(v):
        hh=a*np.sqrt(1+v)
        return v-g*g*sum(f*raw_moment(h,v,v,m) for f,h,m in zip(fs,hh,ms))
    fp=brentq(fixed_residual,1e-12,4*g*g+1)
    fph=a*np.sqrt(1+fp)
    fpr=g*g*sum(f*raw_derivative(h,fp,fp) for f,h in zip(fs,fph))
    radius=g*g*sum(f*raw_derivative(h,d+q,d) for f,h in zip(fs,hs))
    if not opt.success or np.max(np.abs(residual))>1e-7 or q<1e-6:
        raise ValueError(f"No resolved chaotic solution: g={g}, fixed-point radius={fpr:.6g}, residual={residual}")
    return ScalarDMFT(g,ms,fs,d,q,hs,float(residual[0]),float(residual[1]),float(radius),float(fpr))
