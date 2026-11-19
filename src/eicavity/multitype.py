"""Population DMFT boundary-value solver and block cavity quadrature."""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_bvp
from scipy.special import ndtri
from .single_site import solve_scalar,normal_quadrature,centered_moment,raw_moment,rate
from .pair import even_spectrum


@dataclass
class PopulationDMFT:
    variance_profile: np.ndarray
    fractions: np.ndarray
    mean_rates: np.ndarray
    static_variances: np.ndarray
    dynamic_variances: np.ndarray
    mean_inputs: np.ndarray
    time: np.ndarray
    kernels: np.ndarray
    bvp_residual: float
    static_residual: float
    boundary_residual: float

    def conditional(self,n_hermite=96):
        z,w=normal_quadrature(n_hermite)
        mu=self.mean_inputs[:,None]+np.sqrt(self.static_variances[:,None])*z
        q=self.dynamic_variances[:,None]
        means=rate(mu/np.sqrt(1+q))
        alpha=np.sqrt(2/np.pi/(1+q))*np.exp(-mu*mu/(2*(1+q)))
        cov=np.array([[centered_moment(mu[a,j],q[a,0],self.kernels[a]) for j in range(len(z))]
                      for a in range(len(self.fractions))])
        return {'z':z,'weights':w,'mu':mu,'means':means,'alpha':alpha,'covariance':cov}


def solve_populations(variance_profile,mean_rates=(.3,.3),fractions=(.8,.2),dt=.08,horizon=100.,tol=2e-7,continuation_steps=6,degree_dispersion=None):
    G,ms,fs=map(lambda a:np.asarray(a,dtype=float),(variance_profile,mean_rates,fractions))
    p=len(ms)
    if G.shape!=(p,p) or np.any(G<=0) or not np.isclose(fs.sum(),1):
        raise ValueError('Positive p-by-p variance profile and normalized fractions required')
    a=ndtri(ms/2)
    base_g=np.sqrt(fs@G@fs)
    chi=np.ones((p,p)) if degree_dispersion is None else np.broadcast_to(degree_dispersion,(p,p)).astype(float)
    if np.any(chi<0):raise ValueError('Degree dispersion must be nonnegative')
    fixed_counts=bool(np.all(chi==0))
    if degree_dispersion is None:
        scalar=solve_scalar(base_g,ms,fs)
    else:
        from .degree import solve_degree_scalar
        scalar=solve_degree_scalar(base_g,float(fs@chi@fs),ms,fs)
    t,k=scalar.kernel(dt=.25,max_time=horizon)
    y=np.vstack([np.tile(k,(p,1)),np.tile(np.gradient(k,t),(p,1))])
    params=np.log(np.repeat(scalar.dynamic_variance,p) if fixed_counts else
                  np.r_[np.repeat(scalar.static_variance,p),np.repeat(scalar.dynamic_variance,p)])
    last=None
    for step in range(1,continuation_steps+1):
        Gnow=base_g**2+(G-base_g**2)*step/continuation_steps
        W=Gnow@np.diag(fs)
        def decode(par):
            d,q=(np.zeros(p),np.exp(par)) if fixed_counts else (np.exp(par[:p]),np.exp(par[p:]))
            h=a*np.sqrt(1+d+q)
            return d,q,h
        def ode(t,y,par):
            d,q,h=decode(par)
            cov=np.array([centered_moment(h[b],d[b]+q[b],d[b]+y[b],validate=False)-centered_moment(h[b],d[b]+q[b],d[b]) for b in range(p)])
            return np.vstack([y[p:],y[:p]-W@cov])
        def bc(ya,yb,par):
            d,q,h=decode(par)
            m2=np.array([raw_moment(h[b],d[b]+q[b],d[b],ms[b]) for b in range(p)])
            static_source=W@m2+(W*(chi-1))@(ms*ms)
            dynamic=np.r_[ya[p:],yb[:p],ya[:p]-q]
            return dynamic if fixed_counts else np.r_[dynamic,(d-static_source)/np.maximum(d,1)]
        last=solve_bvp(ode,bc,t,y,p=params,tol=tol,bc_tol=min(tol,1e-10),max_nodes=6000)
        if not last.success:
            raise RuntimeError(f'DMFT boundary-value solver failed at continuation {step}: {last.message}')
        t,y,params=last.x,last.y,last.p
    d,q,h=decode(params)
    times=np.arange(round(horizon/dt)+1)*dt
    kernels=last.sol(times)[:p]
    if np.any(kernels < -1e-8) or np.min(q)<1e-5:
        raise RuntimeError('Solution is not a resolved positive chaotic branch')
    kernels[:,-1]=0.
    m2=np.array([raw_moment(h[b],d[b]+q[b],d[b],ms[b]) for b in range(p)])
    static_source=W@m2+(W*(chi-1))@(ms*ms)
    return PopulationDMFT(G,fs,ms,d,q,h,times,kernels,float(np.max(last.rms_residuals)),
                          float(np.max(np.abs(d-static_source))),float(np.max(np.abs(bc(last.y[:,0],last.y[:,-1],last.p)))))


def population_dimension(dmft,n_hermite=96,max_frequency=12.):
    one=dmft.conditional(n_hermite)
    dt=dmft.time[1]-dmft.time[0]
    omega,spectrum=even_spectrum(one['covariance'],dt)
    keep=np.abs(omega)<=max_frequency
    omega,spectrum=omega[keep],spectrum[...,keep]
    G,f=dmft.variance_profile,dmft.fractions
    p=len(f);w=one['weights']; alpha2=one['alpha']**2@w
    h=1/(1+1j*omega)
    Q=np.array([(c.T*w)@c for c in spectrum])
    integrated=np.zeros((p,p),dtype=complex)
    F=np.diag(f)
    Dweights=np.eye(p)[None,None,:,:]
    for lo in range(0,len(omega),32):
        hi=min(len(omega),lo+32)
        product=h[lo:hi,None]*h[None,:]
        u=product[:,:,None]*alpha2
        # R=(I-U G F)^-1. The type axes are last; frequency axes are first.
        R=np.linalg.inv(np.eye(p)-u[..., :,None]*(G@F))
        q=np.moveaxis(Q[:,lo:hi,:]/f[:,None,None],0,-1)
        full=np.einsum('...ac,...c,...bc->...ab',R,q,R.conj(),optimize=True)
        integrated+=full.sum(axis=(0,1))
    v=one['covariance'][...,0]
    means=v@w;seconds=v*v@w
    pair=integrated-np.diag(seconds/f)
    energy=float((f@integrated@f).real)
    radius=float(np.max(np.abs(np.linalg.eigvals(np.diag(alpha2)@G@F))))
    return {'dimension_fraction':float((f@means)**2/energy),'covariance_energy':energy,
            'mean_variances':means.tolist(),'variance_seconds':seconds.tolist(),
            'pair_covariance_energy':pair.real.tolist(),
            'within_population_dimension_fractions':(means**2/(seconds+f*np.diag(pair).real)).tolist(),
            'response_radius':radius,'min_spectral_mass':float(spectrum.min()),
            'imaginary_energy_residual':float(np.max(np.abs(integrated.imag))),**one}
