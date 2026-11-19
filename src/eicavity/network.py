"""Strict EI and moment-matched Gaussian network simulations."""
from dataclasses import dataclass,asdict
import time
import numpy as np
from .single_site import rate,rate_prime


@dataclass
class NetworkConfig:
    n: int=1000
    g: float=4.0
    mean_rate: float=.3
    seed: int=1
    ensemble: str='bernoulli'
    dt: float=.05
    burn: float=500.
    duration: float=4000.
    sample_dt: float=.25
    mean_scale: float=1.
    fraction_e: float=.8
    variance_profile: list | None=None
    mean_connectivity: list | None=None
    degree_dispersion: float | None=None


def make_network(config,mean_input):
    c=config
    ne=round(c.n*c.fraction_e)
    labels=np.r_[np.zeros(ne,dtype=int),np.ones(c.n-ne,dtype=int)]
    fractions=np.bincount(labels,minlength=2)/c.n
    j0=c.mean_scale*np.array([[1.,-6.],[2.,-10.]] if c.mean_connectivity is None else c.mean_connectivity)
    if not np.all(np.linalg.eigvals(j0@np.diag(fractions)).real<0):
        raise ValueError('Coherent mean connectivity must be Hurwitz')
    g2=np.full((2,2),c.g*c.g) if c.variance_profile is None else np.asarray(c.variance_profile)
    probs=j0*j0/(j0*j0+g2)
    weights=np.abs(j0)/probs
    rng=np.random.default_rng(c.seed)
    j=np.empty((c.n,c.n),dtype=np.float64,order='C')
    degree_chi=np.zeros((2,2));degree_mean=np.zeros((2,2));degree_variance=np.zeros((2,2))
    for a in range(2):
        ia=np.flatnonzero(labels==a)
        for b in range(2):
            ib=np.flatnonzero(labels==b)
            if c.ensemble=='bernoulli':
                block=(rng.random((len(ia),len(ib)))<probs[a,b])*weights[a,b]*(1 if b==0 else -1)
            elif c.ensemble=='gaussian':
                block=j0[a,b]+np.sqrt(g2[a,b])*rng.normal(size=(len(ia),len(ib)))
            elif c.ensemble=='degree':
                from .degree import degree_counts
                if c.degree_dispersion is None:raise ValueError('Degree ensemble requires degree_dispersion')
                counts,degree_chi[a,b]=degree_counts(rng,len(ia),len(ib),probs[a,b],c.degree_dispersion)
                adjacency=np.zeros((len(ia),len(ib)),dtype=bool)
                for row,count in enumerate(counts):
                    if count<=len(ib)//2:
                        adjacency[row,rng.choice(len(ib),count,replace=False)]=True
                    else:
                        adjacency[row]=True
                        adjacency[row,rng.choice(len(ib),len(ib)-count,replace=False)]=False
                block=adjacency*weights[a,b]*(1 if b==0 else -1)
                degree_mean[a,b]=counts.mean();degree_variance[a,b]=counts.var()
            else:
                raise ValueError('Unknown connectivity ensemble')
            j[np.ix_(ia,ib)]=block/np.sqrt(c.n)
    # Keep iid diagonal entries; removing autapses changes only a finite-size
    # term but would need a different mean compensation convention.
    external=-np.sqrt(c.n)*(j0@(fractions*c.mean_rate))+np.asarray(mean_input)
    drive=external[labels]
    init_rng=np.random.default_rng(c.seed+1000003)
    x=mean_input[labels]+init_rng.normal(size=c.n)*c.g
    info={'j0':j0,'probabilities':probs,'nonzero_weights_scaled':weights,'fractions':fractions,'external':external}
    if c.ensemble=='degree':
        info.update(degree_expected_dispersion=degree_chi,degree_empirical_means=degree_mean,
                    degree_empirical_variances=degree_variance)
    return j,drive,x,labels,info


def simulate(config,mean_input):
    """Explicit midpoint integration, with the tangent of the same step map.

    The reported Lyapunov exponent uses a single tangent vector after burn-in.
    Trajectories are sampled only after burn-in, in float32 for storage; the
    dynamics and covariance analysis use float64.
    """
    c=config
    j,drive,x,labels,info=make_network(c,np.asarray(mean_input))
    stride=round(c.sample_dt/c.dt)
    if not np.isclose(stride*c.dt,c.sample_dt):
        raise ValueError('sample_dt must be an integer multiple of dt')
    nb,ns=round(c.burn/c.dt),round(c.duration/c.dt)
    if ns%stride:
        raise ValueError('duration must be a multiple of sample_dt')
    trajectory=np.empty((ns//stride,c.n),dtype=np.float32)
    rng=np.random.default_rng(c.seed+2000003)
    tangent=rng.normal(size=c.n)
    tangent/=np.linalg.norm(tangent)
    lyap=[]
    tic=time.monotonic()
    sample=0
    for step in range(nb+ns):
        k1=-x+j@rate(x)+drive
        mid=x+.5*c.dt*k1
        v1=-tangent+j@(rate_prime(x)*tangent)
        vmid=tangent+.5*c.dt*v1
        tangent+=c.dt*(-vmid+j@(rate_prime(mid)*vmid))
        x+=c.dt*(-mid+j@rate(mid)+drive)
        norm=np.linalg.norm(tangent)
        tangent/=norm
        if step>=nb:
            lyap.append(np.log(norm)/c.dt)
            if (step-nb+1)%stride==0:
                trajectory[sample]=x
                sample+=1
    elapsed=time.monotonic()-tic
    lyap=np.asarray(lyap)
    return {'trajectory':trajectory,'connectivity':j,'labels':labels,'drive':drive,
            'lyapunov_blocks':np.array([np.mean(v) for v in np.array_split(lyap,8)]),
            'elapsed_seconds':elapsed,'network_info':info,'config':asdict(c)}
