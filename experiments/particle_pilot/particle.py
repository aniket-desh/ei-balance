"""Finite-neighbor quantile particles coupled to a shared temporal kernel per type."""
import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import ndtri
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
DEGREE = HERE.parent / "degree"
SRC = HERE.parents[1] / "recovered/bundle/ei_full/ei_balance_handoff/src"
sys.path.insert(0, str(SRC))
from eicavity.degree import degree_counts
from eicavity.multitype import population_dimension
from eicavity.single_site import centered_moment, rate

CONFIG = json.loads((DEGREE / "predictions.json").read_text())
G = np.array(CONFIG["G"])
J0 = np.array(CONFIG["J0"])
F = np.array(CONFIG["fractions"])
TARGET = np.array(CONFIG["mean_rates"])
N = 2000
SIZES = (N*F).astype(int)
PROB = J0**2/(J0**2+G)
HORIZON, TOL = 100., 3e-5
QUANTILES = (.1,.25,.5,.75,.9)


def baseline(chi):
    return next(r for r in CONFIG["rows"] if r["chi"] == chi)


def target_offset(shape, q, target):
    scale = np.sqrt(1+q)
    return brentq(lambda h: np.mean(rate((shape+h)/scale))-target,
                  -shape.max()-13*scale, -shape.min()+13*scale, xtol=2e-13)


def neighbor_rows(particles, chi, seed):
    """Each bin labels exactly M/P physical sources with the same rate."""
    rng = np.random.default_rng(seed)
    counts = np.empty((2,2,particles), dtype=int)
    rows = np.empty((2,2,particles,particles))
    for a,b in np.ndindex(2,2):
        M,p = int(SIZES[b]), PROB[a,b]
        if M % particles:
            raise ValueError("Physical source sizes must be divisible by the particle count.")
        colors = np.full(particles, M//particles)
        K,_ = degree_counts(rng,particles,M,p,chi)
        multiplicities = np.vstack([rng.multivariate_hypergeometric(colors,int(k)) for k in K])
        assert np.all(multiplicities.sum(1) == K) and np.all(multiplicities <= colors)
        # Uniform subsets of physical indices, conditional on K; signed weight per contact.
        rows[a,b] = multiplicities * (J0[a,b]/(p*np.sqrt(N)))
        counts[a,b] = K
    return rows, counts


def empirical_update(rows, rates):
    summed = np.einsum("abij,bj->ai",rows,rates,optimize=True)
    return np.sort(summed-summed.mean(1,keepdims=True),axis=1)


def conditional_law(shape, q, target):
    h = target_offset(shape,q,target)
    mu = h+shape
    # Price's formula after c=(1+q) sin(theta); integrate the smooth derivative.
    extent = min(np.pi*.5-1e-6, np.arcsin(q/(1+q)) + .06)
    theta = np.linspace(-.06,extent,77)
    derivative = 2/np.pi*np.mean(np.exp(-mu[:,None]**2/((1+q)*(1+np.sin(theta)))),axis=0)
    primitive = CubicSpline(theta,derivative).antiderivative()
    at_zero = primitive(0.)
    def cov(k):
        angle = np.arcsin(np.clip(k/(1+q),-.999999,.999999))
        return primitive(np.clip(angle,theta[0],theta[-1]))-at_zero
    return h,cov


def temporal_solve(shape, mesh, previous, initial_q):
    lookup = lru_cache(maxsize=32)(lambda b,q: conditional_law(shape[b],q,TARGET[b]))
    def ode(t,y,par):
        q = np.exp(np.clip(par,-20,8))
        c = np.array([lookup(b,q[b])[1](y[b]) for b in range(2)])
        return np.vstack((y[2:],y[:2]-(G*F)@c))
    def boundary(left,right,par):
        return np.r_[left[2:],right[:2],left[:2]-np.exp(np.clip(par,-20,8))]
    sol = solve_bvp(ode,boundary,mesh,previous,p=np.log(initial_q),
                    tol=TOL,bc_tol=1e-9,max_nodes=2200)
    q = np.exp(sol.p)
    if not sol.success or np.any(q<1e-4) or np.any(sol.y[:2]<-1e-7):
        raise RuntimeError(f"Temporal branch unresolved: {sol.message}, q={q}")
    h = np.array([lookup(b,q[b])[0] for b in range(2)])
    return sol,q,h


@dataclass
class ParticleDMFT:
    mu: np.ndarray
    q: np.ndarray
    time: np.ndarray
    kernel: np.ndarray

    variance_profile = G
    fractions = F

    def conditional(self,n_hermite=None):
        # The existing block cavity uses these equally weighted empirical types.
        P = self.mu.shape[1]
        q = self.q[:,None]
        return {"weights":np.full(P,1/P),"mu":self.mu,
                "means":rate(self.mu/np.sqrt(1+q)),
                "alpha":np.sqrt(2/np.pi/(1+q))*np.exp(-self.mu**2/(2*(1+q))),
                "covariance":np.array([[centered_moment(m,self.q[a],self.kernel[a])
                                        for m in pop] for a,pop in enumerate(self.mu)])}


def solve_case(P,chi,seed,artifact,dt=.1):
    rows,counts = neighbor_rows(P,chi,seed)
    ref = baseline(chi)
    shape = np.sqrt(ref["static_variances"])[:,None]*ndtri((np.arange(P)+.5)/P)
    mesh = np.arange(0,HORIZON+.25,.5)
    with np.load(DEGREE/f"chi{chi:g}_one_site.npz") as z:
        archived = np.array([np.interp(mesh,z["time"],k) for k in z["input_kernels"]])
    q = archived[:,0]
    previous = np.vstack((archived,np.gradient(archived,mesh,axis=1)))
    history = []
    for iteration in range(1,501):
        try:
            sol,q,h = temporal_solve(shape,mesh,previous,q)
        except RuntimeError as exc:
            return {"status":"temporal branch unresolved","iterations":iteration,
                    "history":history,"detail":str(exc)}
        previous = sol.sol(mesh)
        mu = shape+h[:,None]
        rates = rate(mu/np.sqrt(1+q[:,None]))
        update = empirical_update(rows,rates)
        residual = float(np.max(np.abs(update-shape)))
        history.append((residual,*q))
        if residual < 2e-5:
            break
        shape = .55*shape+.45*update
    else:
        return {"status":"static iteration did not converge","iterations":iteration,
                "history":history,"static_residual_inf":residual}
    t = np.arange(round(HORIZON/dt)+1)*dt
    kernel = sol.sol(t)[:2]
    kernel[:,-1] = 0.
    # Check the converged ODE against the original 64-node Price integrals,
    # including at zero lag where q and the empirical distribution enter together.
    check_t = np.r_[0.,np.linspace(.131,HORIZON-.27,33)]
    check_k = sol.sol(check_t)[:2]
    ccheck = np.array([[np.mean([centered_moment(m,q[b],k) for m in mu[b]])
                        for k in check_k[b]] for b in range(2)])
    reference_ode_error = float(np.max(np.abs(sol.sol(check_t,nu=1)[2:] -
                                        (check_k-(G*F)@ccheck))))
    model = ParticleDMFT(mu,q,t,kernel)
    pair = population_dimension(model,max_frequency=12.)
    alpha2 = (pair["alpha"]**2).mean(1)
    radius = pair["response_radius"]
    tail_fraction = float(np.max(np.abs(sol.sol(.8*HORIZON)[:2])/q))
    admitted = (radius<1 and tail_fraction<.002 and pair["min_spectral_mass"]>-1e-7 and
                reference_ode_error<3e-4 and np.all(np.isfinite(pair["pair_covariance_energy"])))
    np.savez_compressed(artifact,mu=mu,mean_rate=rates,q=q,t=t,input_kernel=kernel,
                        counts=counts,conditional_gain=pair["alpha"],history=np.asarray(history))
    result = {"status":"converged","iterations":iteration,"history":history,
              "target_rate_error_inf":float(np.max(np.abs(rates.mean(1)-TARGET))),
              "static_residual_inf":residual,"dynamic_bvp_rms_residual_max":float(sol.rms_residuals.max()),
              "reference_ode_error":reference_ode_error,"boundary_residual_inf":float(max(
                  np.max(np.abs(sol.y[2:,0])),np.max(np.abs(sol.y[:2,-1])),np.max(np.abs(sol.y[:2,0]-q)))),
              "mean_offset":h.tolist(),"q":q.tolist(),"delta":shape.var(1).tolist(),
              "mu_quantiles":np.quantile(mu,QUANTILES,axis=1).T.tolist(),
              "mean_squared_gain":alpha2.tolist(),"response_radius":radius,
              "min_spectral_mass":pair["min_spectral_mass"],"tail_fraction_at_80":tail_fraction,
              "pr_admitted":bool(admitted),
              "dimensions":None,"row_sample_count_means":counts.mean(2).tolist(),
              "row_sample_count_variances":counts.var(2).tolist(),
              "artifact":artifact.name}
    if admitted:
        result["dimensions"]=[*pair["within_population_dimension_fractions"],pair["dimension_fraction"]]
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--particles",nargs="+",type=int,default=[100,200,400])
    parser.add_argument("--chi",nargs="+",type=int,default=[0,1,2])
    parser.add_argument("--seeds",nargs="+",type=int,default=[8101,8102,8103])
    args = parser.parse_args()
    def check_rss(signum,frame):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)
        if rss > 1.5*2**30:
            raise MemoryError("Particle pilot exceeded 1.5 GiB")
    signal.signal(signal.SIGALRM,check_rss)
    signal.setitimer(signal.ITIMER_REAL,.25,.25)
    (HERE/"runs").mkdir(exist_ok=True)
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with threadpool_limits(1):
        for chi in args.chi:
            for P in args.particles:
                for seed in args.seeds:
                    prefix = HERE/"runs"/f"chi{chi:g}_P{P}_seed{seed}"
                    out = prefix.with_suffix(".json")
                    if out.exists() and json.loads(out.read_text()).get("code_sha256")==digest:
                        continue
                    start = time.monotonic()
                    try:
                        row = solve_case(P,chi,seed,prefix.with_suffix(".npz"))
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                        row = {"status":"failed","detail":str(exc)}
                    row.update(N=N,chi=chi,particles_per_type=P,seed=seed,code_sha256=digest,
                               seconds=time.monotonic()-start,
                               process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                               (1 if sys.platform=="darwin" else 1024))
                    out.write_text(json.dumps(row,indent=2,allow_nan=False)+"\n")
                    print(prefix.name,row["status"],row.get("iterations"),row.get("dimensions"),
                          f'{row["seconds"]:.1f}s',flush=True)


if __name__ == "__main__":
    main()
