"""Predict cell means, variances, gains and three dimensions from held-out graph seeds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP.parent / "src"))
from eicavity.multitype import population_dimension
from eicavity.network import NetworkConfig, make_network
from eicavity.single_site import centered_moment, rate

SOURCE = json.loads((EXP/"degree/predictions.json").read_text())
G, J0, F = (np.array(SOURCE[k]) for k in ("G", "J0", "fractions"))
TARGET = .3
SEEDS = (15101,15102,15103)
CHI = (0,1,2)
ORDER = 48


def hermite_power(mu, q, terms=ORDER):
    """Hermite variance contributions to Cov[phi(x_t),phi(x_0)] in powers of rho(t)."""
    z = mu / np.sqrt(1+q)
    ratio = q/(1+q)
    weight = 2/np.pi*np.exp(-z*z)
    previous = np.zeros_like(z)
    current = np.ones_like(z)
    coefficients = np.empty((len(z),terms))
    for n in range(terms):
        weight = weight*ratio
        coefficients[:,n] = weight*current**2/(n+1)
        previous,current = current,(z*current-np.sqrt(n)*previous)/np.sqrt(n+1)
    return coefficients


def kernel_basis(chi):
    with np.load(EXP/'degree'/f'chi{chi}_one_site.npz') as site:
        t = site['time']
        k = site['input_kernels']
    rho = k/k[:,0,None]
    powers = np.arange(1,ORDER+1)
    # Green's function for (1-d²/dt²) with an even source gives K(0)=∫exp(-t)C(t)dt.
    fine_t = np.linspace(0,40,2001)
    shapes = CubicSpline(t,rho,axis=1)(fine_t)
    filtered = simpson(np.exp(-fine_t)*shapes[:,None,:]**powers[None,:,None], x=fine_t, axis=-1)
    return t,k[:,0],rho[:,None,:]**powers[None,:,None],filtered


def constrain_mean(rows, q):
    scale = np.sqrt(1+q)
    offset = brentq(lambda h: np.mean(rate((rows+h)/scale))-TARGET,
                    -float(rows.max())-14*scale.max(),-float(rows.min())+14*scale.max())
    return rows+offset


def solve(x, x2, labels, ref, q0, filtered, heterogeneous):
    q = q0[labels].copy()
    mu = np.asarray(ref['mean_inputs'])[labels]
    types = [slice(0,np.sum(labels==0)),slice(np.sum(labels==0),len(labels))]
    residual = float('inf')
    for iteration in range(1,1601):
        m = rate(mu/np.sqrt(1+q))
        raw = x@m
        variance = q0[labels].copy()
        if heterogeneous:
            c = np.empty_like(q)
            for a, sl in enumerate(types):
                c[sl] = hermite_power(mu[sl],q[sl]) @ filtered[a]
            flux = x2@c
            for a, sl in enumerate(types):
                variance[sl] *= flux[sl]/flux[sl].mean()
        update = np.empty_like(mu)
        for sl in types:
            update[sl] = constrain_mean(raw[sl],variance[sl])
        residual = float(max(abs(update-mu).max(),abs(variance-q).max()))
        if residual < 2e-6:
            break
        mu += .45*(update-mu)
        q += .45*(variance-q)
    return mu,q,{'iterations':iteration,'residual':residual,'converged':residual<2e-6}


class Site:
    """Present the two empirical, unequally sized classes to the equal-weight cavity."""
    fractions, variance_profile = F,G

    def __init__(self,mu,q,labels,time,basis):
        self.time = time
        self.mu,self.q,self.labels,self.basis = mu,q,labels,basis

    def conditional(self,n_hermite=None):
        nE = np.sum(self.labels==0)
        gains, covariance = [], []
        for a in range(2):
            mu,q = self.mu[self.labels==a],self.q[self.labels==a]
            c = hermite_power(mu,q) @ self.basis[a]
            alpha = np.sqrt(2/np.pi/(1+q))*np.exp(-mu**2/(2*(1+q)))
            # Replicating every I cell equally is exact for each empirical mean.
            assert nE%len(q) == 0
            covariance.append(np.repeat(c,nE//len(q),axis=0))
            gains.append(np.repeat(alpha,nE//len(q)))
        return {'alpha':np.array(gains),'covariance':np.array(covariance),'weights':np.full(nE,1/nE)}


def predict(chi,seed,out):
    ref = next(row for row in SOURCE['rows'] if row['chi']==chi)
    t,q0,basis,filtered = kernel_basis(chi)
    config = NetworkConfig(n=2000,seed=seed,ensemble='degree',degree_dispersion=chi,
                           fraction_e=float(F[0]),variance_profile=G.tolist(),mean_connectivity=J0.tolist())
    x,_,_,labels,_ = make_network(config,np.asarray(ref['mean_inputs']))
    graph_hash = hashlib.sha256(x.tobytes()).hexdigest()
    x -= J0[labels[:,None],labels[None,:]]/np.sqrt(len(labels))
    x2 = x*x
    results = {'chi':chi,'seed':seed,'graph_sha256':graph_hash,
               'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    cells = {'labels':labels.astype('int8')}
    for mode in ('shared','joint'):
        mu,q,row = solve(x,x2,labels,ref,q0,filtered,mode=='joint')
        cell_gain = np.sqrt(2/np.pi/(1+q))*np.exp(-mu**2/(2*(1+q)))
        cells.update({mode+'_mu':mu,mode+'_q':q,mode+'_gain':cell_gain})
        if row['converged']:
            pair = population_dimension(Site(mu,q,labels,t,basis),max_frequency=12.)
            means = rate(mu/np.sqrt(1+q))
            np.testing.assert_allclose([means[labels==a].mean() for a in range(2)],TARGET,atol=2e-7)
            np.testing.assert_allclose([q[labels==a].mean() for a in range(2)],q0,atol=2e-8)
            chosen = np.linspace(0,len(mu)-1,151,dtype=int)
            approx = hermite_power(mu[chosen],q[chosen]).sum(1)
            covariance_error = float(np.max(abs(approx-centered_moment(mu[chosen],q[chosen],q[chosen]))))
            row.update(covariance_quadrature_error=covariance_error,response_radius=pair['response_radius'],
                       min_spectral_mass=pair['min_spectral_mass'],gain_second=[])
            row['pr_admitted'] = bool(pair['response_radius']<1 and pair['min_spectral_mass']>-1e-6 and covariance_error<3e-5)
            if row['pr_admitted']:
                row['dimensions'] = [*pair['within_population_dimension_fractions'],pair['dimension_fraction']]
            for a in range(2):
                row['gain_second'].append(float(np.mean(cell_gain[labels==a]**2)))
        results[mode]=row
    dest = out/f'chi{chi}_seed{seed}'
    np.savez_compressed(dest.with_suffix('.npz'),**cells)
    dest.with_suffix('.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    return results


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chi',nargs='+',type=int,default=CHI,choices=CHI)
    parser.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    parser.add_argument('--out',type=Path,default=HERE/'predictions')
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    with threadpool_limits(1):
        for chi in args.chi:
            for seed in args.seeds:
                row = predict(chi,seed,args.out)
                print(chi,seed,row['shared'],row['joint'],flush=True)
