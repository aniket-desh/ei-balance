"""Parameter-only nonuniform EI degree predictions, without network fitting."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys, json, time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'recovered/bundle/ei_full/ei_balance_handoff/src'))
from eicavity.multitype import solve_populations,population_dimension
from eicavity.degree import degree_counts

OUT=Path(__file__).resolve().parent
G=np.array([[16.,9.],[25.,12.]])
J0=np.array([[1.,-6.],[2.,-10.]])
F=np.array([.8,.2])


def predict(chi,dt=.08,nh=96,horizon=100.,cutoff=12.):
    one=solve_populations(G,fractions=F,mean_rates=(.3,.3),dt=dt,
                         horizon=horizon,degree_dispersion=chi)
    pair=population_dimension(one,n_hermite=nh,max_frequency=cutoff)
    means=np.array(pair['mean_variances'])
    seconds=np.array(pair['variance_seconds'])
    per_pair=np.array(pair['pair_covariance_energy'])
    block=np.outer(F,F)*per_pair+np.diag(F*seconds)
    result={'chi':chi,'static_variances':one.static_variances.tolist(),
            'dynamic_variances':one.dynamic_variances.tolist(),
            'mean_inputs':one.mean_inputs.tolist(),
            'alpha_second':(pair['alpha']**2@pair['weights']).tolist(),
            'alpha_fourth':(pair['alpha']**4@pair['weights']).tolist(),
            'bvp_residual':one.bvp_residual,'static_residual':one.static_residual,
            'boundary_residual':one.boundary_residual,
            'nu':float(F@means),'block_energies':block.tolist(),
            'block_energy_sum_check':float(block.sum()-pair['covariance_energy']),
            'prediction':{k:v for k,v in pair.items() if not isinstance(v,np.ndarray)},
            'quadrature':{'dt':dt,'horizon':horizon,'n_hermite':nh,'frequency_cutoff':cutoff}}
    diagonal=float(F@seconds)
    result['decomposition']={'diagonal_energy':diagonal,
                             'uncorrelated_dimension_fraction':result['nu']**2/diagonal,
                             'correlation_reduction_factor':diagonal/pair['covariance_energy']}
    arrays={'time':one.time,'input_kernels':one.kernels,
            **{k:v for k,v in pair.items() if isinstance(v,np.ndarray)}}
    return result,arrays


def chi_effective(n,chi):
    probs=J0*J0/(J0*J0+G)
    counts=np.round(n*F).astype(int)
    eff=np.zeros((2,2))
    rng=np.random.default_rng(0)
    for a in range(2):
        for b in range(2):
            _,eff[a,b]=degree_counts(rng,1,int(counts[b]),float(probs[a,b]),chi)
    return eff.tolist()


def main():
    rows=[]
    for chi in [0.,.5,1.,2.]:
        tic=time.monotonic()
        row,arrays=predict(chi)
        row['expected_finite_count_dispersion']={str(n):chi_effective(n,chi) for n in [1000,2000,4000]}
        row['elapsed_seconds']=time.monotonic()-tic
        np.savez_compressed(OUT/f'chi{chi:g}_one_site.npz',**arrays)
        rows.append(row)
        record={'control':'Fixed G,J0,f and leading target means; change degree dispersion and set the O(1) external offset to the parameter-only h(chi). The leading sqrt(N) balance drive is fixed; the full finite-N drive is not fixed.',
                'G':G.tolist(),'J0':J0.tolist(),'fractions':F.tolist(),'mean_rates':[.3,.3],
                'rows':rows}
        (OUT/'predictions.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({'chi':chi,'h':row['mean_inputs'],'delta':row['static_variances'],
                          'q':row['dynamic_variances'],'d':row['prediction']['dimension_fraction'],
                          'elapsed_seconds':row['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
