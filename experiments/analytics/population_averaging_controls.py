"""Parameter-only averaging controls, independent chunked two-population quadrature."""
import os
for _var in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[_var]='1'
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'recovered/bundle/diagnosis/theory/one_site.npz'
OUT=Path(__file__).resolve().parent/'population_averaging_controls.json'

def calculate(f, gain, averaged_spectra=False):
    z=np.load(CACHE)
    omega=z['omega']; keep=abs(omega)<=12
    omega=omega[keep]; spectra=z['spectra'][...,keep]
    weights=z['weights']; variance=z['conditional_variances']
    mean=variance@weights
    H=1/(1+1j*omega)
    W=np.array([[12.8,1.8],[20.,2.4]])
    B=gain[:,None]*W
    averaged=np.einsum('azw,z->aw',spectra,weights)
    full=np.zeros((2,2),complex)
    for lo in range(0,len(omega),32):
        hi=min(lo+32,len(omega))
        t=H[lo:hi,None]*H[None,:]
        R=np.linalg.inv(np.eye(2)-t[...,None,None]*B)
        if averaged_spectra:
            Q=averaged[:,lo:hi,None]*averaged[:,None,:]
        else:
            Q=np.einsum('azj,z,azk->ajk',spectra[:,:,lo:hi],weights,spectra,optimize=True)
        D=np.moveaxis(Q/f[:,None,None],0,-1)
        full+=np.einsum('...ac,...c,...bc->ab',R,D,R.conj(),optimize=True)
    full=full.real
    block=f[:,None]*f[None,:]*full
    E=float(block.sum());nu=float(f@mean)
    return {'nu':nu,'energy':E,'dimension_fraction':nu*nu/E,
            'integrated_X':full.tolist(),'block_energies':block.tolist(),
            'within_population_dimension_fractions':(mean*mean/(f*np.diag(full))).tolist(),
            'cross_population_energy_share':float(2*block[0,1]/E),
            'response_radius':float(max(abs(np.linalg.eigvals(B))))}

def main():
    z=np.load(CACHE);weights=z['weights'];alpha=z['alpha']
    A=alpha*alpha@weights;Abar=(alpha@weights)**2
    results={}
    for fe in [.6,.8]:
        f=np.array([fe,1-fe]);row={}
        for name,gain,avg in [('conditional',A,False),('averaged_spectra',A,True),
                              ('averaged_gain',Abar,False),('both_averaged',Abar,True)]:
            row[name]=calculate(f,gain,avg)
        base=np.array(row['conditional']['block_energies'])
        assert all(np.all(base>=np.array(row[mode]['block_energies'])-1e-12)
                   for mode in ['averaged_spectra','averaged_gain','both_averaged'])
        results[str(fe)]=row
    out={'source_cache':str(CACHE.relative_to(ROOT)),
         'spectral_convention':'Masses include domega/(2pi); symmetric abs(omega)<=12 cutoff.',
         'control':'Same conditional one-site law and fixed W=GF; replacements are approximations, not new network interventions.',
         'alpha_second':A.tolist(),'square_mean_alpha':Abar.tolist(),
         'results':results,'blockwise_averaging_inequalities_pass':True}
    OUT.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
