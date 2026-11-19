"""Parameter-only conditional local laws and averaging controls.

Spectra loaded from the independently archived one-site calculation are
spectral masses. The frequency integration is a sum, with no extra factors.
All controls hold the original one-site covariance and PR numerator fixed.
"""
from pathlib import Path
import argparse
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import numpy as np
from scipy.special import roots_hermitenorm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from eicavity.single_site import centered_moment
sys.path.insert(0, str(ROOT/'experiments/figures'))
from paper_style import COMPOSITION, E, I, apply_style, panel_label, save_figure


def controls():
    cache=np.load(ROOT/'archive/recovered/bundle/diagnosis/theory/one_site.npz')
    summary=json.loads((ROOT/'archive/recovered/bundle/diagnosis/theory/summary.json').read_text())
    keep=np.abs(cache['omega'])<=12
    omega=cache['omega'][keep]
    spectra=cache['spectra'][...,keep]
    weights=cache['weights']
    second=np.sum(weights[None,:]*cache['alpha']**2,axis=1)
    first=np.sum(weights[None,:]*cache['alpha'],axis=1)
    full_q=np.array([(p.T*weights)@p for p in spectra])
    avg=np.sum(spectra*weights[None,:,None],axis=1)
    avg_q=avg[:,:,None]*avg[:,None,:]
    W=np.array([[12.8,1.8],[20.,2.4]])
    H=1/(1+1j*omega)
    results={}
    for fe in [.6,.8]:
        f=np.array([fe,1-fe])
        nu=float(np.dot(f,cache['conditional_variances']@weights))
        record={}
        for name,Q,A in [('conditional',full_q,second),
                         ('averaged_source',avg_q,second),
                         ('averaged_gain',full_q,first**2),
                         ('both_averaged',avg_q,first**2)]:
            integrated=np.zeros((2,2),complex)
            for lo in range(0,len(omega),32):
                hi=min(lo+32,len(omega))
                product=H[lo:hi,None]*H[None,:]
                R=np.linalg.inv(np.eye(2)-product[...,None,None]*A[None,None,:,None]*W)
                D=np.moveaxis(Q[:,lo:hi,:]/f[:,None,None],0,-1)
                X=np.einsum('...ac,...c,...bc->...ab',R,D,R.conj(),optimize=True)
                integrated+=np.sum(X,axis=(0,1))
            energy=float((f@integrated@f).real)
            record[name]={'nu':nu,'energy':energy,'d':nu**2/energy,
                          'temporal_Gram_alignment':float(integrated[0,1].real/np.sqrt(integrated[0,0].real*integrated[1,1].real)),
                          'integrated_X':integrated.real.tolist(),
                          'imaginary_residual':float(abs(integrated.imag).max())}
        reference=summary['predictions'][str(fe)]['dimension_fraction']
        assert abs(record['conditional']['d']/reference-1)<1e-7
        baseline=np.array(record['conditional']['integrated_X'])
        for name in ['averaged_source','averaged_gain','both_averaged']:
            assert np.min(baseline-np.array(record[name]['integrated_X']))>-1e-9
        results[str(fe)]=record
    results['scope']='Parameter-only conditional calculation. Averaging controls retain original local-law statistics and numerator; they are not self-consistent new network ensembles.'
    return cache,summary,results


def main(use_cache=False):
    out=Path(__file__).resolve().parent/'figures'
    out.mkdir(parents=True,exist_ok=True)
    values=out/'conditional_averaging_values.json'
    if use_cache:
        summary=json.loads((ROOT/'archive/recovered/bundle/diagnosis/theory/summary.json').read_text())
        result=json.loads(values.read_text())
    else:
        _,summary,result=controls()
        values.write_text(json.dumps(result,indent=2)+'\n')
    apply_style()
    fig,axs=plt.subplots(2,2,figsize=(7.4,5.15),constrained_layout=True)
    mu=np.linspace(-11,6,550)
    for a,label in enumerate(['E','I']):
        Delta=summary['static_variances'][a]
        q=summary['dynamic_variances'][a]
        h=summary['mean_inputs'][a]
        density=np.exp(-(mu-h)**2/(2*Delta))/np.sqrt(2*np.pi*Delta)
        variance=centered_moment(mu,q,q)
        alpha=np.sqrt(2/(np.pi*(1+q)))*np.exp(-mu**2/(2*(1+q)))
        line = dict(color=(E, I)[a], linestyle=('-', '--')[a])
        axs[0,0].plot(mu,density,label=label,**line)
        axs[0,1].plot(mu,variance,**line)
        axs[1,0].plot(mu,alpha,**line)
    for ax in axs.flat[:3]:
        ax.set_xlim(-10,5)
        ax.set_xticks([-10,-5,0,5])
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'Operating point $\mu$')
    axs[0,0].set_ylabel('Probability density')
    axs[0,0].legend(loc='upper left')
    axs[0,1].set_ylabel(r'Conditional variance $v(\mu)$')
    axs[1,0].set_ylabel(r'Mean slope $\alpha(\mu)$')
    names=['conditional','averaged_source','averaged_gain','both_averaged']
    short=['Full\nlaw','Average\nsource','Average\ngain','Average\nboth']
    x=np.arange(4)
    for offset,fe,hatch in [(-.18,.6,''),(.18,.8,'///')]:
        vals=[result[str(fe)][name]['d'] for name in names]
        axs[1,1].bar(x+offset,vals,width=.34,color=COMPOSITION[fe],
                     edgecolor='white',linewidth=.5,hatch=hatch,
                     label=f'{round(100*fe)}:{round(100*(1-fe))}')
    axs[1,1].set_xticks(x,short)
    axs[1,1].set_ylabel(r'Predicted dimension $D/N$')
    axs[1,1].legend(title='E:I',loc='upper left',labelspacing=.25)
    for ax,letter in zip(axs.flat,'ABCD'):
        panel_label(ax,letter)
    save_figure(fig,out/'conditional_local_laws')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-cache',action='store_true',help='Redraw with the saved averaging controls.')
    main(use_cache=parser.parse_args().from_cache)
