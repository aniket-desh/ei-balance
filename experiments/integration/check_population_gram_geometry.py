"""Independent time-series checks of population energy geometry identities."""
from pathlib import Path
import json
import numpy as np


def evaluate(xe, xi):
    T=xe.shape[1]
    cee=xe@xe.T/T;cii=xi@xi.T/T;cei=xe@xi.T/T
    ke=xe.T@xe/T;ki=xi.T@xi/T
    te=np.trace(cee);ti=np.trace(cii)
    ee=np.sum(cee*cee);ii=np.sum(cii*cii);ei=np.sum(cei*cei)
    overlap=ei/np.sqrt(ee*ii)
    de=te*te/ee;di=ti*ti/ii
    w=np.array([te,ti])/(te+ti)
    c=np.block([[cee,cei],[cei.T,cii]])
    direct=np.trace(c)**2/np.trace(c@c)
    reduced=1/(w[0]**2/de+w[1]**2/di+2*overlap*w[0]*w[1]/np.sqrt(de*di))
    errors=[abs(ei-np.trace(ke@ki)),abs(ee-np.trace(ke@ke)),
            abs(ii-np.trace(ki@ki)),abs(direct-reduced)]
    return {'overlap':float(overlap),'D_E':float(de),'D_I':float(di),
            'D_direct':float(direct),'D_from_population_geometry':float(reduced),
            'maximum_identity_error':float(max(errors))}


def main():
    T=12
    basis=np.eye(T)*np.sqrt(T)
    # All chosen basis vectors have zero temporal mean after an orthogonal
    # rotation away from the constant direction.
    q,_=np.linalg.qr(np.column_stack([np.ones(T),np.eye(T)[:,:T-1]]))
    modes=q[:,1:].T*np.sqrt(T)
    shared=evaluate(modes[:3],modes[:3])
    orthogonal=evaluate(modes[:3],modes[3:6])
    rng=np.random.default_rng(810221)
    xe=rng.normal(size=(5,40));xi=rng.normal(size=(3,40))
    xe-=xe.mean(1,keepdims=True);xi-=xi.mean(1,keepdims=True)
    general=evaluate(xe,xi);rescaled=evaluate(3*xe,.2*xi)
    result={'shared_modes':shared,'orthogonal_modes':orthogonal,
            'general':general,'rescaled':rescaled,
            'population_rescaling_overlap_error':abs(general['overlap']-rescaled['overlap'])}
    assert abs(shared['D_direct']-3)<1e-12
    assert abs(orthogonal['D_direct']-6)<1e-12
    assert max(x['maximum_identity_error'] for x in [shared,orthogonal,general,rescaled])<1e-10
    assert result['population_rescaling_overlap_error']<1e-14
    out=Path(__file__).with_suffix('.json');out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
