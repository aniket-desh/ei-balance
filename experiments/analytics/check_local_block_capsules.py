"""Independent checks of compact captured moments and four-window algebra."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,itertools
from pathlib import Path
import numpy as np
from scipy.special import ndtr
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
CAPS=HERE/'local_block_capsules';RUNTIME=ROOT/'next_round/runtime'

def distinct_four(x):
    nb=len(x);s1=x.sum(0);s2=(x*x).sum(0);s3=(x**3).sum(0);s4=(x**4).sum(0)
    return (s1**4-6*s1*s1*s2+3*s2*s2+8*s1*s3-6*s4)/(nb*(nb-1)*(nb-2)*(nb-3))

def diagonal_source(mean_rate,second_rate):
    m,q=mean_rate,second_rate;nb=len(m)
    sm=m.sum(0);sm2=(m*m).sum(0);sq=q.sum(0)
    termqq=(sq*sq-(q*q).sum(0))/(nb*(nb-1))
    termqmm=(sq*(sm*sm-sm2)-2*sm*(q*m).sum(0)+2*(q*m*m).sum(0))/(nb*(nb-1)*(nb-2))
    return termqq-2*termqmm+distinct_four(m)

def validate_identity():
    rng=np.random.default_rng(901091)
    m=rng.uniform(.1,1,(7,11));q=m*m+rng.uniform(.2,.8,(7,11))
    direct=[]
    for a,b,c,d in itertools.combinations(range(len(m)),4):
        for (i,j),(k,l) in [((a,b),(c,d)),((a,c),(b,d)),((a,d),(b,c))]:
            v1=.5*(q[i]+q[j])-m[i]*m[j]
            v2=.5*(q[k]+q[l])-m[k]*m[l]
            direct.append(v1*v2)
    error=float(abs(diagonal_source(m,q)-np.mean(direct,axis=0)).max())
    assert error<2e-14
    return error

def main():
    validations=[];max_direct=0.;all_shapes=True
    for path in sorted(CAPS.glob('*.npz')):
        name=path.stem
        with np.load(path) as z:
            arrays={k:z[k] for k in ['mean_x','second_x','mean_rate','second_rate','mean_gain','second_gain']}
            nb,N=arrays['mean_x'].shape
            verified=float(z['verified_duration_at_capture'])
            assert nb<=int(verified//float(z['block_duration']))
            assert all(v.shape==(nb,N) and np.isfinite(v).all() for v in arrays.values())
        record=json.loads((RUNTIME/'runs'/f'{name}.json').read_text());config=record['job']['config']
        raw=RUNTIME/'tmp'/f'{name}_trajectory.npy';direct_error=None
        if raw.exists():
            traj=np.load(raw,mmap_mode='r');ns=round(500/config['sample_dt'])
            errors=[]
            # Two independently reduced blocks verify partial-prefix capture too.
            for b in sorted(set([0,nb-1])):
                x=np.array(traj[b*ns:(b+1)*ns],float);r=2*ndtr(x);g=np.sqrt(2/np.pi)*np.exp(-.5*x*x)
                wanted={'mean_x':x.mean(0),'second_x':(x*x).mean(0),'mean_rate':r.mean(0),'second_rate':(r*r).mean(0),'mean_gain':g.mean(0),'second_gain':(g*g).mean(0)}
                errors.extend(float(abs(arrays[k][b]-v).max()) for k,v in wanted.items())
                del x,r,g
            direct_error=max(errors);max_direct=max(max_direct,direct_error)
            del traj
            assert direct_error<1e-11
        source_error=None
        degree=ROOT/'next_round/degree/ingredient_diagnostics'/f'{name}.json'
        if nb==16 and degree.exists():
            d=json.loads(degree.read_text());row=next(v for v in d['rows'] if v['empty_blocks']==0)
            V2=diagonal_source(arrays['mean_rate'],arrays['second_rate']);ne=round(N*config['fraction_e'])
            predicted=np.array([V2[:ne].mean(),V2[ne:].mean()])
            source_error=float(abs(predicted-np.array(row['v_squared_fourwindow_by_population'])).max())
            assert source_error<1e-12
        validations.append({'name':name,'blocks':nb,'verified_duration':verified,
                            'direct_raw_max_abs':direct_error,'independent_fourwindow_source_max_abs':source_error})
    result={'fourwindow_source_identity_max_abs':validate_identity(),'capsules_checked':len(validations),
             'raw_checked':sum(v['direct_raw_max_abs'] is not None for v in validations),
             'maximum_direct_raw_error':max_direct,'verified_prefix_and_shape_checks_pass':all_shapes,
             'records':validations,
             'scope':'Capsules are compact summaries of already authorized v1 trajectories, not new simulations. Missing trajectories are explicit in coverage.json.'}
    (CAPS/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
