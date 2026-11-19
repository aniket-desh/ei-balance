"""Measure the participation ratio of each fixed 100-neuron principal submatrix."""

from itertools import combinations
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import ndtr

HERE=Path(__file__).resolve().parent


def readout_moments(means,covariances,network_size,block_duration):
    blocks,k=means.shape
    pairs=list(combinations(range(blocks),2))
    index={p:i for i,p in enumerate(pairs)}
    # A pair of blocks gives a covariance centered without knowing the stationary mean.
    matrices=np.stack([.5*(covariances[a]+covariances[b]+np.outer(means[a]-means[b],means[a]-means[b])) for a,b in pairs])
    diagonals=matrices.diagonal(axis1=1,axis2=2)
    traces=diagonals.sum(1)
    flat=matrices.reshape(len(pairs),-1)
    gram=flat@flat.T
    gram_diagonal=diagonals@diagonals.T
    result={}
    for stop in (blocks//2,blocks):
        for gap in (0,1):
            matchings=[]
            for a,b,c,d in combinations(range(stop),4):
                if min(b-a,c-b,d-c)>gap:
                    matchings.extend([(index[a,b],index[c,d]),(index[a,c],index[b,d]),(index[a,d],index[b,c])])
            if not matchings:continue
            left,right=np.array(matchings).T
            tr=float(traces[[i for i,(a,b) in enumerate(pairs) if b<stop and b-a>gap]].mean())
            energy=float(gram[left,right].mean()); diagonal=float(gram_diagonal[left,right].mean())
            pr=tr**2/energy if energy!=0 else None
            result[f'T{int(stop*block_duration)}_gap{gap}']={'trace':tr,'frobenius_energy':energy,'diagonal_energy':diagonal,
                'PR':pr,'normalized_PR':pr/k if pr is not None else None,'pair_energy':network_size*(energy-diagonal)/(k*(k-1))}
    return result


def measure_case(job,trajectory_path):
    name=job['name']; ids=np.asarray(job['subsets']); c=job['config']; n=c['n']; block=500
    traj=np.load(trajectory_path,mmap_mode='r'); stride=round(block/c['sample_dt']); nb=len(traj)//stride
    if len(traj)!=round(c['duration']/c['sample_dt']) or len(traj)%stride or ids.shape[:1]!=(2,):
        raise ValueError('Incomplete recording or malformed readout indices.')
    pop,draws,k=ids.shape;unique,inverse=np.unique(ids,return_inverse=True); loc=inverse.reshape(ids.shape)
    means=np.empty((pop,draws,nb,k)); cov=np.empty((pop,draws,nb,k,k))
    for b in range(nb):
        x=np.asarray(traj[b*stride:(b+1)*stride,unique],dtype=float); rates=2*ndtr(x)
        for a in range(pop):
            for d in range(draws):
                z=rates[:,loc[a,d]]; mu=z.mean(0); means[a,d,b]=mu
                cov[a,d,b]=z.T@z/stride-np.outer(mu,mu)
    del traj
    windows=[[readout_moments(means[a,d],cov[a,d],n,block) for d in range(draws)] for a in range(pop)]
    source=json.loads((HERE/'runs'/(name+'.json')).read_text()); destination=HERE/'measurements'
    small=destination/(name+'_blocks.npz'); ii,jj=np.tril_indices(k)
    record={'name':name,'config':c,'block_duration':block,'subset_indices':ids.tolist(),'readouts':windows,
            'connectivity_sha256':source['connectivity_sha256'],'native_library_sha256':source['native_library_sha256'],
            'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
            'measurement_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'block_statistics':small.name}
    with TemporaryDirectory(dir=destination,prefix='.readout-') as temp:
        tmp=Path(temp); np.savez_compressed(tmp/small.name,indices=ids,block_mean_rate=means,
            block_cov_lower=cov[...,ii,jj],block_duration=block,network_size=n)
        (tmp/small.name).replace(small)
        result=tmp/(name+'.json');result.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n'); result.replace(destination/result.name)
