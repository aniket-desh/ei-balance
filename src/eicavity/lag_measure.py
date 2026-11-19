"""Lag-resolved pair moments with separated-block covariance estimates."""
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len


def sampled_pairs(labels, count=12000, seed=20261024):
    rng=np.random.default_rng(seed)
    groups=[np.flatnonzero(labels==a) for a in range(2)]
    pairs=[]
    for a,b in [(0,0),(0,1),(1,1)]:
        ia,ib=groups[a],groups[b]
        if a==b:
            ii,jj=np.triu_indices(len(ia),1)
            selected=rng.choice(len(ii),min(count,len(ii)),replace=False)
            pairs.append(np.column_stack([ia[ii[selected]],ia[jj[selected]]]))
        else:
            selected=rng.choice(len(ia)*len(ib),min(count,len(ia)*len(ib)),replace=False)
            pairs.append(np.column_stack([ia[selected//len(ib)],ib[selected%len(ib)]]))
    return pairs


def block_pair_covariances(rates,pairs,lag_steps,nblocks=8,batch_size=96):
    """C_ij(tau)=mean r_i(t+tau) r_j(t), after centering each block."""
    output=[[] for _ in pairs]
    for block_index,rb in enumerate(np.array_split(rates,nblocks)):
        rb=rb-rb.mean(axis=0)
        length=len(rb)
        nfft=next_fast_len(2*length-1)
        transformed=rfft(rb,n=nfft,axis=0)
        indices=np.asarray(lag_steps)%nfft
        denominator=length-np.abs(lag_steps)
        if np.any(denominator<=0):raise ValueError('Lags exceed temporal block length')
        for k,ij in enumerate(pairs):
            cov=np.empty((len(ij),len(indices)))
            for lo in range(0,len(ij),batch_size):
                chosen=ij[lo:lo+batch_size]
                spectrum=transformed[:,chosen[:,0]]*transformed[:,chosen[:,1]].conj()
                cc=irfft(spectrum,n=nfft,axis=0)
                cov[lo:lo+len(chosen)]=(cc[indices]/denominator[:,None]).T
            output[k].append(cov)
    return [np.array(x) for x in output]


def moment_from_blocks(covariances,n,within_population=False,min_separation=2):
    pairings=[(b,c) for b in range(len(covariances))
              for c in range(b+min_separation,len(covariances))]
    if not pairings:raise ValueError('No temporal-block pairs remain')
    moment=np.zeros((covariances.shape[-1],)*2)
    for b,c in pairings:
        product=covariances[b].T@covariances[c]
        moment+=(product+product.T)/(2*len(pairings)*covariances.shape[1])
    moment*=n
    if within_population:
        # Each sampled unordered pair represents both directed orientations.
        moment=(moment+moment[::-1,::-1])/2
    return moment


def validate_fft_convention():
    rng=np.random.default_rng(719)
    r=rng.normal(size=(57,4));ij=np.array([[0,1],[2,3]])
    steps=np.array([-7,-2,0,3,9])
    got=block_pair_covariances(r,[ij],steps,nblocks=1)[0][0]
    z=r-r.mean(axis=0)
    wanted=[]
    for i,j in ij:
        row=[]
        for lag in steps:
            row.append(np.mean(z[lag:,i]*z[:len(z)-lag,j]) if lag>=0
                       else np.mean(z[:len(z)+lag,i]*z[-lag:,j]))
        wanted.append(row)
    np.testing.assert_allclose(got,wanted,rtol=1e-12,atol=1e-14)
    return float(np.max(np.abs(got-np.array(wanted))))
