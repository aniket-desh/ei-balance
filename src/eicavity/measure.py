"""Time-centered covariance estimators and explicit sampling diagnostics."""
import numpy as np
from scipy.signal import fftconvolve
from .single_site import rate,rate_prime


def summarize(trajectory,labels,sample_dt,nblocks=8):
    x=np.asarray(trajectory,dtype=float)
    r=rate(x)
    n=x.shape[1]
    means=r.mean(axis=0)
    xc=x-x.mean(axis=0)
    rc=r-means
    cov=rc.T@rc/len(rc)
    diag=np.diag(cov)
    energy=np.sum(cov*cov)/n
    blocks=[]
    covariances=[]
    for rb,xb in zip(np.array_split(r,nblocks),np.array_split(x,nblocks)):
        rb=rb-rb.mean(axis=0)
        cb=rb.T@rb/len(rb)
        covariances.append(cb)
        blocks.append({'mean_variance':float(np.trace(cb)/n),'energy':float(np.sum(cb*cb)/n),
                       'dimension_fraction':float((np.trace(cb)/n)**2/(np.sum(cb*cb)/n)),
                       'input_variance':float(np.var(xb,axis=0).mean())})
    # Cross products of estimates from separated time blocks remove the
    # leading positive covariance-estimation bias if blocks are independent.
    # Keep the full matrix to expose disagreement between block choices.
    cross=np.array([[np.sum(c1*c2)/n for c2 in covariances] for c1 in covariances])
    cross_values=np.array([cross[i,j] for i in range(nblocks) for j in range(i+2,nblocks)])
    cross_energy=float(cross_values.mean())
    maxlag=min(round(40/sample_dt),len(x)//4)
    pad=2**int(np.ceil(np.log2(2*len(x))))
    k=np.empty((maxlag+1,n))
    c=np.empty_like(k)
    denominator=(len(x)-np.arange(maxlag+1))[:,None]
    for start in range(0,n,64):
        stop=min(n,start+64)
        fx=np.fft.rfft(xc[:,start:stop],n=pad,axis=0)
        fr=np.fft.rfft(rc[:,start:stop],n=pad,axis=0)
        k[:,start:stop]=np.fft.irfft(np.abs(fx)**2,n=pad,axis=0)[:maxlag+1]/denominator
        c[:,start:stop]=np.fft.irfft(np.abs(fr)**2,n=pad,axis=0)[:maxlag+1]/denominator
    result={'mean_rate':float(means.mean()),'mean_variance':float(diag.mean()),
            'variance_second':float(np.mean(diag**2)),'covariance_energy':float(energy),
            'offdiagonal_energy':float(energy-np.mean(diag**2)),
            'dimension_fraction':float(diag.mean()**2/energy),
            'cross_energy':cross_energy,'cross_dimension_fraction':float(diag.mean()**2/cross_energy),
            'cross_energy_std':float(cross_values.std(ddof=1)),
            'input_dynamic_variance_mean':float(np.mean(np.var(x,axis=0))),
            'input_dynamic_variance_cv':float(np.std(np.var(x,axis=0))/np.mean(np.var(x,axis=0))),
            'input_static_variance':float(np.var(x.mean(axis=0))),
            'input_mean':float(x.mean()),'alpha_mean':float(rate_prime(x).mean()),
            'alpha_second':float(np.mean(rate_prime(x).mean(axis=0)**2)),
            'mean_by_population':[float(means[labels==a].mean()) for a in [0,1]],
            'blocks':blocks}
    arrays={'covariance':cov,'input_means':x.mean(axis=0),'rate_means':means,
            'input_variances':np.var(x,axis=0),'rate_variances':diag,
            'alpha':rate_prime(x).mean(axis=0),'cross_energy_matrix':cross,
            'lag':np.arange(maxlag+1)*sample_dt,'input_autocovariance':k.mean(axis=1),
            'rate_autocovariance':c.mean(axis=1),'rate_autocovariance_each':c.T}
    return result,arrays
