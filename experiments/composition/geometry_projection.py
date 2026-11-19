"""Exact covariance-energy decompositions used only as diagnostics.

All matrices are covariances of rate fluctuations. The cross estimator is a
bilinear form of two independently estimated covariance matrices; its sectors
can be negative on finite records even though the stationary self-energy
sectors are nonnegative. Population energies are ordered, so EI and IE are
both retained and sum to twice the rectangular cross contribution.
"""
import numpy as np


def population_basis(labels):
    labels=np.asarray(labels)
    counts=np.bincount(labels)
    return np.eye(len(counts))[labels]/np.sqrt(counts)[None,:]


def cross_geometry(C,D,labels):
    C,D=np.asarray(C),np.asarray(D)
    N=len(labels)
    U=population_basis(labels)
    ZC,ZD=C@U,D@U
    AC,AD=U.T@ZC,U.T@ZD
    total=float(np.sum(C*D)/N)
    corner=float(np.sum(AC*AD)/N)
    mixed=float((np.sum(ZC*ZD)-np.sum(AC*AD))/N)
    perp=total-corner-2*mixed
    block=np.array([[np.sum(C[np.ix_(labels==a,labels==b)]*D[np.ix_(labels==a,labels==b)])/N
                     for b in range(U.shape[1])]for a in range(U.shape[1])])
    return dict(total=total,ordered_population_energies=block,
                population_corner=corner,population_mixed_one_orientation=mixed,
                perpendicular=perp,population_touching=corner+2*mixed)


def check():
    rng=np.random.default_rng(276)
    n=11
    labels=np.r_[np.zeros(7,int),np.ones(4,int)]
    X,Y=rng.normal(size=(n,20)),rng.normal(size=(n,21))
    C,D=X@X.T,Y@Y.T
    out=cross_geometry(C,D,labels)
    U=population_basis(labels);P=U@U.T;Q=np.eye(n)-P
    expected=[np.trace((P@C@P)@(P@D@P))/n,
              np.trace((P@C@Q)@(Q@D@P))/n,
              np.trace((Q@C@Q)@(Q@D@Q))/n]
    np.testing.assert_allclose(expected,[out['population_corner'],out['population_mixed_one_orientation'],out['perpendicular']],rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(out['ordered_population_energies'].sum(),out['total'],rtol=1e-13)
    np.testing.assert_allclose(sum([expected[0],2*expected[1],expected[2]]),out['total'],rtol=1e-13)
    return dict(pass_exact_projection=True,total=out['total'],corner=expected[0],mixed_twice=2*expected[1],perpendicular=expected[2])

if __name__=='__main__':
    import json
    print(json.dumps(check(),indent=2))
