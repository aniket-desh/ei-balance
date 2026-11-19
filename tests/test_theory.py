"""Independent checks of normalizations, limiting cases, and closure algebra."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import simpson
from eicavity.single_site import normal_quadrature,rate,rate_prime,raw_moment,raw_derivative,solve_scalar
from eicavity.pair import even_spectrum,block_covariance_energy,conditional_dimension
from eicavity.network import NetworkConfig,make_network
from eicavity.critical import critical_prefactor,solution_at_variance


class TheoryTests(unittest.TestCase):
    def test_critical_coefficient_and_coupling_transversality(self):
        p=critical_prefactor();d=solution_at_variance(1e-4)
        assert_allclose(p['universal_constant'],4*np.pi**2/3-12,rtol=1e-12)
        eps=d.g/p['critical_g']-1
        assert_allclose(d.dynamic_variance/eps,3.,rtol=3e-5)
        assert_allclose((1-d.response_radius)/eps,p['critical_g']**2*p['curvature_second'],rtol=1e-4)
        assert_allclose((d.static_variance-p['critical_static_variance'])/d.dynamic_variance,
                        (2*p['critical_static_variance']-1)/3,rtol=5e-6)

    def test_gaussian_quadrature_and_price_formula(self):
        z,w=normal_quadrature(100)
        assert_allclose([w.sum(),w@z,w@(z*z)],[1,0,1],atol=1e-13)
        for mu,v,c in [(-1.3,2.,.8),(.7,.9,-.3),(0,3,2.5)]:
            x=mu+np.sqrt(v)*z[:,None]
            y=mu+c/np.sqrt(v)*z[:,None]+np.sqrt(v-c*c/v)*z[None,:]
            exact=np.sum(w[:,None]*w[None,:]*rate(x)*rate(y))
            assert_allclose(raw_moment(mu,v,c),exact,rtol=1e-9,atol=1e-11)
            dc=1e-5
            deriv=(raw_moment(mu,v,c+dc)-raw_moment(mu,v,c-dc))/(2*dc)
            assert_allclose(raw_derivative(mu,v,c),deriv,rtol=1e-8)

    def test_two_frequency_resolvent_against_cavity_equations(self):
        rng=np.random.default_rng(8301)
        for size in [1,2,3,5]:
            for repeat in range(8):
                f=rng.dirichlet(np.ones(size));F=np.diag(f)
                G=rng.uniform(.05,.4,(size,size))
                u=(rng.uniform(.1,.8,size)+1j*rng.uniform(-.3,.3,size))
                U=np.diag(u);q=rng.uniform(.1,2,size);Q=np.diag(q)
                full,psi=block_covariance_energy(G,f,u,q)
                gammaf=np.linalg.solve(np.eye(size)-G@F@U.conj(),G)
                gammaeta=G@F@Q@G.T+G@F@psi@F@G.T
                rhs=U@gammaeta@U.conj()+U@gammaf.conj()@Q+Q@gammaf.T@U.conj()
                assert_allclose(psi,rhs,atol=1e-12,rtol=1e-11)
                assert_allclose(gammaf,G+G@F@U.conj()@G+G@F@U.conj()@gammaf@U.conj()@F@G,atol=1e-12)
                if size==1:
                    assert_allclose(full[0,0],q[0]/abs(1-G[0,0]*u[0])**2)

    def test_dale_and_population_stability(self):
        cfg=NetworkConfig(n=500,seed=44)
        j,_,_,labels,info=make_network(cfg,np.array([-2.,-2.]))
        self.assertTrue(np.all(j[:,labels==0]>=0))
        self.assertTrue(np.all(j[:,labels==1]<=0))
        assert_allclose(info['probabilities']*(1-info['probabilities'])*info['nonzero_weights_scaled']**2,cfg.g**2)
        self.assertTrue(np.all(np.linalg.eigvals(info['j0']@np.diag(info['fractions'])).real<0))

    def test_dmft_self_consistency_and_spectral_normalization(self):
        dmft=solve_scalar(4.,(.3,.3),(.8,.2))
        res=conditional_dimension(dmft,dt=.08,n_hermite=96)
        w=res['weights']; c=res['covariance'];k=res['kernel'];t=res['time']
        avg=np.einsum('a,z,azt->t',dmft.fractions,w,c)
        assert_allclose(avg,dmft.mean_covariance(k),atol=2e-11)
        assert_allclose(np.sum(dmft.fractions[:,None]*w*res['means']**2)*16,dmft.static_variance,rtol=1e-10)
        kpp=(-k[4:]+16*k[3:-1]-30*k[2:-2]+16*k[1:-3]-k[:-4])/(12*(t[1]-t[0])**2)
        assert_allclose(kpp,k[2:-2]-16*avg[2:-2],atol=2e-6)
        assert_allclose(res['response_radius'],dmft.response_radius,rtol=1e-10)
        self.assertLess(res['kernel_tail_fraction'],1e-10)
        self.assertGreater(res['min_spectral_mass'],-1e-10)
        omega,mass=even_spectrum(c,.08)
        assert_allclose(mass.sum(axis=-1),c[...,0],rtol=1e-12,atol=1e-14)
        self.assertLess(res['discarded_spectral_mass_fraction'],1e-8)
        # Input covariance obeys the leak filter in frequency space.
        omega,km=even_spectrum(k,.08)
        _,cm=even_spectrum(avg,.08)
        # Multiplication by omega^2 magnifies tiny FFT tail roundoff. Check
        # the retained spectral band tightly and the full integrated error.
        band=np.abs(omega)<12
        assert_allclose(((1+omega**2)*km)[band],(16*cm)[band],atol=1e-10)
        self.assertLess(np.sum(np.abs((1+omega**2)*km-16*cm))/np.sum(np.abs(16*cm)),1e-6)

if __name__=='__main__':
    unittest.main()
