"""Independent constraints for degree statistics, fraction weights, and lag sign."""
import unittest
import itertools
import numpy as np
from numpy.testing import assert_allclose
from eicavity.degree import degree_counts,solve_degree_scalar,critical_degree
from eicavity.single_site import solve_scalar
from eicavity.composition import critical_composition,dimension_at_fraction
from eicavity.network import NetworkConfig,make_network
from eicavity.lag_measure import validate_fft_convention
from eicavity.multitype import solve_populations
from eicavity.pair import block_covariance_energy


class ExtensionTests(unittest.TestCase):
    def test_exact_finite_pool_sampling_variance(self):
        # Exhaust every subset, rather than checking a simulator against itself.
        values=np.array([.1,.3,.7,1.4,1.8]);m=len(values)
        probabilities=np.array([.04,.14,.26,.30,.18,.08])
        k=np.arange(m+1);ek=probabilities@k;p=ek/m
        vk=probabilities@(k-ek)**2;chi=vk/(m*p*(1-p))
        first=second=0.
        for degree,pk in enumerate(probabilities):
            sums=np.array([values[list(s)].sum() for s in itertools.combinations(range(m),degree)])
            first+=pk*sums.mean();second+=pk*(sums*sums).mean()
        predicted=m*p*(1-p)*((m-chi)/(m-1)*values.var()+chi*values.mean()**2)
        assert_allclose(second-first*first,predicted,rtol=1e-13)

    def test_degree_ensemble_moments_and_dale_signs(self):
        rng=np.random.default_rng(79013)
        for chi in [0.,.5,1.,2.]:
            counts,effective=degree_counts(rng,120000,200,.17,chi)
            self.assertTrue(np.all((counts>=0)&(counts<=200)))
            self.assertLess(abs(counts.mean()-34),.08)
            if chi>0:assert_allclose(counts.var(),effective*200*.17*.83,rtol=.025)
            else:assert_allclose(counts,34)
            cfg=NetworkConfig(n=125,ensemble='degree',degree_dispersion=chi,seed=19)
            j,_,_,labels,_=make_network(cfg,np.array([-2.,-2.]))
            self.assertTrue(np.all(j[:,labels==0]>=0))
            self.assertTrue(np.all(j[:,labels==1]<=0))

    def test_degree_limits_and_susceptibility(self):
        old=solve_scalar(4.)
        new=solve_degree_scalar(4.,1.)
        assert_allclose([new.static_variance,new.dynamic_variance,new.response_radius],
                        [old.static_variance,old.dynamic_variance,old.response_radius],rtol=1e-10)
        zero=solve_degree_scalar(4.,0.)
        small=solve_degree_scalar(4.,1e-5)
        coefficient=16*.3**2/(1-zero.response_radius)
        assert_allclose(small.static_variance/1e-5,coefficient,rtol=1e-4)
        critical=critical_degree(1e-7)
        assert_allclose(critical['static_variance']/np.sqrt(1e-7),
                        critical['small_chi_static_variance_coefficient'],rtol=.002)

    def test_critical_fraction_minimum_and_identity_resolvent(self):
        result=critical_composition([.8,.1],[1,2],[.2,.3],[.08,.27],[.8,.2])
        opt=np.array(result['optimal_fractions'])
        c=np.array(result['coefficients'])
        for f in np.linspace(.05,.95,101):
            self.assertGreaterEqual(np.sum(c/[f,1-f]),result['minimum_heterogeneity_factor']-1e-13)
        assert_allclose(np.sum(c/opt),result['minimum_heterogeneity_factor'])
        # R=I: covariance energy is sum_a f_a E[v_a^2].
        tensor=np.zeros((2,2,2));tensor[0,0,0]=.7;tensor[1,1,1]=1.2
        result=dimension_at_fraction(tensor,[.4,.6],[.7,1.2],[.8,.2])
        assert_allclose(result['covariance_energy'],.8*.7+.2*1.2)
        assert_allclose(result['pair_covariance_energy'],0,atol=1e-14)

    def test_fft_lag_orientation(self):
        self.assertLess(validate_fft_convention(),1e-13)

    def test_directed_connection_response_phase(self):
        # With only 1->0 connectivity, x_0=H eta_0+J_01 H r_1.
        # The cross spectrum is J_01 H C_11, fixing its phase independently
        # of the cavity equations and of the Fourier measurement code.
        frequencies=np.array([.3,.8]);h=1/(1+1j*frequencies)
        u=np.array([h.prod(),.2*h.prod()]);q=np.array([.4,.7]);f=np.array([.8,.2])
        g=np.array([[0.,.6],[0.,0.]])
        _,psi=block_covariance_energy(g,f,u,q)
        assert_allclose(psi[0,1],.6*u[0]*q[1],atol=1e-14)

    def test_uniform_population_fraction_null(self):
        # Identical local laws and fixed uniform edge variance: fractions
        # redistribute sampling but leave the whole-network energy invariant.
        u=.13-.08j;q=.7;g2=2.
        for fe in [.1,.5,.8,.95]:
            f=np.array([fe,1-fe])
            full,_=block_covariance_energy(np.full((2,2),g2),f,[u,u],[q,q])
            assert_allclose(f@full@f,q/abs(1-g2*u)**2,atol=1e-13)

    def test_population_degree_solver_recovers_scalar_branch(self):
        for chi in [0.,2.]:
            scalar=solve_degree_scalar(4.,chi)
            population=solve_populations(np.full((2,2),16.),degree_dispersion=chi,continuation_steps=1)
            assert_allclose(population.static_variances,scalar.static_variance,atol=1e-9,rtol=1e-7)
            assert_allclose(population.dynamic_variances,scalar.dynamic_variance,rtol=1e-7)
            self.assertLess(population.boundary_residual,1e-9)


if __name__=='__main__':unittest.main()
