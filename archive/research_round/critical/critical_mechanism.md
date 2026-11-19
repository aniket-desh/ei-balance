# Critical mechanism: curvature, bias and a disorder-controlled crossover

This calculation audits the strongest proposed physical claim. It derives a
new uniform small-degree-dispersion crossover within the existing conditional
DMFT and checks it against exact one-site continuation and resonant pair
integration. It does **not** measure a finite-network critical exponent.

## What changes in the paper's interpretation

The exponent 3/2 is not unique evidence for endogenous heterogeneity. A
homogeneous population biased away from the transfer function's inflection
point has nonzero local curvature and the same generic 3/2 law. At the
project's mean rate0.3, even the chi=0 fixed-degree homogeneous law is in that
class. The clean heterogeneity control is instead mean rate1 for phi=2Phi:
zero degree dispersion has an odd centered transfer and the cubic law,
whereas any fixed positive dispersion generates static disorder and an inner
3/2 law. The inner regime shrinks as dispersion tends to zero.

The distinctive candidate insight is therefore a **curvature-controlled
crossover produced self-consistently by degree disorder**, rather than the
unqualified statement that heterogeneity changes the exponent.

## 1. The two exponents are not a mere notation mismatch

Primary reference checked: Clark, Abbott and Litwin-Kumar, *Dimension of
Activity in Random Neural Networks*, PRL131,118401(2023),
https://arxiv.org/html/2207.12373v3 . Its main text gives intensive PR
proportional to (g-1)^3 for tanh near its zero-mean transition and collective
timescale proportional to (g-1)^-2. Our source already recovers that result in
its homogeneous odd limit. Both calculations have the same gap-based pair
geometry, but different relations between response gap and distance from onset.

Kadmon and Sompolinsky's 2015 primary abstract,
https://arxiv.org/abs/1508.06486 and DOI10.1103/PhysRevX.5.041030, already
emphasizes that onset properties depend on the transfer function. A broad
priority claim for curvature-sensitive onset would be unjustified. The full
literature on the particular PR crossover derived here has not been exhausted.

Primary follow-up: Clark, Marschall, van Meegen and Litwin-Kumar,
*Connectivity Structure and Dynamics of Nonlinear Recurrent Neural Networks*,
PRX15,041019(2025), https://arxiv.org/html/2409.01969v3 , SectionVI and
AppendixJ already treat general neuron-parameter heterogeneity, same-neuron
spectral products, and the difference between heterogeneity inside recurrent
dynamics and independent readout amplitudes. The broad conditional-heterogeneity
construction must be credited to that framework. Its checked sections do not
state the degree-induced mean-one crossover derived below; this is not an
exhaustive novelty determination. Search-engine keyword coverage was poor,
so absence of a search hit is not evidence of priority.

Let r=g/g_c-1 and define the stationary conditional gain gap

    delta = 1-g^2 A(q),   A(q)=E_mu[(E_u phi'(mu+u))^2].

It is not 1-g^2 E[phi'(x)^2] inside one time-dependent trajectory.
At onset put A=E phi'(mu)^2, B=E phi''(mu)^2 and C=E phi'''(mu)^2.
Price expansion gives

    F(K)=A(q)K+B(q)K^2/2+C(q)K^3/6+...,
    K''=delta K-(g^2 B/2)K^2-(g^2 C/6)K^3+... .

Multiplication by2K' and integration from the mixing tail gives

    (K')^2=delta K^2-(g^2 B/3)K^3-(g^2 C/12)K^4+...,
    delta=(g^2 B/3)q+(g^2 C/12)q^2+...,  q=K(0).

If B>0, the quadratic covariance term dominates for sufficiently small q,
giving q=3delta/(g_c^2 B) and K/q=sech^2(sqrt(delta)tau/2). If B=0 and C>0,
q^2=12delta/(g_c^2 C) and K/q=sech(sqrt(delta)tau).

The pair resonance gives d=PR/N proportional to delta^(3/2) in both cases.
The shape-dependent coefficient differs. A generic positive transversality
delta~r yields r^(3/2). In the homogeneous odd case delta~r^2/3, giving r^3.
Thus the mechanism concerns the vanishing/nonvanishing quadratic covariance
coefficient and self-consistent gap relation, not a different PR definition.

## 2. Explicit biased homogeneous control

For phi(x)=2Phi(x) and Gaussian static input mu~N(h,Delta),

    A=(2/pi)/sqrt(1+2Delta) * exp[-h^2/(1+2Delta)],
    B/A=Delta/(1+2Delta)+h^2/(1+2Delta)^2.

B vanishes only when BOTH h=0 and Delta=0. A symmetric distribution with
h=0 but Delta>0 still has B>0 because the mean squared curvature is positive.
A point distribution with Delta=0 but h!=0 also has B>0.

For chi=0, prescribe mean rate m and let a=Phi^-1(m/2). Then Delta=0,
h=a sqrt(1+q), A(q)=phi'(a)^2/(1+q), and g_c=1/phi'(a).
Expanding g^2 A(q) gives delta=q-2r+O(q^2,qr,r^2).
At onset g_c^2 B=a^2. Combining with the energy condition gives

    q/r -> 6/(3-a^2),   delta/r -> 2a^2/(3-a^2),

provided a^2<3 so this is a continuous positive branch. At m=.3,
a=-1.0364334, these limits are3.1155789 and1.1155789. Exact q=1e-4
continuation gives3.1155669 and1.1154233. This homogeneous biased network
therefore has the generic three-halves law.

At m=1, a=0, B=0, and the next order is required:
q/r->2, delta/r^2->1/3. At q=1e-4 the computed values are2.0000667 and
.3332889. The gap prefactor is12/pi^2, and d~4r^3/(sqrt(3)pi^2), reproducing
the homogeneous odd reference.

The a^2=3 transversality boundary and a^2>3 branch behavior are outside the
present generic continuous-onset claim. No exponent is asserted there.

## 3. Uniform small-chi crossover at the inflection point

Set m=1, so h=0 exactly, d_s=Delta, s=1+d_s+q, and lambda=(2/pi)g^2.
The exact one-site static equation is

    d_s=lambda[(pi/2)chi+arcsin(d_s/s)].

The homoclinic energy condition is

    1=2lambda integral_0^1 (1-y)/sqrt[s^2-(d_s+qy)^2] dy.

To expand the latter, use

    2 integral(1-y)(d_s+qy)^2 dy
      =d_s^2+(2/3)d_s q+q^2/6,

which gives

    lambda=1+d_s+q-d_s^2/2-d_s q/3-q^2/12+O((d_s+q)^3),
    delta=d_s q/3+q^2/12+O(q(d_s+q)^2).

For the static equation,

    lambda arcsin(d_s/s)
     =d_s-d_s^3/3-d_s^2 q/3-d_s q^2/12+O((d_s+q)^4),

so

    d_s(2d_s+q)^2=6pi chi+O((d_s+q)^4+chi(d_s+q)).

At onset q=0,

    Delta_c=(3pi chi/2)^(1/3)[1+O(chi^(1/3))].

This is the missing a=0 case left open by the source's sqrt(chi) result for
nonzero bias. It is a variance exponent1/3; the static standard deviation
has exponent1/6. Do not conflate these.

Let dc=(3pi chi/2)^(1/3), Q=q/dc and Z=Delta/dc. The leading uniform curve is

    Z(2Z+Q)^2=4,
    r/dc=(Z+Q-1)/2,
    delta/dc^2=Z Q/3+Q^2/12.

Here r=g/g_c(chi)-1, not distance from g_c(0). For Q<<1,
Z=1-Q/3+..., q~3r and delta~dc*r. For Q>>1 with q still small,
Z~4/Q^2, Delta~6pi chi/q^2, q~2r and delta~r^2/3.
The crossover occurs at r_chi~chi^(1/3). The limits chi->0 and r->0 do
not commute in the observed coupling-distance exponent.

The biased m=.3 case does not acquire cubic outer behavior by removing
heterogeneity, because its point operating point already has nonzero curvature.

## 4. Parameter-free interpolation of kernel and PR

Define beta=Delta/3, gamma=1/12 and retain the mixed normal form

    (K')^2=K^2(delta-beta K-gamma K^2),
    delta=beta q+gamma q^2.

Writing U=1/K gives U'^2=delta U^2-beta U-gamma. Solving the shifted
hyperbolic equation and imposing the peak gives

    K(tau)=2delta/[beta+sqrt(beta^2+4gamma delta) cosh(sqrt(delta)tau)].

Equivalently, with a=4Delta/(4Delta+q),

    k_a(s)=K/q=2/[a+(2-a)cosh s],   s=sqrt(delta)tau.

It interpolates from sech^2(s/2) at a=1 to sech(s) at a=0.
Let theta=arccos[a/(2-a)]. Its Fourier transform is

    f_a(x)=4pi sinh(theta*x)/[(2-a)sin(theta)sinh(pi*x)],

with the continuous a=1 limit4pi*x/sinh(pi*x). The resonant pair integral
is then

    C(a)=(1/4pi) integral_-infinity^infinity f_a(x)^2/(1+x^2) dx,
    d~(A^2/A4) delta^(3/2)/C(a).

In the joint small-chi mean1 limit A^2/A4->1. The endpoints are
C(0)=pi^2/12 and C(1)=4pi^2/3-12. The full scaling collapse is

    d/chi -> (3pi/2) [Z Q/3+Q^2/12]^(3/2)/C(4Z/(4Z+Q)).

No coefficient or crossover scale is fitted.

## 5. Numerical checks and failed extensions

`critical_crossover.py/json/log` record exact branch continuations for
chi=1e-3,1e-6,1e-9,1e-12 and seven Q values; the normal-form static/gap
ratios converge toward one as chi decreases at fixed Q. At larger q or chi
they depart substantially, as the small-parameter expansion predicts. These
departures are retained rather than selecting only asymptotic points.

Five independently resonance-resolved pair integrations use exact DMFT
kernels. At mean1, chi=1e-9, Q=.1,1,10, ratios of actual pair dimension to
the mixed-shape prediction using the actual gap/Delta are1.000159,
1.001054,1.004108. Mean1 chi0 q=.001 gives1.000092 for the odd-kernel
coefficient. Biased homogeneous mean.3 chi0 q=.001 gives1.01939 for the
quadratic-kernel asymptote; this finite-gap discrepancy is recorded.

These are tests within the conditional theory. Small negative interpolated
FFT tails at some points are recorded in the JSON; no numerical tail clipping
was used to force agreement. A full continuum/quadrature refinement campaign
was not performed for every new crossover point.

`crossover_lyapunov.py/json/log` independently solves the normal-form
Schrodinger operator

    H_a/delta=-partial_s^2+1-3a k_a(s)-6(1-a)k_a(s)^2.

Its translation mode approaches zero on grid refinement. The ground state
predicts lambda_max/delta ->5/8 at the quadratic endpoint and3/2 at the
cubic endpoint; numerical endpoint values at ds=.01 are within2e-5 relative.
This is a scalar normal-form stability test, not a network Lyapunov test.

## 6. Necessary visibility and recording conditions

The extensive critical theory needs N*d>>1. In the mean1 small-chi inner
regime d~(dc*r)^(3/2)/C_quad. Therefore

    r >> C_quad^(2/3)/(dc*N^(2/3)),    while r << dc.

A nonempty asymptotic window requires N*chi>>1, up to order-one constants.
This is necessary, not sufficient: finite-network threshold fluctuations,
response localization and record-length bias can impose stronger conditions.
At crossover delta~chi^(2/3), tau_pair~chi^(-2/3); in the inner region
tau_pair~1/(dc*r), whereas tau_neuron~1/sqrt(dc*r). This diagonal-pair-lag
time is NOT a universal recording-duration criterion. The centering
contraction psi(0,tau) instead has integrated time
tau_E~f_a(0)^2/[4C(a)]delta^-1/2. Connected four-time estimator contractions
still need separate analysis. The rejected stronger inference and full
corrected derivation are preserved in critical_recording_correction.md.

For the scalar UNIFORM-variance mean.3 Bernoulli critical example in the
original critical section, d~.112974 r^(3/2). This is not the coefficient of
the nonuniform60:40 g4 benchmark. In that scalar example, at N1000,
r=.02 predicts dimensionD=.320<1; even r=.1 givesD=3.57. The condition
N*d>>1 requires r much larger than approximately.0427. Simulating a few
N500/1000 points arbitrarily close to onset would therefore not validate
the thermodynamic exponent. No such misleading exponent campaign was run.
For the exact small-chi theoretical tests above, N*chi>>1 would require
enormous dense networks; the tests must remain explicitly analytical/numerical
DMFT checks until a feasible size-duration regime is identified.

## Paper use and unresolved work

Adopt the explicit biased homogeneous control now; it prevents an incorrect
novelty claim. The uniform mean1 crossover is a promising central mechanism:
degree disorder generates curvature self-consistently, producing a shrinking
three-halves region and a calculable crossover to the odd homogeneous cubic
law. Its derivation and deterministic tests are ready for scrutiny, but prior
work and direct network verification remain to be assessed. Do not call the
exponent alone a newly established universal class or claim finite-network
critical validation from these calculations.
