# Degree dispersion and finite-size skew

This is a diagnostic of the implemented degree generator. It explains why a
single degree-variance parameter does not determine finite-size corrections.
It does not predict the sign or magnitude of the PR residual.

For chi>1, the generator draws P from a Beta distribution with parameters
alpha=p*s and beta=(1-p)*s, then K|P from Binomial(M,P), where

    s=(M-chi)/(chi-1).

Its falling-factorial moments are

    F1=E[K]=M p,
    F2=E[K(K-1)]=M(M-1) p(ps+1)/(s+1),
    F3=E[K(K-1)(K-2)]=M(M-1)(M-2) p(ps+1)(ps+2)/[(s+1)(s+2)].

Use E[K^2]=F2+F1 and E[K^3]=F3+3F2+F1. With mu=M p,

    kappa2 = F2+F1-mu^2 = M p(1-p)(s+M)/(s+1),
    kappa3 = F3+3F2+F1-3mu(F2+F1)+2mu^3
           = M p(1-p)(1-2p)(s+M)(s+2M)/[(s+1)(s+2)].

Since (s+M)/(s+1)=chi, substitution gives

    kappa3 / [M p(1-p)(1-2p)]
       = chi * [((2chi-1)M-chi)/(M+chi-2)].

At chi=2 this factor is exactly 6-4/M, whereas the variance factor is 2.
Consequently, the standardized degree skew is approximately 3/sqrt(2) times
its Bernoulli value, for large M and fixed p. The skew still vanishes as
M^(-1/2); the leading Gaussian conditional law is unchanged by this observation.

For chi<=1 the implemented K is an integer constant plus independent
Bernoulli(fraction) and Binomial(L,p), L=round(chi*M). Therefore

    kappa2 = L p(1-p) + fraction(1-fraction),
    kappa3 = L p(1-p)(1-2p)
             + fraction(1-fraction)(1-2fraction).

The generator families on either side of chi=1 share their intended limiting
variance, but their higher count cumulants have different chi dependence.

To isolate one input component, let each presynaptic population b have a
constant rate m_b and edge weight t_ab/sqrt(N). Its degree-forcing fluctuation
is

    eta_a = sum_b t_ab*m_b*(K_ab-M_b*p_ab)/sqrt(N).

Independent degree draws across b give

    kappa2(eta_a)=sum_b (t_ab*m_b)^2 kappa2(K_ab)/N,
    kappa3(eta_a)=sum_b (t_ab*m_b)^3 kappa3(K_ab)/N^(3/2).

The first is O(1); the second is O(N^(-1/2)). The beta-binomial chi=2 choice
therefore enhances this particular finite-size skew coefficient substantially.
The JSON records its actual values for the nonuniform EI controls.

This component is not the full static input: rates differ across neurons,
and those rate differences and their temporal fluctuations correlate with
recurrent connectivity. A controlled PR correction would also require the
other same-order source and response terms. Count skew alone supplies neither
a fitted PR correction nor a causal explanation of the observed endpoint bias.

The accompanying script checks all three degree-generator regimes against
finite-count probability masses at four pool sizes and two edge probabilities.
