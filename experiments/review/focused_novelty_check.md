# Focused comparison with the closest Clark papers

Checked 11 November 2026 against primary full texts. This is a comparison with
three specified papers, not a comprehensive priority search. It limits the
claims made for the current EI calculation.

## What the preceding frameworks already contain

Clark, Abbott and Litwin-Kumar (2023), *Dimension of Activity in Random Neural
Networks*, develops two-site cavity theory for cross-covariance statistics,
activity PR, and collective timescales. It provides the method being adapted
here. [Primary paper](https://arxiv.org/html/2207.12373v3), main Eqs. 2–3.

Clark, Marschall, van Meegen and Litwin-Kumar (2025), *Connectivity Structure
and Dynamics of Nonlinear Recurrent Neural Networks*, Appendix J, explicitly
retains an outer average over neuron-specific parameters after multiplying
two local spectra or two local averaged responses: Eqs. 156a and 156b are the
direct counterparts of our Q and u. Their Eq. 158 propagates these quantities
through the random-mode model's collective response. Section VI distinguishes
recurrent gain heterogeneity from readout rescaling; those effects are not new
discoveries here. Section VIII.4 notes that the random-mode model does not
enforce Dale's law. The explicit firing-rate gains of Appendix J.3 are drawn
independently of connectivity. A self-consistent balance-generated label is
therefore a different specialization, whose endogenous dependence still needs
the cavity assumptions stated in our derivation.
[Primary paper](https://arxiv.org/html/2409.01969v3), Section VI, Appendix J,
and Sections VIII.3–VIII.4.

Clark (2026), *Linear equivalence of nonlinear recurrent neural networks*,
derives covariance and response representations for a typical quenched
coupling realization. Section I.1, Eq. 4, specifies zero-mean i.i.d. couplings;
Section VII explicitly limits the structured-connectivity conclusion and
lists cell-type or regional structure among extensions. Its derivation does
not establish the present strict-Dale block-mean, nonuniform-variance,
degree-correlated ensemble. Positive rates alone are not a distinction:
footnote 2 says mean-centering lifts the main text's sign-flip restriction.
Nor should an ensemble-averaged PR calculation here be described as a new
fixed-connectivity covariance equivalence or an improvement on that paper's
elementwise result. Its relevant warning is that recurrent disorder–activity
dependence requires a derivation; external labels cannot simply be relabeled
as endogenous ones without conditions.
[Primary paper](https://arxiv.org/html/2604.23489v3), Sections I.1 and VII,
footnote 2, and Appendix G.

## The precise specialization being established here

The following correspondences are mathematical comparisons made for this
project, rather than claims that the previous papers prove the EI extension:

| Present construction | Relationship to the existing framework |
|---|---|
| Q_a = E_z[c_a(z,w1)c_a(z,w2)] | The same-label spectral product already has a general counterpart in the 2025 paper's Eq. 156a. |
| u_a = E_z[s_a(z,w1)s_a(z,w2)] | Its response counterpart already appears in Eq. 156b. |
| Self-consistent z-dependent balanced operating points | Requires the balanced one-site construction and treatment of the label's dependence on recurrent connectivity. |
| R = [I − diag(u)GF]^{-1} and population block energies | The explicit EI population calculation and its normalizations are derived in this archive under stated closure assumptions. |
| Fixed-GF composition control versus degree-dispersion control | Distinct parameter paths, with explicit local-law and geometry consequences and separate numerical tests. |
| Blockwise early-averaging inequalities | Proved here for common leak, nonnegative response coefficients, a convergent response expansion, and fixed local laws. This comparison does not establish historical priority. |
| Typical quenched covariance matrix | Outside the validated claims of this paper; the archive's earlier fixed-connectivity attempt remains incomplete. |

A restrained contribution paragraph is:

> We adapt established two-site covariance theory to a balanced two-population
> ensemble with a self-consistent operating-point distribution. Conditional on
> cavity decoupling and controlled coherent feedback, we derive population
> covariance energies, distinguish composition changes at fixed aggregate
> coupling from degree-dispersion interventions, and establish blockwise
> bounds on the errors caused by averaging local laws too early. Strict-Dale
> simulations test these predictions and their finite-network limitations.

The new network results must determine whether the last sentence supports a
quantitative domain or instead documents a residual failure. It should not be
upgraded because the algebra is complete. The scalar critical crossover can
remain supporting material without becoming the main novelty or a prerequisite
for finishing Nischal's original project.
