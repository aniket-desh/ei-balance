# Primary literature check for manuscript extraction

Checked 11 November 2026. This is a focused comparison, not an exhaustive
priority search. Bibliographic facts and the claims below were checked on
primary full texts or journal records.

- Clark, Abbott and Litwin-Kumar (2023), *Dimension of Activity in Random
  Neural Networks*, PRL131,118401, DOI10.1103/PhysRevLett.131.118401.
  [Primary full text](https://arxiv.org/html/2207.12373v3): introduction and
  two-site construction. The paper supplies self-consistent cross-covariance
  theory, activity PR and collective timescales. Our manuscript credits this
  foundation rather than claiming two-site covariance theory as new.
- Clark, Marschall, vanMeegen and Litwin-Kumar (2025), *Connectivity Structure
  and Dynamics of Nonlinear Recurrent Neural Networks*, PRX15,041019,
  DOI10.1103/2jt7-c8cq.
  [Primary author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC12782219/),
  sectionVI and AppendixJ: general single-neuron heterogeneity and the
  distinction between recurrent gains and readout rescaling already appear.
  AppendixJ also retains same-label spectral and response products directly;
  those moments themselves are not claimed as a new construction here.
- Kadmon and Sompolinsky (2015), *Transition to Chaos in Random Neuronal
  Networks*, PRX5,041030, DOI10.1103/PhysRevX.5.041030.
  [Journal record](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.5.041030):
  multitype EI balance, positive rates, chaotic rate dynamics and transfer-function
  dependence of onset. The manuscript does not claim those foundations anew.
- Mastrogiuseppe and Ostojic (2017), *Intrinsically-generated fluctuating
  activity in excitatory-inhibitory networks*, PLOSComputBiol13,e1005498,
  DOI10.1371/journal.pcbi.1005498.
  [Primary full text](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005498),
  “Connectivity with stochastic in-degree” and its Methods derivation:
  fixed/random degree and the resulting static heterogeneity are established
  prior results. Our degree intervention concerns their conditional
  covariance-geometry consequences, not discovery of degree-induced disorder.
- vanVreeswijk and Sompolinsky (1996), *Chaos in Neuronal Networks with
  Balanced Excitatory and Inhibitory Activity*, Science274,1724–1726,
  DOI10.1126/science.274.5293.1724.
  [Primary journal abstract via PubMed](https://pubmed.ncbi.nlm.nih.gov/8939866/):
  basis for the introductory statement about balanced irregular activity.
- Clark (2026), *Linear equivalence of nonlinear recurrent neural networks*,
  [arXiv2604.23489v3](https://arxiv.org/html/2604.23489v3), abstract and introduction:
  fixed-connectivity covariance equivalence has its own cavity derivation and
  assumptions. SectionVII restricts the derivation to i.i.d.couplings and
  lists structured/cell-type connectivity as extensions. Positive rates are
  not by themselves a distinction, since mean-centering lifts the stated
  sign-flip restriction. We do not present the archive's imperfect fixed-connectivity
  implementation as a validated extension or novelty of the current paper.

The new population averaging proposition is stated as a result of the current
conditional algebra, without an unsupported first-in-literature claim.

The normalized cross-population covariance energy is also an established
measure. Kornblith, Norouzi, Lee and Hinton(2019), *Similarity of Neural Network
Representations Revisited*, PMLR97:3519–3529,
[primary paper](https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf),
section3 equations1,4 and table1, gives the identical linear centered-kernel
alignment formula. The paper credits it as linear CKA. Its use to interpret
the EI population prediction is not a claim to have invented the measure.
