# Notation for the conditional EI paper

Names vary across the literature. The definitions below are the conventions
used by this manuscript. All covariances concern neuron-wise temporally
centered rates unless an input covariance is explicitly named.

| Symbol | Conventional name | Definition and intuitive meaning |
|---|---|---|
| $a,b\in\{E,I\}$ | Cell-class or population indices | Postsynaptic population is the first connectivity index; presynaptic population is the second. |
| $N_a,f_a,F$ | Population size and fraction | $N_a=f_aN$, $F=\mathrm{diag}(f_a)$. Fractions count neurons. |
| $x_i,r_i$ | Input or preactivation, and firing rate | $r_i=\phi(x_i)$, with $\phi=2\Phi$ in the numerical model. |
| $J_{ij}$ | Synaptic coupling | The signed influence of neuron $j$ on neuron $i$. Its mean is $J^0_{ab}/\sqrt N$ and centered variance is $G_{ab}/N$. |
| $J^0,G,W$ | Mean connectivity, variance profile, aggregate variance matrix | $W=GF$. Population fractions enter the total variance arriving from a presynaptic class. |
| $\bar m_a$ | Population mean rate | The leading balance solution satisfies $J^0F\bar m+I=0$. It need not equal a measured finite-network mean exactly. |
| $h_a^{\rm ext}$ | Order-one external input | The residual external drive added to the leading $\sqrt N I_a$ term. It is distinct from the measured or theoretical population input mean $h_a$. |
| $h_a$ | Population input mean | The center of the operating-point distribution. For positive erf, $h_a=\sqrt{1+\Delta_a+q_a}\,\Phi^{-1}(\bar m_a/2)$. |
| $z,\mu_a(z)$ | Quenched label and operating point | $z\sim\mathcal N(0,1)$ is fixed for a neuron; $\mu_a(z)=h_a+\sqrt{\Delta_a}z$ is its time-averaged input in the conditional closure. |
| $\Delta_a$ | Quenched or static input variance | Variance across neuronal operating points. Its square root, not $\Delta_a$ itself, is the static input standard deviation. |
| $u_a(t),K_a(\tau),q_a$ | Centered input process, autocovariance kernel, temporal variance | $x_a(t)\mid z=\mu_a(z)+u_a(t)$; $K_a(\tau)=\mathbb E[u_a(t+\tau)u_a(t)]$, $q_a=K_a(0)$. |
| $m_a(z)$ | Conditional mean rate | $\mathbb E_u\phi(\mu_a(z)+u)$, the mean rate of a neuron with a specified operating point. |
| $c_a(z,\tau),v_a(z)$ | Conditional rate autocovariance and temporal variance | $c_a$ subtracts the neuron's own mean rate; $v_a(z)=c_a(z,0)$. A common input kernel can produce different output kernels at different $z$. |
| $\alpha_a(z)$ | Time-averaged local gain | $\mathbb E_u\phi'(\mu_a(z)+u)$. It is the mean local slope before recurrent propagation, not a Lyapunov exponent or the whole recurrent response. |
| $A_a,A_{4a}$ | Gain second and fourth moments | $A_a=\mathbb E_z\alpha_a(z)^2$, $A_{4a}=\mathbb E_z\alpha_a(z)^4$. These average powers of time-averaged slopes across neurons; they are not instantaneous slope moments. |
| $H(\omega)$ | Leak transfer function | $(1+i\omega)^{-1}$ for common unit leak time and the Fourier convention $e^{-i\omega\tau}$. |
| $Q_a(\omega_1,\omega_2)$ | Same-label spectral product or local source moment | $\mathbb E_z[c_a(z,\omega_1)c_a(z,\omega_2)]$. Both factors belong to the same neuron; averaging them separately discards source heterogeneity. |
| $U,R$ | Local response-product matrix and population resolvent | $U=H_1H_2\mathrm{diag}(A_a)$, $R=(I-UW)^{-1}$. The inverse sums repeated recurrent propagation. |
| $\mathsf D_{\rm src},X,\Psi$ | Source matrix, full pair-energy kernel, distinct-site kernel | $\mathsf D_{\rm src}=\mathrm{diag}(Q_a/f_a)$, $X=R\mathsf D_{\rm src}R^\dagger$, $\Psi=X-\mathsf D_{\rm src}$. Diagonal neuronal contributions are restored in $X$. |
| $C_{ij}(\tau)$ | Rate cross-covariance | $\langle[r_i(t+\tau)-m_i][r_j(t)-m_j]\rangle_t$. It is not Pearson correlation. |
| $\nu,\mathcal E$ | Mean temporal rate variance and normalized covariance energy | $\nu=\mathrm{tr}C/N$, $\mathcal E=\mathrm{tr}C^2/N$ at zero lag. One measures total variance; the other measures its concentration, including pair correlations. |
| $\mathcal E_{ab}$ | Population covariance-energy contribution | $N^{-1}\sum_{i\in a,j\in b}C_{ij}^2$. Within-population terms include the diagonal; total energy is $\mathcal E_{EE}+2\mathcal E_{EI}+\mathcal E_{II}$. |
| $D,d$ | Participation ratio and dimension fraction | $D=(\mathrm{tr}C)^2/\mathrm{tr}C^2$, $d=D/N=\nu^2/\mathcal E$. It is an activity-variance dimension. |
| $D_a,d_a$ | Population participation ratio and per-neuron fraction | $D_a=(\mathrm{tr}C_{aa})^2/\mathrm{tr}C_{aa}^2$, $d_a=D_a/N_a$. Absolute dimensions and per-neuron fractions answer different questions. |
| $w_a$ | Fraction of total temporal variance | $w_a=\mathrm{tr}C_{aa}/\mathrm{tr}C=f_a\bar v_a/\nu$. These weights are distinct from neuron fractions $f_a$. |
| $\mathsf K_a$ | Temporal Gram matrix | For centered neuron-by-time data $Y_a$, $\mathsf K_a=Y_a^\top Y_a/T$. It is distinct from the one-site input autocovariance kernel $K_a(\tau)$. |
| $\mathcal O_{EI}$ | Temporal Gram alignment, or linear CKA | $\mathcal E_{EI}/\sqrt{\mathcal E_{EE}\mathcal E_{II}}$. It compares the populations' temporal variance geometry and is invariant under separate common population amplitude rescalings. |
| $K_{ab}^{\rm count},\chi_{ab}$ | Incoming degree and count dispersion | $\chi_{ab}=\mathrm{Var}(K_{ab}^{\rm count})/[N_bp_{ab}(1-p_{ab})]$. This count is distinct from the autocovariance kernel; rounding gives a finite-size difference between requested and actual dispersion. |
| $D_{\rm source}$ | Weighted filtered-source dimension | A frozen-row quadratic-form diagnostic involving the presynaptic covariance after leak filtering and connectivity weighting. It must not be silently identified with the network's raw activity PR. |

In numerical spectral arrays, a stored entry can be a spectral **mass**,
already containing $\mathrm d\omega/(2\pi)$. Such arrays are summed once;
multiplying by an additional integration factor changes the observable.

For normalized covariance eigenvalues $p_\ell=\lambda_\ell/\sum_k\lambda_k$,
PR is $D=1/\sum_\ell p_\ell^2$, the exponential of order-two Rényi entropy.
It is distinct from the exponential of Shannon entropy. The paper uses the
trace definition throughout.
