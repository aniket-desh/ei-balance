"""Write the final degree-test findings only after the declared cohort completes."""
from pathlib import Path
import json,numpy as np

HERE=Path(__file__).resolve().parent

def main():
    r=json.loads((HERE/'comparison.json').read_text())
    if not r['complete_cohort']:raise RuntimeError('Degree cohort is not complete.')
    groups=r['groups'];largest=max(g['N'] for g in groups)
    lines=['# Completed nonuniform EI degree test','',
           f"All {r['complete_networks']} declared degree networks completed and are retained. The original18-network test and its results are preserved in initial_18/. Six N4000 endpoint networks were declared as a staged extension after the N2000 residuals, before observing their own outcomes. Every parameter point has three fresh seed labels. The leading connectivity and external-input path is specified in degree_control_derivation.tex and adopted_protocol.json.",'',
           '| N | chi | theory PR/N | network mean | network SE | mean relative error |',
           '|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        m=g['metrics'];lines.append(f"| {g['N']} | {g['chi']:g} | {g['theory']['prediction']['dimension_fraction']:.6f} | {m['d']['mean']:.6f} | {m['d']['se']:.6f} | {100*g['relative_PR_error']:+.1f}% |")
    lines+=['','The degree effect transfers to genuinely nonuniform E/I local laws. Ratios of endpoint cohort means are:','']
    for N in sorted({g['N'] for g in groups}):
        pair=[next(g for g in groups if g['N']==N and g['chi']==c) for c in [0.,2.]]
        lines.append(f"- N{N}: {pair[0]['metrics']['d']['mean']/pair[1]['metrics']['d']['mean']:.2f}-fold, compared with {pair[0]['theory']['prediction']['dimension_fraction']/pair[1]['theory']['prediction']['dimension_fraction']:.2f}-fold predicted.")
    lines+=['','These are ratios of cohort means, without a fitted amplitude or an uncertainty calculation that treats coupled seed labels across conditions as independent. The mean PR, ratio of mean moments, and conventional network intervals remain separate in comparison.json.','',
            'The test also identifies quantitative limits. At the largest tested size:','']
    for g in [g for g in groups if g['N']==largest]:
        q=np.array(g['metrics']['q_temporal_input']['mean'])/np.array(g['theory']['dynamic_variances'])-1
        lines.append(f"- chi{g['chi']:g}: PR mean error {100*g['relative_PR_error']:+.1f}%; E/I temporal input variance errors {100*q[0]:+.1f}% / {100*q[1]:+.1f}%.")
    lines += ['', 'A large directional effect is not a uniform quantitative validation. The N2000 chi2 mean was 30% above the leading prediction and is retained even if later size points improve. Local-law and population-energy ingredient checks determine what may be claimed; a favorable final ratio alone does not.','']
    dg=max(abs(g['metrics']['gap1_minus_primary_d']['mean']) for g in groups)
    dt=max(abs(g['metrics']['T4000_minus_primary_d']['mean']) for g in groups)
    lines += [f"Mean gap1-minus-gap0 shifts are at most {dg:.3g}; T4000-minus-T8000 shifts are at most {dt:.3g}. These diagnostics constrain recording effects for the reported observables but do not prove the infinite-time limit. B1000 is checked only on the selected endpoint seed4001 examples.",'',
              'The figure degree_validation.pdf/png/svg shows the retained degree cohort, conditional theory, raw finite-T input-mean variances, temporal variances, and all three population energy contributions. Hollow chi1 symbols are a distinct composition-campaign reference with exactly the same Bernoulli connectivity law and simulation/estimator settings; they are not counted as extra degree replicates or pooled into the primary cohort.','']
    for ref in r.get('same_law_composition_references',[]):
        m=ref['metrics']['d'];lines.append(f"- Same-law reference N{ref['N']}: n={m['count']}, mean PR/N={m['mean']:.6f}, network SE={m['se']:.6f}.")
    status=HERE/'ingredient_diagnostics/status.json'
    if status.exists():
        local=json.loads(status.read_text());complete=[x for x in local['records'] if x['status']=='complete']
        lines+=['',f"Corrected local-moment diagnostics cover {len(complete)} of the {r['complete_networks']} degree records. Coverage is explicitly recorded in ingredient_diagnostics/status.json; these are not additional networks. The remaining primary summaries retain raw finite-T local moments.",'']
    lines+=['The selected chi0 N1000 seed4001 diagnostic shows that measured rate-mean heterogeneity inserted into the exact finite-pool row contraction predicts its observed static spread within about 2%. A local Gaussian expansion involving measured Var(q) and Cov(mu,q) explains much of the rate heterogeneity but retains a 7–12% residual in that example. It is a calibrated explanation, not a parameter-only finite-size PR correction. The later-size diagnostics are also retained.','',
            'The exact finite-pool covariance and third-cumulant identities, numerical refinements, static susceptibility, measured-source substitutions, and rejected assumptions are retained in this folder. Main-paper material, appendix algebra, and limitations are identified in manuscript_placement.md. No complete stationary finite-N theory, universal monotone convergence, or attractor probability is claimed.','']
    (HERE/'findings.md').write_text('\n'.join(lines))
    refs=r.get('same_law_composition_references',[])
    assert not any(p['missing'] for p in refs)
    caption=(f"Nonuniform EI degree intervention at fixed G, J0 and f=(0.8,0.2), with leading target mean rate0.3. The O(sqrt(N)) balance drive is fixed, while the O(1) offset h(chi) is recomputed from parameter-only theory before each condition. Lines are the conditional prediction without a fitted covariance amplitude. All {r['complete_networks']} degree-network outcomes are retained: the initial18 test plus a six-network N4000 staged endpoint extension. Each degree point has three fresh seed labels. A: PR/N, with faint individual network dots and one-network-SE error bars. Circles, squares and triangles denote N1000, N2000 and N4000. Hollow symbols at chi1 are separate same-law composition-campaign Bernoulli references (N2000 n12, N4000 n4); they are not pooled with or counted as degree replicates. B: full-record input-mean variances, blue E and red I; these finite-T statistics include temporal mean-estimation noise, with cross-block diagnostics reported separately. C: temporal input variances with the same population colors. D: energy contributions per total N, counting E-I twice, so the displayed terms sum to total energy. Primary energies use T8000, B500, gap0 and the four-window mean-free estimator; gap1 and T4000 are checked for every network and B1000 for selected endpoint examples. Correcting energy does not make PR exactly unbiased or prove temporal independence. The large dimensionality effect and the remaining quantitative discrepancies are both visible. The primary runner measures temporal fluctuations rather than Lyapunov exponents.\n")
    (HERE/'figure_caption.txt').write_text(caption)
    print(json.dumps({'networks':r['complete_networks'],'largestN':largest,'local_diagnostic_count':len(complete) if status.exists() else None},indent=2))

if __name__=='__main__':main()
