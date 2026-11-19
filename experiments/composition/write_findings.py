"""Render the prospective composition evidence without upgrading its claims."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent


def fmt(x,n=5):return f'{x:.{n}f}'

def main():
    d=json.loads((HERE/'prospective_summary.json').read_text());s=[]
    s+=['# Prospective composition findings','',f"Completed primary measurements: {d['completed_primary_count']} of {d['expected_count']} prespecified composition jobs.", '', 'The primary estimator is B500, T8000, gap0. All planned statuses are retained. The independent statistical units are seed clusters. Conventional network-mean t intervals describe the finite-record estimator; they do not prove stationarity, window independence, or asymptotic approximation bounds.','', '| fE | N | Networks | Mean PR/N ± network SE | 95% mean interval | Leading prediction | Relative mean error |','|---|---:|---:|---:|---:|---:|---:|']
    for key,x in d['groups'].items():
        f,N=key.split('_N');m=x['mean'][2];se=x['se'][2]if x['se']else float('nan');ci=x['ci95_t'][2]if x['ci95_t']else[float('nan')]*2
        s.append(f"| {f[1:]} | {N} | {x['n']} | {fmt(m)} ± {fmt(se)} | [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(x['parameter_only']['d'])} | {x['relative_mean_errors_nu_energy_d'][2]*100:+.1f}% |")
    s+=['','## Ingredient errors remain visible','','| fE,N | nu error | Covariance-energy error | Relative PR-error interval | Mean ratio minus ratio of mean moments |','|---|---:|---:|---:|---:|']
    for k,x in d['groups'].items():
        ci=x.get('relative_error_ci95_t_nu_energy_d',[[float('nan')]*2]*3)[2];r=x['relative_mean_errors_nu_energy_d']
        s.append(f"| {k} | {r[0]*100:+.1f}% | {r[1]*100:+.1f}% | [{ci[0]*100:+.1f}%, {ci[1]*100:+.1f}%] | {fmt(x['mean_ratio_minus_ratio_of_means'],6)} |")
    s+=['','A broad interval containing the leading prediction is not an equivalence result. Any accuracy statement must give the observable, network size, and uncertainty. The formal first-order correction has not been validated as a finite-network repair.','', '## Static and temporal variance','','| fE,N | Population | Mean Delta | Mean q | Mean Delta+q | Leading Delta+q | Across-network corr(Delta,q) |','|---|---|---:|---:|---:|---:|---:|']
    for k,x in d['groups'].items():
        for pop,z in x['Delta_q_countermotion'].items():
            s.append(f"| {k} | {'E'if pop=='0'else'I'} | {fmt(z['Delta']['mean'],3)} | {fmt(z['q']['mean'],3)} | {fmt(z['total_Delta_plus_q']['mean'],3)} | {fmt(z['parameter_only_total'],3)} | {fmt(z.get('corr',float('nan')),3)} |")
    s+=['','These empirical correlations are exploratory, particularly in four-network groups. Total input variance is not held fixed by the measured networks. The exact conditional fixed-total-variance derivative and parameter-only degree tangent are documented separately; they do not provide a full quenched correction.','', '## Population covariance geometry','','The following secondary diagnostic normalizes the ordered EE,EI,IE,II energy matrix by its total. It tests allocation independently of the raw amplitude, whose errors remain above. The error is the Frobenius distance to the predicted normalized matrix divided by the norm of that prediction.','', '| fE,N | Error of mean normalized network matrix | Error of normalized ensemble mean energy matrix | Mean fraction touching population-uniform axes |','|---|---:|---:|---:|']
    for k,x in d['groups'].items():
        z=x['normalized_population_geometry'];s.append(f"| {k} | {z['ensemble_mean_shape_relative_error']*100:.2f}% | {z['ratio_of_mean_energies_relative_shape_error']*100:.2f}% | {x['coherent_projection_fraction']['mean']*100:.3f}% |")
    s+=['','A small population-uniform sector rules out that particular subspace as the entire source of excess energy. It does not exclude nonuniform collective modes or quantify all coherent feedback effects.','', '## Paired size comparisons','','Common-seed contrasts use only the prespecified first four seed clusters. Additional N2000 networks increase within-point precision; they are not inserted as independent unmatched observations into a size contrast.','']
    for k,x in d['paired_size'].items():
        z=x['nu_energy_d_difference'];m=z['mean'][2];ci=z['ci95_t'][2]if z['ci95_t']else[float('nan')]*2
        s.append(f"- {k}: paired PR/N change {fmt(m)}; 95% paired interval [{fmt(ci[0])}, {fmt(ci[1])}]; {len(x['seeds'])} seed clusters.")
    s+=['','## E and I dimensions','','Population fractions use d_a=f_a nu_a^2/E_aa; absolute dimensions are D_a=N_a d_a. The cross-population rectangular block has no PR here.','','| fE,N | Mean dE ± network SE | Mean dI ± network SE | Mean DE ± network SE | Mean DI ± network SE | Predicted DE,DI |','|---|---:|---:|---:|---:|---:|']
    for k,x in d['groups'].items():
        ff=x['within_population_dimension_fractions'];aa=x['within_population_absolute_dimensions'];se=ff['se']or[float('nan')]*2;sa=aa['se']or[float('nan')]*2;pa=x['leading_within_population_absolute_dimensions']
        s.append(f"| {k} | {fmt(ff['mean'][0])} ± {fmt(se[0])} | {fmt(ff['mean'][1])} ± {fmt(se[1])} | {fmt(aa['mean'][0],2)} ± {fmt(sa[0],2)} | {fmt(aa['mean'][1],2)} ± {fmt(sa[1],2)} | {fmt(pa[0],2)}, {fmt(pa[1],2)} |")
    s+=['','The separately normalized cross-population energy E_EI/sqrt(E_EE E_II) is also retained in the JSON. It is a temporal-Gram overlap diagnostic, not communication capacity or a rectangular PR.','']
    s+=['','## What this does not establish','','The campaign does not derive every finite-size vertex or predict the full distribution of network-specific outliers. The conditional Gaussian local and pair closures remain assumptions about the fluctuating branch. Finite-window gain and static-variance checks on selected retained networks diagnose estimator and local-law effects but are not independent parameter-only predictions. Report a residual when ingredient errors or sampling uncertainty prevent a stronger accuracy claim.','']
    (HERE/'prospective_findings.md').write_text('\n'.join(s))


if __name__=='__main__':main()
