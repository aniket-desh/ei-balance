"""Regenerate paper tables from checked campaign summaries, without fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args()
    comp_path=ROOT/'experiments/composition/prospective_summary.json'
    degree_path=ROOT/'experiments/degree/comparison.json'
    comp=json.loads(comp_path.read_text())
    degree=json.loads(degree_path.read_text())
    if not args.allow_incomplete:
        assert comp['completed_primary_count']==comp['expected_count']==40
        assert degree['complete_cohort']
        assert degree['complete_networks']==degree['expected_networks']==24
    comp_rows=[]
    interval_rows=[]
    population_rows=[]
    for key,group in sorted(comp['groups'].items(),key=lambda item:
                          (int(item[0].split('_N')[1]),float(item[0].split('_')[0][1:]))):
        fe=float(key.split('_')[0][1:]);size=int(key.split('_N')[1])
        if group['n']<2:
            continue
        reference=np.array([group['parameter_only'][name] for name in ['nu','energy','d']])
        ratios=np.array(group['mean'])/reference
        ci=np.array(group['ci95_t'])/reference[:,None]
        assert np.isfinite(ratios).all() and np.isfinite(ci).all()
        comp_rows.append(f'{fe:.1f}&{size}&{group["n"]}&{ratios[0]:.3f}&{ratios[1]:.3f}&{ratios[2]:.3f}&$[{ci[2,0]:.3f},{ci[2,1]:.3f}]$'+r'\\')
        for j,symbol in enumerate([r'$\nu$',r'$\energy$',r'$d$']):
            interval_rows.append(f'{fe:.1f}&{size}&{symbol}&{100*(ratios[j]-1):+.2f}&$[{100*(ci[j,0]-1):+.2f},{100*(ci[j,1]-1):+.2f}]$'+r'\\')
        if size>=2000:
            frac=group['within_population_dimension_fractions']
            absolute=group['within_population_absolute_dimensions']
            for a,label in enumerate(['E','I']):
                predicted_fraction=group['parameter_only']['within_population_dimension_fractions'][a]
                predicted_absolute=group['leading_within_population_absolute_dimensions'][a]
                population_rows.append(f'{fe:.1f}&{size}&{label}&{group["n"]}&{predicted_fraction:.4f}&${frac["mean"][a]:.4f}\\pm{frac["se"][a]:.4f}$&{predicted_absolute:.2f}&${absolute["mean"][a]:.2f}\\pm{absolute["se"][a]:.2f}$'+r'\\')
    pre=r'''\begin{table}[t]
\centering
\caption{Fresh composition and size series. Ratios compare the mean of each
network observable with its parameter-only prediction. The interval is a
conventional 95\% Student-$t$ interval for the mean dimension ratio across
independent seed clusters within each condition; it does not include
uncontrolled finite-record bias or provide simultaneous coverage. The
staged allocation and every outcome are retained.}
\label{tab:composition}
\begin{tabular}{rrrrrrr}
\toprule
$f_E$&$N$&$n$&$\bar\nu/\nu_0$&$\bar\energy/\energy_0$&$\bar d/d_0$&95\% interval\\
\midrule
'''
    (HERE/'composition_table.tex').write_text(pre+'\n'.join(comp_rows)+r'''
\bottomrule
\end{tabular}
\end{table}
''')
    interval_pre=r'''\begin{table}[htp]
\centering
\caption{Descriptive mean-error intervals for the fresh composition series.
Errors and interval endpoints are percentages relative to the parameter-only
prediction for each observable. These are pointwise finite-record sampling
intervals, not a uniform accuracy guarantee or an all-orders remainder bound.}
\label{tab:ingredient_intervals}
\begin{tabular}{rrrrr}
\toprule
$f_E$&$N$&Observable&Mean error (\%)&95\% error interval (\%)\\
\midrule
'''
    (HERE/'composition_intervals.tex').write_text(interval_pre+'\n'.join(interval_rows)+r'''
\bottomrule
\end{tabular}
\end{table}
''')
    population_pre=r'''\begin{table}[htp]
\centering
\small
\caption{Original composition size series. Fractions are
$d_a=D_a/N_a$; absolute dimensions are $D_a$. Measured entries are means of
network ratios with network standard errors. Both are obtained from the
same primary mean-difference covariance estimator; they are not additional
independent observations.}
\label{tab:population_dimensions}
\begin{tabular}{rrrrrrrr}
\toprule
$f_E$&$N$&Class&$n$&Predicted $d_a$&Measured $d_a$&Predicted $D_a$&Measured $D_a$\\
\midrule
'''
    (HERE/'population_dimensions.tex').write_text(population_pre+'\n'.join(population_rows)+r'''
\bottomrule
\end{tabular}
\end{table}
''')
    calibration_path=ROOT/'experiments/composition/local_calibration_summary.json'
    calibration=json.loads(calibration_path.read_text())
    calibration_rows=[]
    for key,group in sorted(calibration['groups'].items(),key=lambda item:
      (int(item[0].split('_N')[1].split('_')[0]),float(item[0].split('_')[0][1:]),int(item[0].split('_pop')[1]))):
        fe=float(key.split('_')[0][1:]);size=int(key.split('_N')[1].split('_')[0]);pop=int(key.split('_pop')[1])
        if size<2000:
            continue
        metrics=group['metrics'];n=metrics['A_Gaussian_individual_mu_q_error']['n']
        if n<2:
            continue
        cells=[]
        for name in ['A_Gaussian_individual_mu_q_error','A_Gaussian_empirical_mu_common_q_error',
                     'A_Gaussian_population_h_Delta_q_error','A_cross_block_vs_square_full_mean']:
            m=metrics[name]
            digits=3 if name=='A_cross_block_vs_square_full_mean' else 2
            cells.append(f'${100*m["mean"]:+.{digits}f}\\pm{100*m["se"]:.{digits}f}$')
        calibration_rows.append(f'{fe:.1f}&{size}&{["E","I"][pop]}&{n}&'+'&'.join(cells)+r'\\')
    calibration_pre=r'''\begin{table}[htp]
\centering
\footnotesize
\caption{Gain calibration diagnostics on the instrumented subset. Entries are
mean percentage errors with network standard errors, normalized by the
measured squared time-averaged gain moment. These are population moment
errors, not a uniform per-neuron guarantee. The three Gaussian formulas use
measured individual $(\mu_i,q_i)$; measured $\mu_i$ with common $q$; or a
Gaussian operating-point distribution with measured population $(h,\Delta,q)$.
The block check compares cross-block gain products with the squared full-record
mean. These same-record calibrations are not parameter-only predictions or
tests of the entire Gaussian trajectory law. The $N=2000$ subset consists of
initial seed 3001 and all eight additional instrumented networks per
composition; the $N=4000$ subset uses all four.}
\label{tab:gain_calibration}
\begin{tabular}{rrrrrrrr}
\toprule
$f_E$&$N$&Class&$n$&\shortstack{Individual\\$\mu_i,q_i$}&\shortstack{Empirical $\mu_i$\\common $q$}&\shortstack{Gaussian $\mu$\\common $q$}&Gain block check\\
\midrule
'''
    (HERE/'gain_calibration.tex').write_text(calibration_pre+'\n'.join(calibration_rows)+r'''
\bottomrule
\end{tabular}
\end{table}
''')
    degree_rows=[]
    for group in sorted(degree['groups'],key=lambda g:(g['N'],g['chi'])):
        metric=group['metrics']['d'];n=metric['count']
        degree_rows.append(f'{group["N"]}&{group["chi"]:g}&{n}&{group["theory"]["prediction"]["dimension_fraction"]:.6f}&{metric["mean"]:.6f}&{metric["se"]:.6f}'+r'\\')
    degree_pre=r'''\begin{table}[t]
\centering
\caption{Nonuniform degree test. Measurements are network means with
standard errors. The theoretical value is the ideal requested-dispersion
prediction; integer-count rounding is recorded separately. The initial 18
networks and the staged six-network extension are distinguished in the text.}
\label{tab:degree}
\begin{tabular}{rrrrrr}
\toprule
$N$&$\chi$&$n$&Predicted $d$&Measured $d$&Network SE\\
\midrule
'''
    (HERE/'degree_table.tex').write_text(degree_pre+'\n'.join(degree_rows)+r'''
\bottomrule
\end{tabular}
\end{table}
''')
    record={'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [comp_path,degree_path,calibration_path]},
            'composition_complete':comp['completed_primary_count'],
            'composition_expected':comp['expected_count'],
            'degree_complete':degree['complete_networks'],
            'degree_expected':degree['expected_networks'],
            'composition_table_rows':len(comp_rows),
            'degree_table_rows':len(degree_rows),
            'population_dimension_rows':len(population_rows),
            'gain_calibration_rows':len(calibration_rows),
            'generated_from_incomplete_data':bool(args.allow_incomplete)}
    (HERE/'table_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
