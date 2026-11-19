"""Typeset every selected integration and tangent check, without pooling."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def identity(case):
    parts=case.split('_')
    if 'f60' in parts:control='60:40'
    elif 'f80' in parts:control='80:20'
    elif 'chi0' in parts:control=r'$\chi=0$'
    elif 'chi2' in parts:control=r'$\chi=2$'
    else:raise ValueError('Unrecognized auxiliary control: '+case)
    size=int(next(p[1:] for p in parts if p.startswith('N')))
    seed=int(next(p[4:] for p in parts if p.startswith('seed')))
    return control,size,seed


def main():
    summary_path=ROOT/'experiments/runtime/runtime_completion_summary.json'
    summary=json.loads(summary_path.read_text())
    assert summary['status']=='complete'
    assert summary['primary_cases_completed']==summary['primary_cases_expected']==64
    assert summary['time_step_checks_completed']==summary['time_step_checks_expected']==5
    assert summary['outlier_diagnostic_cases_completed']==summary['outlier_diagnostic_cases_expected']==3
    rows=[]
    for record in summary['time_step_checks']:
        control,size,seed=identity(record['case'])
        assert record['same_connectivity_SHA256']
        changes=[100*record['relative_changes'][key] for key in ['nu','energy','d']]
        rows.append(f'{control}&{size}&{seed}&{record["compared_duration"]:g}&{changes[0]:+.3f}&{changes[1]:+.3f}&{changes[2]:+.3f}'+r'\\')
    pre=r'''\begin{table}[htp]
\centering
\small
\caption{Selected time-step checks at the same connectivity and initial
condition. Entries are percentage changes from midpoint step 0.05 to 0.025.
The largest-size comparison uses matched 4000-time prefixes; the others use
8000 recorded times. These changes combine integration and finite-trajectory
sampling effects; one selected outlier is retained rather than averaged away.}
\label{tab:time_step_checks}
\begin{tabular}{rrrrrrr}
\toprule
Control&$N$&Seed&Duration&$\Delta\nu$(\%)&$\Delta\energy$(\%)&$\Delta d$(\%)\\
\midrule
'''
    (HERE/'time_step_checks.tex').write_text(pre+'\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{table}
''')
    tangent_files=sorted((ROOT/'experiments/runtime/checks').glob('*_tangent.json'))
    assert len(tangent_files)==4
    tangents=[]
    for path in tangent_files:
        record=json.loads(path.read_text())
        assert record['duration']==2000 and len(record['eight_duration_blocks'])==8
        control,size,seed=identity(record['case_name'])
        blocks=record['eight_duration_blocks']
        tangents.append(f'{control}&{size}&{seed}&{record["largest_finite_time_exponent"]:.4f}&{min(blocks):.4f}&{max(blocks):.4f}'+r'\\')
    pre=r'''\begin{table}[htp]
\centering
\small
\caption{Selected tangent-flow checks at $N=2000$. The largest finite-time
exponent is measured over 2000 leak times after 500 burn-in times. Eight
successive temporal blocks are descriptive checks, not independent network
replicates. These are extra trajectories on four connectivity realizations
already counted in the primary campaign, not four additional network draws.}
\label{tab:tangent_checks}
\begin{tabular}{rrrrrr}
\toprule
Control&$N$&Seed&Largest exponent&Minimum block&Maximum block\\
\midrule
'''
    (HERE/'tangent_checks.tex').write_text(pre+'\n'.join(tangents)+r'''
\bottomrule
\end{tabular}
\end{table}
''')
    outlier=[]
    original=next(record for record in summary['time_step_checks']
                  if record['case']=='composition_f80_N1000_seed3001')
    v=original['baseline']
    outlier.append(f'SciPy&0.05&8000&{v["nu"]:.6f}&{v["energy"]:.6f}&{v["d"]:.6f}'+r'\\')
    for record in summary['outlier_diagnostics']:
        assert record['same_connectivity_SHA256']
        for v in record['prefixes']:
            outlier.append(f'Native&{record["dt"]:g}&{v["duration"]:g}&{v["nu"]:.6f}&{v["energy"]:.6f}&{v["d"]:.6f}'+r'\\')
    pre=r'''\begin{table}[htp]
\centering
\small
\caption{Bounded follow-up of the retained 80:20, $N=1000$, seed3001
outlier. All rows use the same connectivity, initial condition, drive, and
mean-difference estimator. The native backend is algebraically equivalent
to the SciPy action but can change floating-point phase after long chaotic
evolution. Each native run contributes several prefixes, not independent
replicates. The original primary outcome remains unchanged.}
\label{tab:outlier_followup}
\begin{tabular}{rrrrrr}
\toprule
Backend&Step&Duration&$\nu$&$\energy$&$d$\\
\midrule
'''
    (HERE/'outlier_followup.tex').write_text(pre+'\n'.join(outlier)+r'''
\bottomrule
\end{tabular}
\end{table}
''')
    provenance={'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [summary_path]+tangent_files},
                'time_step_comparisons':len(rows),'tangent_cases':len(tangents),
                'outlier_followup_runs':3,'outlier_followup_table_rows':len(outlier)}
    (HERE/'auxiliary_table_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2))


if __name__=='__main__':
    main()
