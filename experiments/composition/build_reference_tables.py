"""Reproduce composition prediction and historical diagnostic tables."""
from pathlib import Path
import json,math

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def write(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2)+'\n')


def main():
    base=json.loads((ROOT/'recovered/bundle/diagnosis/theory/summary.json').read_text())
    coeff=json.loads((ROOT/'archive/integration/corrected_coefficients.json').read_text())['results']
    rows=[]
    for fe in ['0.6','0.8']:
        b=coeff[fe]
        for n in [1000,2000,4000]:
            eps=n**-.5
            rows.append(dict(fraction_e=float(fe),N=n,nu0=b['nu0'],energy0=b['E0'],d0=b['d0'],nu_first=b['nu0']+eps*b['nu1'],energy_first=b['E0']+eps*b['E1'],d_strict_first=b['d0']+eps*b['d1'],d_ratio_of_truncated_moments=(b['nu0']+eps*b['nu1'])**2/(b['E0']+eps*b['E1'])))
    write('composition_theory_targets.json',{'one_site_reference':base,'formal_firstorder_targets':rows,'firstorder_status':'Formal conditional stationary closure; missingeven-orderterms; not controlledfiniteNrepair.'})
    old=json.loads((ROOT/'archive/research_round/quenched/duration_cohort_joint.json').read_text())['conditions']['mean_free_B500_T16000']
    b=coeff['0.6'];nu,E=old['mean_nu_E'];d=old['mean_d']
    res={'nu_ratio':nu/b['nu0'],'energy_ratio':E/b['E0'],'mean_ratio_d_ratio':d/b['d0'],'ratio_of_means_d_ratio':old['plugin_d_of_means']/b['d0'],'exact_log_decomposition_of_ratio_of_means':{'2_log_nu_ratio':2*math.log(nu/b['nu0']),'minus_log_energy_ratio':-math.log(E/b['E0']),'sum_log_d_ratio':math.log(old['plugin_d_of_means']/b['d0'])},'firstorder_N1000_relative_nu':b['nu1']/b['nu0']/math.sqrt(1000),'firstorder_N1000_relative_energy':b['E1']/b['E0']/math.sqrt(1000),'firstorder_N1000_relative_d':b['d1']/b['d0']/math.sqrt(1000),'scope':'Historical16cohort; fixedrecordestimates, not infinite-timeensembletheorem.'}
    write('historical_discrepancy_accounting.json',res)
    rows=[]
    paths=sorted((ROOT/'recovered/bundle/diagnosis/runs').glob('*/hybrid/summary_2048.json'))
    audit=HERE/'historical_hybrid_audit.json'
    expected={ROOT/row['source'] for row in json.loads(audit.read_text())} if audit.exists() else set()
    if not paths or expected.difference(paths):
        print('Skipping the hybrid audit because historical inputs are missing.')
        return
    for p in paths:
        d=json.loads(p.read_text());h=d['heldout_measurement']
        row={'run':p.parent.parent.name,'source':str(p.relative_to(ROOT)),'scope':'Prior short/own-block-centered heldout estimator, not correctedlongrecordnewcampaign; measuredsourcesarecalibrateddiagnostics.','heldout_energy':h['cross_energy'],'heldout_d':h['cross_dimension_fraction'],'cases':{k:{s:v.get(s)for s in ['status','response_radius','dimension_fraction','covariance_energy']}for k,v in d['cases'].items()}}
        for v in row['cases'].values():
            if v['status']=='regular':v['energy_prediction_over_heldout']=v['covariance_energy']/row['heldout_energy']
        rows.append(row)
    write('historical_hybrid_audit.json',rows)


if __name__=='__main__':main()
