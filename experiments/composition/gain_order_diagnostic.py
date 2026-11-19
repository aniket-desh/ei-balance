"""Check retained formal gain coefficients without fitting a replacement."""
from pathlib import Path
import json,math
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    d=json.loads((HERE/'prospective_summary.json').read_text())
    c=json.loads((ROOT/'archive/integration/corrected_coefficients.json').read_text())['results'];rows=[]
    for key,g in d['groups'].items():
        fe=float(key.split('_')[0][1:]);N=int(key.split('_N')[1]);a0=c[str(fe)]['A0'];a1=c[str(fe)]['A1']
        for pop in [0,1]:
            z=g['local'][str(pop)]['A_timeaveraged_gain_squared'];mean=z['mean'];pred=a0[pop]+a1[pop]/math.sqrt(N);ci=z['ci95_t'];r=N*(mean-pred);rci=[N*(x-pred)for x in ci]if ci else None
            rows.append({'fraction_e':fe,'N':N,'population':pop,'n_networks':z['n'],'mean_A':mean,'leading_A0':a0[pop],'formal_A1':a1[pop],'formal_first_A':pred,'sqrtN_scaled_total_shift':math.sqrt(N)*(mean-a0[pop]),'N_scaled_residual_after_formal_first':r,'N_scaled_residual_ci95_t':rci,'scope':'Coefficient diagnostic, not a fit. Constancy is compatible with an N^-1 remainder but does not prove it. No second-order coefficient is adopted.'})
    (HERE/'gain_order_diagnostic.json').write_text(json.dumps({'rows':rows,'status':'Existing formal coefficients checked against new cohorts; no fitted repair.'},indent=2)+'\n')


if __name__=='__main__':main()
