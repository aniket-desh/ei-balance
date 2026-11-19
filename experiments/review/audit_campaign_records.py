"""Independent normalization/control-path audit of every completed run record."""
from pathlib import Path
import json,time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def main():
    records=[];errors=[];unfinished=[];bounds=[]
    targetW=np.array([[12.8,1.8],[20.,2.4]])
    targetJ=np.array([[.8,-1.2],[1.6,-2.]])
    degree=json.loads((ROOT/'next_round/degree/predictions.json').read_text())
    degreeh={float(r['chi']):np.array(r['mean_inputs']) for r in degree['rows']}
    maxerr={'energy_sum':0.,'d_ratio':0.,'population_symmetry':0.,'nu_weighting':0.,
            'W_control':0.,'J0F_control':0.,'h_degree':0.,'joint_population_dimension_identity':0.}
    for path in sorted((ROOT/'next_round/runtime/runs').glob('*.json')):
        raw=json.loads(path.read_text());name=raw['name']
        if raw['status']!='complete':
            unfinished.append({'name':name,'status':raw['status']});continue
        job=raw['job'];c=job['config'];f=np.array([c['fraction_e'],1-c['fraction_e']])
        W=np.array(c['variance_profile'])*f
        J=np.array(c['mean_connectivity'])*f
        maxerr['W_control']=max(maxerr['W_control'],float(np.max(np.abs(W-targetW))))
        maxerr['J0F_control']=max(maxerr['J0F_control'],float(np.max(np.abs(J-targetJ))))
        if not np.allclose(W,targetW,rtol=0,atol=1e-13):errors.append(name+':W')
        if not np.allclose(J,targetJ,rtol=0,atol=1e-13):errors.append(name+':J0F')
        if job['experiment']=='degree':
            dh=float(np.max(np.abs(np.array(job['mean_input'])-degreeh[float(c['degree_dispersion'])])))
            maxerr['h_degree']=max(maxerr['h_degree'],dh)
            if dh>1e-12:errors.append(name+':degree_h')
        for estimator in raw['estimators']:
            for prefix in estimator['prefixes']:
                for e in prefix['gaps']:
                    E=np.array(e['population_energy']);energy=e['energy'];nu=e['nu'];d=e['d']
                    err=[abs(E.sum()-energy),abs(nu*nu/energy-d),float(np.max(np.abs(E-E.T))),abs(f@np.array(e['nu_by_population'])-nu)]
                    for key,val in zip(['energy_sum','d_ratio','population_symmetry','nu_weighting'],err):
                        maxerr[key]=max(maxerr[key],val)
                    if max(err)>1e-10:errors.append(name+':normalization')
                    if not np.isfinite(E).all() or not np.isfinite([nu,energy,d]).all():errors.append(name+':nonfinite')
                    if np.all(np.diag(E)>0) and np.all(np.array(e['nu_by_population'])>0):
                        vp=np.array(e['nu_by_population']);N=c['n']
                        Da=N*f*f*vp*vp/np.diag(E)
                        weights=f*vp/nu
                        overlap=E[0,1]/np.sqrt(E[0,0]*E[1,1])
                        inverse=np.sum(weights*weights/Da)+2*overlap*np.prod(weights)/np.sqrt(np.prod(Da))
                        errjoint=abs(1/(N*d)-inverse)
                        maxerr['joint_population_dimension_identity']=max(maxerr['joint_population_dimension_identity'],errjoint)
                        if errjoint>1e-11:errors.append(name+':joint_population_dimension')
                    bound=2*E[0,1]/energy
                    bounds.append({'name':name,'B':estimator['block_duration'],'T':prefix['duration'],
                                   'gap':e['empty_blocks'],'cross_energy_fraction':float(bound),
                                   'coherent_projection_fraction':e.get('coherent_projection_fraction')})
        records.append(name)
    out={'scope':'Algebra/control-path check only; finite-record block estimates need not obey every exact stationary covariance bound.',
         'timestamp_unix':time.time(),'completed_records_checked':len(records),'completed_names':records,
         'unfinished_records':unfinished,'errors':errors,'maximum_absolute_errors':maxerr,
         'measurement_geometry_diagnostics':bounds,'all_completed_record_checks_passed':not errors}
    (Path(__file__).with_suffix('.json')).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['completed_names','measurement_geometry_diagnostics']},indent=2))
    if errors:raise RuntimeError(errors)

if __name__=='__main__':main()
