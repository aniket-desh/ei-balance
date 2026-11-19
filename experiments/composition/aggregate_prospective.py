"""Aggregate prespecified prospective composition networks, with all statuses.

Network realizations, not neurons or temporal quadruples, define uncertainty.
B500,T8000,gap0 is the campaign primary; alternate gaps/durations are paired
sensitivity checks. Small-n t intervals are descriptive and not equivalence
proofs. Cross-seed covariances are exploratory at small sample sizes.
"""
from pathlib import Path
import json
import numpy as np
from scipy.stats import t

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
REF=json.loads((HERE/'composition_theory_targets.json').read_text())['one_site_reference']


def stats(x):
    x=np.asarray(x,float);n=len(x);mean=x.mean(0)
    if n>1:
        sd=x.std(0,ddof=1);se=sd/np.sqrt(n);half=t.ppf(.975,n-1)*se
        ci=np.stack([mean-half,mean+half],axis=-1).tolist()
        sd=sd.tolist();se=se.tolist()
    else:sd=se=ci=None
    return {'n':n,'mean':mean.tolist(),'sd':sd,'se':se,'ci95_t':ci}


def pick(d,B=500.,duration=8000.,gap=0):
    for e in d.get('estimators',[]):
        if e['block_duration']!=B:continue
        for pre in e['prefixes']:
            if pre['duration']!=duration:continue
            for g in pre['gaps']:
                if g['empty_blocks']==gap:return g,e
    return None,None


def target(fe):
    x=REF['predictions'][str(fe)];f=np.array([fe,1-fe]);v2=np.array(x['variance_seconds']);X=np.array(x['pair_covariance_energy'])+np.diag(v2/f)
    return dict(nu=float(f@x['mean_variances']),energy=x['covariance_energy'],d=x['dimension_fraction'],population_energy=(f[:,None]*X*f[None,:]).tolist(),Delta=REF['static_variances'],q=REF['dynamic_variances'],A=REF['alpha_second'],h=REF['mean_inputs'],within_population_dimension_fractions=x['within_population_dimension_fractions'])


def build():
    jobs=json.loads((ROOT/'next_round/runtime/campaign_jobs.json').read_text())
    if isinstance(jobs,dict):jobs=jobs.get('jobs',jobs.get('campaign',[]))
    extension=ROOT/'next_round/runtime/extension_jobs.json'
    if extension.exists():
        more=json.loads(extension.read_text()); jobs.extend(more if isinstance(more,list)else more['jobs'])
    expected=list({j['name']:j for j in jobs if j['experiment']=='composition'}.values())
    statuses=[];rows=[]
    for j in expected:
        p=ROOT/'next_round/runtime/runs'/f"{j['name']}.json"
        d=json.loads(p.read_text())if p.exists()else{'status':'not_started'}
        statuses.append({'name':j['name'],'status':d['status']})
        if d['status']!='complete':continue
        primary,e=pick(d)
        if primary is None:continue
        assert abs(np.array(primary['population_energy']).sum()-primary['energy'])<1e-11
        c=j['config'];local=e['population_local_stats']
        f=np.array([c['fraction_e'],1-c['fraction_e']]); vp=np.array(primary['nu_by_population']); eb=np.array(primary['population_energy']); dp=f*vp*vp/np.diag(eb); primary['within_population_dimension_fractions']=dp.tolist();primary['within_population_absolute_dimensions']=(c['n']*f*dp).tolist();primary['whole_network_absolute_dimension']=c['n']*primary['d'];primary['cross_population_temporal_Gram_overlap_diagnostic']=float(eb[0,1]/np.sqrt(eb[0,0]*eb[1,1]))
        row={'name':j['name'],'N':c['n'],'fraction_e':c['fraction_e'],'seed':c['seed'],'primary':primary,'local':local,'sensitivity':[],'source':str(p.relative_to(ROOT))}
        for B,T,gap in [(500.,4000.,0),(500.,8000.,1),(1000.,8000.,0),(1000.,8000.,1)]:
            val,_=pick(d,B,T,gap)
            if val:
                row['sensitivity'].append({'B':B,'T':T,'empty_blocks':gap,'d':val['d'],'energy':val['energy'],'nu':val['nu'],'relative_d_minus_primary':val['d']/primary['d']-1})
        rows.append(row)
    out={'scope':'Prospective networks; primaryB500,T8000,gap0. Windowindependence notproved. Smallnintervalsare not equivalence tests. Everyprespecified statusretained.','expected_count':len(expected),'completed_primary_count':len(rows),'statuses':statuses,'primary_fields':['nu','energy','d'],'groups':{},'paired_composition':{},'paired_size':{},'N2000_stage_subgroups':{},'per_network':rows}
    for N,fe in sorted({(r['N'],r['fraction_e'])for r in rows}):
        rs=[r for r in rows if r['N']==N and r['fraction_e']==fe];tar=target(fe)
        X=np.array([[r['primary'][k]for k in ['nu','energy','d']]for r in rs]);S=stats(X);m=X.mean(0);plugin=m[0]**2/m[1];a=m[0]/m[1];identity=np.mean((X[:,0]-a*X[:,1])**2/X[:,1])
        if S['ci95_t'] is not None:S['relative_error_ci95_t_nu_energy_d']=(np.array(S['ci95_t'])/np.array([tar['nu'],tar['energy'],tar['d']])[:,None]-1).tolist()
        S.update({'seeds':[r['seed']for r in rs],'parameter_only':tar,'relative_mean_errors_nu_energy_d':(m/np.array([tar['nu'],tar['energy'],tar['d']])-1).tolist(),'plugin_ratio_of_means':float(plugin),'mean_ratio_minus_ratio_of_means':float(m[2]-plugin),'positive_ratio_identity':float(identity),'population_energy':stats(np.array([r['primary']['population_energy']for r in rs])),'coherent_projection_fraction':stats([r['primary']['coherent_projection_fraction']for r in rs]),'local':{},'Delta_q_countermotion':{}})
        Eblocks=np.array([r['primary']['population_energy']for r in rs]); shape=Eblocks/X[:,1,None,None]; shape0=np.array(tar['population_energy'])/tar['energy']
        S['normalized_population_geometry']={'mean_and_uncertainty':stats(shape),'parameter_only':shape0.tolist(),'ratio_of_mean_energies':(Eblocks.mean(0)/X[:,1].mean()).tolist(),'ratio_of_mean_energies_relative_shape_error':float(np.linalg.norm(Eblocks.mean(0)/X[:,1].mean()-shape0)/np.linalg.norm(shape0)),'per_network_relative_frobenius_shape_error':stats([np.linalg.norm(z-shape0)/np.linalg.norm(shape0)for z in shape]),'ensemble_mean_shape_relative_error':float(np.linalg.norm(shape.mean(0)-shape0)/np.linalg.norm(shape0)),'scope':'Normalizedenergyallocationisaseparateobservable,nofitofPRamplitudes. Rawblockenergiesandtotalerrorsremainreported.'}
        S['within_population_dimension_fractions']=stats([r['primary']['within_population_dimension_fractions']for r in rs]);S['within_population_absolute_dimensions']=stats([r['primary']['within_population_absolute_dimensions']for r in rs]);S['whole_network_absolute_dimension']=stats([r['primary']['whole_network_absolute_dimension']for r in rs]);S['leading_within_population_absolute_dimensions']=(N*np.array([fe,1-fe])*np.array(tar['within_population_dimension_fractions'])).tolist();S['cross_population_temporal_Gram_overlap_diagnostic']=stats([r['primary']['cross_population_temporal_Gram_overlap_diagnostic']for r in rs]);eb0=np.array(tar['population_energy']);S['leading_cross_population_temporal_Gram_overlap']=float(eb0[0,1]/np.sqrt(eb0[0,0]*eb0[1,1]))
        if len(rs)>1:S['cov_nu_energy_d']=np.cov(X,rowvar=False,ddof=1).tolist()
        for pop in [0,1]:
            keys=[k for k,v in rs[0]['local'][pop].items()if isinstance(v,(float,int))and k not in ['population','count']]
            S['local'][str(pop)]={k:stats([r['local'][pop][k]for r in rs])for k in keys}
            D=np.array([r['local'][pop]['Delta_input_means']for r in rs]);q=np.array([r['local'][pop]['q_temporal_input']for r in rs]);V=D+q
            counter={'Delta':stats(D),'q':stats(q),'total_Delta_plus_q':stats(V),'parameter_only_total':tar['Delta'][pop]+tar['q'][pop]}
            if len(rs)>1:
                C=np.cov(np.stack([D,q],axis=1),rowvar=False,ddof=1);counter.update(covariance=C.tolist(),corr=float(C[0,1]/np.sqrt(C[0,0]*C[1,1])),total_variance_cancellation_fraction=float(-2*C[0,1]/(C[0,0]+C[1,1])))
            S['Delta_q_countermotion'][str(pop)]=counter
        out['groups'][f'f{fe}_N{N}']=S
    for fe in [.6,.8]:
        for stage,selected in [('initial_four',set(range(3001,3005))),('extension_eight',set(range(3005,3013)))]:
            sub=[r for r in rows if r['N']==2000 and r['fraction_e']==fe and r['seed']in selected]
            if sub:
                val=stats([[r['primary'][k]for k in ['nu','energy','d']]for r in sub]);val['seeds']=[r['seed']for r in sub];out['N2000_stage_subgroups'][f'f{fe}_{stage}']=val
    for N in sorted({r['N']for r in rows}):
        aa={r['seed']:r for r in rows if r['N']==N and r['fraction_e']==.6};bb={r['seed']:r for r in rows if r['N']==N and r['fraction_e']==.8};seeds=sorted(set(aa)&set(bb)&{3001,3002,3003,3004})
        if seeds:
            ratio=[aa[s]['primary']['d']/bb[s]['primary']['d']for s in seeds];diff=[aa[s]['primary']['d']-bb[s]['primary']['d']for s in seeds]
            out['paired_composition'][str(N)]={'seeds':seeds,'d60_minus_d80':stats(diff),'d60_over_d80':stats(ratio),'leading_ratio':target(.6)['d']/target(.8)['d']}
    for fe in [.6,.8]:
        sizes=sorted({r['N']for r in rows if r['fraction_e']==fe})
        for n0,n1 in zip(sizes[:-1],sizes[1:]):
            aa={r['seed']:r for r in rows if r['N']==n0 and r['fraction_e']==fe};bb={r['seed']:r for r in rows if r['N']==n1 and r['fraction_e']==fe};seeds=sorted(set(aa)&set(bb)&{3001,3002,3003,3004})
            if seeds:
                delta=[[bb[s]['primary'][k]-aa[s]['primary'][k]for k in ['nu','energy','d']]for s in seeds]
                out['paired_size'][f'f{fe}_{n0}_to_{n1}']={'seeds':seeds,'nu_energy_d_difference':stats(delta),'fraction_d_increased':sum(z[2]>0 for z in delta)/len(delta)}
    (HERE/'prospective_summary.writing').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n'); (HERE/'prospective_summary.writing').replace(HERE/'prospective_summary.json')
    print(json.dumps({'completed':len(rows),'expected':len(expected),'groups':{k:{q:v[q]for q in ['n','mean','ci95_t','relative_mean_errors_nu_energy_d']}for k,v in out['groups'].items()}},indent=2))
    return out


if __name__=='__main__':build()
