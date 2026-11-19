"""Selected retained-trajectory checks of degree local moments and PR factors.

Only protocol-retained endpoint seed4001 records are analyzed; these are
diagnostics, not additional independently sampled networks.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
import json,itertools
import numpy as np
from scipy.special import ndtr

HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parent/"runtime"
OUT=HERE/"ingredient_diagnostics"
OUT.mkdir(exist_ok=True)

def calculate(name):
    record=json.loads((RUNTIME/"runs"/(name+".json")).read_text())
    if record["status"]!="complete":return {"name":name,"status":"not_complete"}
    path=RUNTIME/"tmp"/(name+"_trajectory.npy")
    capsule=HERE.parent/'analytics/local_block_capsules'/(name+'.npz')
    v2capsule=RUNTIME/'runs'/(name+'_B500_neuron_statistics.npz')
    N=record['job']['config']['n'];dt=record["job"]["config"]["sample_dt"];ns=round(500/dt)
    if v2capsule.exists():
        with np.load(v2capsule) as z:
            mx=z['block_mean_x'];sx=z['block_q_x']+mx*mx
            mr=z['block_mean_rate'];sr=z['block_variance_rate']+mr*mr;ag=z['block_mean_gain']
        nb=len(mx);moment_source=str(v2capsule.relative_to(HERE.parents[1]))
    elif capsule.exists():
        with np.load(capsule) as z:
            mx=z['mean_x'];sx=z['second_x'];mr=z['mean_rate'];sr=z['second_rate'];ag=z['mean_gain']
        nb=len(mx);moment_source=str(capsule.relative_to(HERE.parents[1]))
    elif path.exists():
        x=np.load(path,mmap_mode='r');nb=x.shape[0]//ns
        mx=np.empty((nb,N));sx=np.empty((nb,N));mr=np.empty((nb,N));sr=np.empty((nb,N));ag=np.empty((nb,N))
        for b in range(nb):
            block=np.asarray(x[b*ns:(b+1)*ns],dtype=float)
            rate=2*ndtr(block)
            mx[b]=block.mean(0);sx[b]=np.square(block).mean(0)
            mr[b]=rate.mean(0);sr[b]=np.square(rate).mean(0)
            ag[b]=(np.sqrt(2/np.pi)*np.exp(-.5*block*block)).mean(0)
        moment_source='retained float32 input trajectory, recomputed float64 block moments'
    else:return {"name":name,"status":"raw_and_capsule_not_retained"}
    if nb*500 != record['job']['config']['duration']:
        return {'name':name,'status':'capsule_not_full_duration','blocks':nb}
    Ne=round(N*record["job"]["config"]["fraction_e"])
    groups=[slice(0,Ne),slice(Ne,N)];Na=np.array([Ne,N-Ne]);f=Na/N
    pairs=list(itertools.combinations(range(nb),2));pp=np.array(pairs);index={p:i for i,p in enumerate(pairs)}
    vr=.5*(sr[pp[:,0]]+sr[pp[:,1]])-mr[pp[:,0]]*mr[pp[:,1]]
    vx=.5*(sx[pp[:,0]]+sx[pp[:,1]])-mx[pp[:,0]]*mx[pp[:,1]]
    energies=[vr[:,g]@vr[:,g].T/Na[a] for a,g in enumerate(groups)]
    qcentered=[vx[:,g]-vx[:,g].mean(1,keepdims=True) for g in groups]
    qspread=[v@v.T/Na[a] for a,v in enumerate(qcentered)]
    source=next(e for e in record["estimators"] if e["block_duration"]==500)
    primary=next(p for p in source["prefixes"] if p["duration"]==8000)
    rows=[]
    for gap in [0,1,2]:
        valid=[i for i,(b,c) in enumerate(pairs) if c-b>=gap+1]
        delta=[];a2=[];q=[];nu=[];rate_static=[];mean_rate_squared=[]
        for a,g in enumerate(groups):
            bx=mx[:,g];centered=bx-bx.mean(1,keepdims=True)
            deltapair=np.mean(centered[pp[:,0]]*centered[pp[:,1]],axis=1)
            gainpair=np.mean(ag[pp[:,0],g]*ag[pp[:,1],g],axis=1)
            delta.append(float(deltapair[valid].mean()));a2.append(float(gainpair[valid].mean()))
            q.append(float(vx[valid,g].mean()));nu.append(float(vr[valid,g].mean()))
            br=mr[:,g];brcentered=br-br.mean(1,keepdims=True)
            rate_static.append(float(np.mean(np.mean(brcentered[pp[:,0]]*brcentered[pp[:,1]],axis=1)[valid])))
            mean_rate_squared.append(float(np.mean((br.mean(1)[pp[:,0]]*br.mean(1)[pp[:,1]])[valid])))
        selected=list(itertools.combinations(range(nb),4))
        selected=[p for p in selected if min(np.diff(p))>=gap+1]
        diag=[];qvar=[];mu_q_cov=[]
        for a in range(2):
            g=groups[a]
            values=[];qvalues=[]
            for aa,bb,cc,dd in selected:
                for p1,p2 in [((aa,bb),(cc,dd)),((aa,cc),(bb,dd)),((aa,dd),(bb,cc))]:
                    values.append(energies[a][index[p1],index[p2]])
                    qvalues.append(qspread[a][index[p1],index[p2]])
            diag.append(float(np.mean(values)))
            qvar.append(float(np.mean(qvalues)))
            triple=[]
            bx=mx[:,g]-mx[:,g].mean(1,keepdims=True)
            for aa,bb,cc in itertools.combinations(range(nb),3):
                if min(np.diff([aa,bb,cc]))<gap+1:continue
                for single,other in [(aa,(bb,cc)),(bb,(aa,cc)),(cc,(aa,bb))]:
                    triple.append(np.mean(bx[single]*qcentered[a][index[other]]))
            mu_q_cov.append(float(np.mean(triple)))
        G=np.array(record['job']['config']['variance_profile'])
        chi_eff=np.array(record['network_info']['degree_expected_dispersion'])
        static_source=np.sum(G*f[None,:]*((Na[None,:]-chi_eff)/(Na[None,:]-1)*np.array(rate_static)[None,:]+chi_eff*np.array(mean_rate_squared)[None,:]),axis=1)
        row={"empty_blocks":gap,"Delta_crossblock_input_means":delta,
             "A_crossblock_timeaveraged_gain_squared":a2,
             "q_crossblock_temporal_input":q,"nu_crossblock_temporal_rate":nu,
             "v_squared_fourwindow_by_population":diag,
             "q_variance_across_neurons_fourwindow":qvar,
             "input_mean_temporal_q_covariance_threewindow":mu_q_cov,
             "static_rate_mean_variance_crossblock":rate_static,
             "population_mean_rate_squared_crossblock":mean_rate_squared,
             "static_row_contraction_measured_source_diagnostic":static_source.tolist(),
             "diagonal_rate_energy":float(f@diag),"quadruples":len(selected)}
        measured=next((p for p in primary["gaps"] if p["empty_blocks"]==gap),None)
        if measured is not None:
            nutotal=float(f@nu)
            np.testing.assert_allclose(nutotal,measured["nu"],rtol=1e-9,atol=1e-10)
            row.update(energy=measured["energy"],d=measured["d"],
                       d_uncorrelated=nutotal**2/row["diagonal_rate_energy"],
                       correlation_factor=row["diagonal_rate_energy"]/measured["energy"],
                       factor_identity_error=(nutotal**2/row["diagonal_rate_energy"])*(row["diagonal_rate_energy"]/measured["energy"])-measured["d"])
        rows.append(row)
    result={"name":name,"status":"complete","N":N,"chi":record["job"]["config"]["degree_dispersion"],
            "scope":"Available retained-trajectory or captured-local-moment diagnostic; coverage is reported in status.json, not assumed to be the full cohort. Crossblock identities require stationary independent windows. Negative corrected finite-time values are not clipped. It does not create another network replicate.",
            "moment_source":moment_source,
            "moment_blocks":nb,
            "B":500.,"T":8000.,"rows":rows,
            "raw_full_record_population_local_stats":source["population_local_stats"]}
    (OUT/(name+".json")).write_text(json.dumps(result,indent=2)+"\n")
    return result

def main():
    jobs=json.loads((RUNTIME/"campaign_jobs.json").read_text())
    extension=RUNTIME/'degree_extension_jobs.json'
    if extension.exists():jobs+=json.loads(extension.read_text())
    selected=[j["name"] for j in jobs if j["experiment"]=="degree"]
    records=[]
    for name in selected:
        saved=OUT/(name+".json")
        if saved.exists():r=json.loads(saved.read_text())
        else:
            path=RUNTIME/"runs"/(name+".json")
            r=calculate(name) if path.exists() else {"name":name,"status":"not_started"}
        records.append(r)
        print(name,r["status"],flush=True)
    (OUT/"status.json").write_text(json.dumps({"selected":selected,"records":records},indent=2)+"\n")

if __name__=="__main__":main()
