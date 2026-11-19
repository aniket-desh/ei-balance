"""Freeze v2 from the unchanged first-campaign engine, recording every change."""
from pathlib import Path
p=Path(__file__).with_name('engine.py');s=p.read_text()
s=s.replace("def analyze(traj,labels,sample_dt=.25,B=500.,prefixes=(4000.,8000.),name='network'):","def analyze(traj,labels,sample_dt=.25,B=500.,prefixes=(4000.,8000.),name='network'):")
s=s.replace("    local={'mean_x'", "    xb=np.empty((nb,N));gb=np.empty((nb,N));qxb=np.empty((nb,N))\n    local={'mean_x'")
s=s.replace("        gain=np.sqrt(2/np.pi)*np.exp(-.5*x*x);local['mean_gain']+=gain.sum(0);gain2time+=np.square(gain).sum(0)","        gain=np.sqrt(2/np.pi)*np.exp(-.5*x*x);local['mean_gain']+=gain.sum(0);gain2time+=np.square(gain).sum(0)\n        xb[k]=x.mean(0);qxb[k]=np.square(x).mean(0)-np.square(xb[k]);gb[k]=gain.mean(0)")
s=s.replace("        del C,r,x,gain;discard_map(cs)","        del C,r,x,gain;discard_map(cs);discard_map(traj)")
s=s.replace("                del chunk\n            discard_map(cs)","                del chunk;discard_map(cs)\n            discard_map(cs)")
s=s.replace("    p,q=index[(0,1)],index[(2,3)]\n    # Literal dense covariance check on first four windows validates contractions.\n    dense0=.5*(np.array(cs[0])+np.array(cs[1]))+.5*np.outer(D[p],D[p]);dense1=.5*(np.array(cs[2])+np.array(cs[3]))+.5*np.outer(D[q],D[q])\n    dense=np.array([[np.sum(dense0[np.ix_(ga,gb)]*dense1[np.ix_(ga,gb)])/N for gb in groups] for ga in groups])\n    denseerror=float(np.max(np.abs(pairenergy(p,q)-dense)))\n    np.testing.assert_allclose(pairenergy(p,q),dense,rtol=5e-11,atol=5e-12)\n    del dense0,dense1;discard_map(cs)",'''    p,q=index[(0,1)],index[(2,3)]
    # Full formula was checked at N<=2000 in v1. At larger N independently
    # recheck the same identity on fixed spatial tiles, avoiding four N² copies.
    ix=np.r_[groups[0][:32],groups[1][:32]]
    small=np.array(cs[:4,ix[:,None],ix[None,:]])
    ds0=D[p,ix];ds1=D[q,ix]
    dense0=.5*(small[0]+small[1])+.5*np.outer(ds0,ds0)
    dense1=.5*(small[2]+small[3])+.5*np.outer(ds1,ds1)
    exactsmall=float(np.sum(dense0*dense1)/N)
    compactsmall=.25*(sum(np.sum(small[i]*small[j]) for i in [0,1] for j in [2,3])
       +sum(ds0@small[j]@ds0 for j in [2,3])+sum(ds1@small[i]@ds1 for i in [0,1])+(ds0@ds1)**2)/N
    denseerror=abs(exactsmall-compactsmall)
    np.testing.assert_allclose(exactsmall,compactsmall,rtol=5e-11,atol=5e-12)
    del small,dense0,dense1;discard_map(cs)
    caps=HERE/'runs'/f'{name}_B{int(B)}_neuron_statistics.npz'
    np.savez_compressed(caps,labels=labels,mean_x=local['mean_x'],q_x=qx,
        mean_rate=local['mean_rate'],variance_rate=vm,mean_gain=local['mean_gain'],
        block_mean_x=xb,block_q_x=qxb,block_mean_gain=gb,
        block_mean_rate=means,block_variance_rate=diag)
    diagdiff=.5*(diag[pp[:,0]]+diag[pp[:,1]])+.5*np.square(D)
    diag_gram=[diagdiff[:,g]@diagdiff[:,g].T/N for g in groups]''')
s=s.replace("'dense_formula_error':denseerror,'prefixes':[]}","'dense_formula_error':denseerror,'dense_formula_check_scope':'Fixed32neuronperpopulationtile; fullidentity validatedv1 N<=2000','neuron_statistics_file':str(caps.relative_to(ROOT)),'prefixes':[]}")
s=s.replace("            energies=[];coh=[]", "            energies=[];coh=[];diagonal_energy=[]")
s=s.replace("energies.append(pairenergy(p,q));coh.append(ecoh[p,q])", "energies.append(pairenergy(p,q));coh.append(ecoh[p,q]);diagonal_energy.append([gg[p,q] for gg in diag_gram])")
s=s.replace("            nu=nu_pair[ii].mean(0);nutotal=float(Na@nu/N)",'''            nu=nu_pair[ii].mean(0);nutotal=float(Na@nu/N)
            window_ix=np.triu_indices(k,k=gap+1);ingredient=[]
            for a,g in enumerate(groups):
                xc=xb[:k,g]-xb[:k,g].mean(1)[:,None]
                dstatic=(xc@xc.T/len(g))[window_ix].mean()
                ga=(gb[:k,g]@gb[:k,g].T/len(g))[window_ix].mean()
                input_nupair=.5*(qxb[pp[ii,0]][:,g].mean(1)+qxb[pp[ii,1]][:,g].mean(1))+.5*np.square(xb[pp[ii,0]][:,g]-xb[pp[ii,1]][:,g]).mean(1)
                ingredient.append({'population':a,'Delta_cross_block_inputmeans':float(dstatic),
                    'A_cross_block_timeaveraged_gain':float(ga),'q_mean_free_input':float(input_nupair.mean()),
                    'scope':'Unbiased only for independent stationary windows conditional on fixed connectivity; ensemble label variation remains.'})''')
s=s.replace("'energy':float(total),'nu_by_population'", "'energy':float(total),'diagonal_population_energy':np.mean(diagonal_energy,axis=0).tolist(),'ingredient_cross_block':ingredient,'nu_by_population'")
s=s.replace("action=SparseAction(S,B,sum(labels==0));r['connectivity_sha256']",'''if job.get('native_library') and c.ensemble in ['bernoulli','degree']:
            from native_action import ConstantAction
            action=ConstantAction(S,B,labels,info,Path(job['native_library']));r['action_backend']='compiled_constant_CSR'
        else:
            action=SparseAction(S,B,sum(labels==0));r['action_backend']='SciPy_sparse_complement'
        r['engine_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        r['connectivity_sha256']''')
s=s.replace("HERE/'resource_audit.json'", "HERE/(args.jobs.stem+'_resource_audit.json')")
s=s.replace('gb=np.empty((nb,N))','gblock=np.empty((nb,N))').replace('gb[k]=gain.mean(0)','gblock[k]=gain.mean(0)').replace('block_mean_gain=gb','block_mean_gain=gblock').replace('gb[:k,g]','gblock[:k,g]')
s=s.replace("if job.get('native_library') and c.ensemble in ['bernoulli','degree']:","if job.get('native_library') and Path(job['native_library']).is_file() and c.ensemble in ['bernoulli','degree']:")
s=s.replace("r['connectivity_sha256']=hashlib.sha256(J.tobytes()).hexdigest();del J","r['connectivity_sha256']=hashlib.sha256(J.tobytes()).hexdigest();del J\n        if job.get('native_library') and Path(job['native_library']).is_file():r['native_library_sha256']=hashlib.sha256(Path(job['native_library']).read_bytes()).hexdigest()")
Path(__file__).with_name('engine_v2.py').write_text(s)
