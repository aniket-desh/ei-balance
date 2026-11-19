"""Compare the blind closures to three held-out networks at each dispersion."""
import json
import tarfile

from particle import (HERE, SRC, DEGREE, N, F, TARGET, SIZES, QUANTILES, G, PROB, baseline,
                      degree_counts, centered_moment, conditional_law, temporal_solve,
                      population_dimension, ParticleDMFT)
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import ndtri
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import hashlib


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sample_law():
    rng = np.random.default_rng(123547)
    M,p,n = 80,.25,30000
    colors = np.full(20,4)
    values = .7 + (np.arange(20)-9.5)/10
    checks = {}
    for chi in (0,1,2):
        K,eff = degree_counts(rng,n,M,p,chi)
        sums = np.empty(n)
        for k in np.unique(K):
            mask = K == k
            h = rng.multivariate_hypergeometric(colors,int(k),size=mask.sum())
            sums[mask] = h@values
        expected = M*p*(1-p)*((M-eff)/(M-1)*values.var()+eff*values.mean()**2)
        empirical = float(sums.var())
        variance_se = float(np.std((sums-sums.mean())**2)/np.sqrt(n))
        assert abs(empirical-expected)<6*variance_se
        assert abs(sums.mean()-M*p*values.mean())<6*np.sqrt(expected/n)
        if chi == 0:
            assert np.all(K==M*p)
        if chi == 1:
            # Independent inclusion of each physical source gives the same exact second moment.
            np.testing.assert_allclose(expected,M*p*(1-p)*np.mean(values**2))
        checks[chi] = {"draws":n,"M":M,"p":p,"count_range":[int(K.min()),int(K.max())],
                       "empirical_sum_variance":empirical,"exact_sum_variance":expected,
                       "estimated_mc_se_of_variance":variance_se}
    checks["physical_chi0_rounding_dispersion"] = [[degree_counts(rng,1,int(SIZES[b]),PROB[a,b],0)[1]
                                                   for b in range(2)] for a in range(2)]
    return checks


def check_dynamic_reference():
    ref = baseline(0)
    with np.load(DEGREE/"chi0_one_site.npz") as z:
        t = np.arange(0,100.5,.5)
        kernel = np.array([np.interp(t,z["time"],row) for row in z["input_kernels"]])
    sol,q,h = temporal_solve(np.zeros((2,100)),t,np.vstack((kernel,np.gradient(kernel,t,axis=1))),kernel[:,0])
    diff = float(np.max(np.abs(q-ref["dynamic_variances"])))
    assert diff<1e-4
    mu = np.linspace(-5.,2.,100)
    qcheck = 2.1
    h,cov = conditional_law(mu,qcheck,.3)
    lags = np.linspace(0,qcheck,35)
    direct = np.mean([centered_moment(x+h,qcheck,lags) for x in mu],axis=0)
    error = float(np.max(np.abs(direct-cov(lags))))
    assert error<2e-8
    return {"homogeneous_chi0_max_q_error":diff,"empirical_price_table_vs_original_quadrature_max_error":error}


def held_case(chi,seed):
    base = HERE.parent/"dense_sweeps/runs"/f"dense_degree_{chi}_N2000_seed{seed}"
    source = base.with_suffix(".json")
    capsule = base.with_name(base.name+"_B500_neuron_statistics.npz")
    record = json.loads(source.read_text())
    assert record["status"]=="complete" and record["job"]["config"]["n"]==N
    est = next(e for e in record["estimators"] if e["block_duration"]==500)
    prefix = next(e for e in est["prefixes"] if e["duration"]==8000)
    primary = next(r for r in prefix["gaps"] if r["empty_blocks"]==0)
    with np.load(capsule) as z:
        labels,mu = z["labels"],z["mean_x"]
        gain,block_mu,block_q = z["block_mean_gain"],z["block_mean_x"],z["block_q_x"]
    assert np.all(np.bincount(labels)==SIZES) and len(gain)==16
    observed = {"chi":chi,"seed":seed,"files_sha256":{source.name:sha(source),capsule.name:sha(capsule)},
                "mu_quantiles":[],"mean_squared_gain":[],"mu_q_split_correlation":[],
                "conditional_gain_rmse_with_shared_q":[],"conditional_gain_rmse_with_cell_q":[],
                "mean_rate":[r["mean_rate"] for r in est["population_local_stats"]],
                "q":[j["q_mean_free_input"] for j in primary["ingredient_cross_block"]],
                "delta":[j["Delta_cross_block_inputmeans"] for j in primary["ingredient_cross_block"]]}
    for b in range(2):
        ix = labels==b
        observed["mu_quantiles"].append(np.quantile(mu[ix],QUANTILES).tolist())
        g = gain[:,ix]
        a = np.mean((g.sum(0)**2-(g*g).sum(0))/(len(g)*(len(g)-1)))
        observed["mean_squared_gain"].append(float(a))
        # Means and within-cell temporal variances summarized on opposite halves.
        m1,m2 = np.split(block_mu[:,ix],2)
        s1,s2 = np.split((block_q+block_mu**2)[:,ix],2)
        corr = [spearmanr(m1.mean(0),s2.mean(0)-m2.mean(0)**2).statistic,
                spearmanr(m2.mean(0),s1.mean(0)-m1.mean(0)**2).statistic]
        observed["mu_q_split_correlation"].append(np.mean(corr))
        gain1,gain2 = np.split(g,2)
        rmse_shared,rmse_local = [],[]
        # This diagnostic supplies observed means and q from one half only.
        # The independently recorded gain on the other half is the target.
        for m,s,held_gain in ((m1,s1,gain2),(m2,s2,gain1)):
            avg = m.mean(0)
            q = s.mean(0)-avg**2
            target = held_gain.mean(0)
            for noise,errors in ((q.mean(),rmse_shared),(q,rmse_local)):
                prediction = np.sqrt(2/np.pi/(1+noise))*np.exp(-avg**2/(2*(1+noise)))
                errors.append(np.mean((prediction-target)**2))
        observed["conditional_gain_rmse_with_shared_q"].append(float(np.sqrt(np.mean(rmse_shared))))
        observed["conditional_gain_rmse_with_cell_q"].append(float(np.sqrt(np.mean(rmse_local))))
    np.testing.assert_allclose(observed["mean_squared_gain"],
                               [k["A_cross_block_timeaveraged_gain"] for k in primary["ingredient_cross_block"]])
    d = F*np.square(primary["nu_by_population"])/np.diag(primary["population_energy"])
    observed["dimensions"] = [*d,primary["d"]]
    return observed


def aggregate(rows,fields):
    return {key:{label:op(np.asarray([r[key] for r in rows]),axis=0).tolist()
                 for label,op in [("mean",np.mean),("min",np.min),("max",np.max)]} for key in fields}


def check_pr_grid(rows):
    viable = [r for r in rows if r["chi"]==1 and r["particles_per_type"]==400 and r.get("pr_admitted")]
    if not viable:
        return {"status":"no admitted P=400 solution at chi=1"}
    chosen = viable[0]
    with np.load(HERE/"runs"/chosen["artifact"]) as z:
        t = np.arange(1251)*.08
        k = CubicSpline(z["t"],z["input_kernel"],axis=1)(t)
        refined = population_dimension(ParticleDMFT(z["mu"],z["q"],t,k),max_frequency=12.)
    ds = np.array([*refined["within_population_dimension_fractions"],refined["dimension_fraction"]])
    return {"chi":1,"particles":400,"seed":chosen["seed"],"primary_dt":.1,"refined_dt":.08,
            "max_abs_dimension_change":float(np.max(np.abs(ds-chosen["dimensions"])))}


def main():
    cols = ["mu_quantiles","mean_squared_gain","dimensions","q","delta"]
    digest = sha(HERE/"particle.py")
    rows = [json.loads(p.read_text()) for p in sorted((HERE/"runs").glob("*.json"))]
    assert len(rows)==27 and all(r["code_sha256"]==digest for r in rows)
    result = {"fields":cols,"quantile_levels":QUANTILES,"dimension_order":["E","I","joint"],
              "constant_and_bernoulli_limits":check_sample_law(),"kernel_checks":check_dynamic_reference(),
              "pr_fft_check":check_pr_grid(rows),"held_out":[],"by_chi":{}}
    for chi in (0,1,2):
        group = [r for r in rows if r["chi"]==chi]
        observed = [held_case(chi,s) for s in (5101,5102,5103)]
        result["held_out"].extend(observed)
        ref = baseline(chi)
        qgauss = np.sqrt(ref["static_variances"])[:,None]*ndtri(QUANTILES)+np.array(ref["mean_inputs"])[:,None]
        exact = {"mu_quantiles":qgauss.tolist(),"mean_squared_gain":ref["alpha_second"],
                 "dimensions":[*ref["prediction"]["within_population_dimension_fractions"],ref["prediction"]["dimension_fraction"]],
                 "q":ref["dynamic_variances"],"delta":ref["static_variances"]}
        comparison = {"networks":aggregate(observed,cols+["mu_q_split_correlation","mean_rate",
                    "conditional_gain_rmse_with_shared_q","conditional_gain_rmse_with_cell_q"]),"gaussian":exact,
                      "particle":{}}
        for P in (100,200,400):
            chosen = [r for r in group if r["particles_per_type"]==P]
            valid = [r for r in chosen if r["status"]=="converged"]
            accepted = [r for r in valid if r["pr_admitted"]]
            entry = {"seeds":{r["seed"]:r["status"] for r in chosen},
                     "pr_accepted_seeds":[r["seed"] for r in accepted]}
            if valid:
                entry.update(aggregate(valid,[k for k in cols if k!="dimensions"]))
                if accepted:
                    entry.update(aggregate(accepted,["dimensions"]))
            comparison["particle"][P] = entry
        result["by_chi"][chi] = comparison
    result["sources"] = {}
    source_files = [HERE/"particle.py",HERE/"evaluate.py",DEGREE/"predictions.json",
                    *(DEGREE/f"chi{chi}_one_site.npz" for chi in (0,1,2)),
                    *(SRC/"eicavity"/f for f in ("single_site.py","degree.py","multitype.py","pair.py"))]
    with tarfile.open(HERE/"source.tar.gz","w:gz") as bundle:
        for p in source_files:
            name = str(p.relative_to(HERE.parents[1]))
            result["sources"][name] = sha(p)
            bundle.add(p,arcname=name)
    (HERE/"validation.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    with threadpool_limits(1):
        main()
