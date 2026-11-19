"""Aggregate the prespecified nonuniform degree cohort and draw its figure."""
from pathlib import Path
import json
import sys
from collections import defaultdict
import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUNS=ROOT/"next_round/runtime/runs"

def choice(record,B=500.,duration=8000.,gap=0):
    estimator=next(e for e in record["estimators"] if e["block_duration"]==B)
    prefix=next(p for p in estimator["prefixes"] if p["duration"]==duration)
    return estimator,next(g for g in prefix["gaps"] if g["empty_blocks"]==gap)

def summary(values):
    x=np.asarray(values,dtype=float)
    n=len(x)
    m=x.mean(axis=0)
    sd=x.std(axis=0,ddof=1) if n>1 else np.full(m.shape,np.nan)
    se=sd/np.sqrt(n)
    def native(v):
        a=np.asarray(v)
        if a.ndim==0:return float(a) if np.isfinite(a) else None
        return a.tolist()
    interval=student_t.ppf(.975,n-1)*se if n>1 else np.full(m.shape,np.nan)
    return {"count":n,"mean":native(m),"sd":native(sd),"se":native(se),
            "t95_low":native(m-interval),"t95_high":native(m+interval)}

def aggregate():
    protocol=json.loads((HERE/"adopted_protocol.json").read_text())
    expected=[j["name"] for j in protocol["jobs"]]
    networks=[];missing=[];groups=defaultdict(list)
    for name in expected:
        path=RUNS/(name+".json")
        if not path.exists():
            missing.append(name);continue
        record=json.loads(path.read_text())
        if record["status"]!="complete":
            missing.append(name);continue
        cfg=record["job"]["config"]
        est,main=choice(record)
        _,gap1=choice(record,gap=1)
        _,half=choice(record,duration=4000.)
        longblock=choice(record,B=1000.)[1] if any(e['block_duration']==1000. for e in record['estimators']) else None
        local=est["population_local_stats"]
        row={"name":name,"N":cfg["n"],"chi":cfg["degree_dispersion"],
             "seed":cfg["seed"],"status":record["status"],"primary":main,
             "gap1":gap1,"T4000":half,"B1000":longblock,
             "local_statistics":local,"network_info":record["network_info"],
             "raw_estimator_scope":"Population local statistics use the full T8000 record. Raw input-mean variance includes finite-time sampling noise. Primary energy uses mean-free four-window estimator; its ratio retains finite-record bias.",
             "state_description":record.get("state_description")}
        np.testing.assert_allclose(np.sum(main["population_energy"]),main["energy"],rtol=1e-12,atol=1e-13)
        networks.append(row);groups[(row["N"],row["chi"])].append(row)
    theory=json.loads((HERE/"predictions.json").read_text())
    grouped=[]
    for (N,chi),rows in sorted(groups.items()):
        prediction=next(p for p in theory["rows"] if p["chi"]==chi)
        r={"N":N,"chi":chi,"seeds":[x["seed"] for x in rows],
           "theory":prediction,"metrics":{}}
        fields={"d":[x["primary"]["d"] for x in rows],
                "nu":[x["primary"]["nu"] for x in rows],
                "energy":[x["primary"]["energy"] for x in rows],
                "block_energy":[x["primary"]["population_energy"] for x in rows],
                "Delta_raw_input_means":[[p["Delta_input_means"] for p in x["local_statistics"]] for x in rows],
                "q_temporal_input":[[p["q_temporal_input"] for p in x["local_statistics"]] for x in rows],
                "A_timeaveraged_gain_squared":[[p["A_timeaveraged_gain_squared"] for p in x["local_statistics"]] for x in rows],
                "rate_population_means":[[p["mean_rate"] for p in x["local_statistics"]] for x in rows],
                "gap1_minus_primary_d":[x["gap1"]["d"]-x["primary"]["d"] for x in rows],
                "T4000_minus_primary_d":[x["T4000"]["d"]-x["primary"]["d"] for x in rows],
                "B1000_minus_primary_d":[x["B1000"]["d"]-x["primary"]["d"] for x in rows if x["B1000"] is not None]}
        for key,value in fields.items():
            if value:r["metrics"][key]=summary(value)
        r["ratio_of_mean_moments"]=r["metrics"]["nu"]["mean"]**2/r["metrics"]["energy"]["mean"]
        r["mean_of_network_ratios"]=r["metrics"]["d"]["mean"]
        r["relative_PR_error"]=r["metrics"]["d"]["mean"]/prediction["prediction"]["dimension_fraction"]-1
        grouped.append(r)
    result={"complete_cohort":not missing,"expected_networks":len(expected),
            "complete_networks":len(networks),"missing":missing,
            "primary_protocol":{"T":8000.,"B":500.,"gap":0},
            "uncertainty_unit":"Independent connectivity realization; error bars are network SE, not temporal/pair/neuron pseudoreplicates.",
            "scope":"Finite-size nonuniform EI degree test; deterministic predictions are conditional closure, never fitted to measured PR.",
            "groups":grouped,"networks":networks}
    references=[]
    for N,seeds in [(2000,range(3001,3013)),(4000,range(3001,3005))]:
        rr=[];names=[];refmissing=[]
        for seed in seeds:
            name=f'composition_f80_N{N}_seed{seed}';path=RUNS/(name+'.json')
            if not path.exists():refmissing.append(name);continue
            record=json.loads(path.read_text())
            if record['status']!='complete':refmissing.append(name);continue
            cfg=record['job']['config']
            np.testing.assert_allclose(cfg['variance_profile'],theory['G'],rtol=1e-14,atol=1e-13)
            np.testing.assert_allclose(cfg['mean_connectivity'],theory['J0'],rtol=1e-14,atol=1e-13)
            np.testing.assert_allclose(record['job']['mean_input'],theory['rows'][2]['mean_inputs'],rtol=1e-14,atol=1e-13)
            assert cfg['ensemble']=='bernoulli' and cfg['fraction_e']==.8 and cfg['mean_rate']==.3
            assert cfg['dt']==.05 and cfg['sample_dt']==.25 and cfg['burn']==500. and cfg['duration']==8000.
            est,measurement=choice(record)
            rr.append({'measurement':measurement,'local':est['population_local_stats']});names.append(name)
        if not rr:continue
        metrics={'d':summary([x['measurement']['d'] for x in rr]),
                 'block_energy':summary([x['measurement']['population_energy'] for x in rr]),
                 'Delta_raw_input_means':summary([[p['Delta_input_means'] for p in x['local']] for x in rr]),
                 'q_temporal_input':summary([[p['q_temporal_input'] for p in x['local']] for x in rr])}
        references.append({'N':N,'chi':1.,'names':names,'missing':refmissing,'metrics':metrics,
                           'role':'Distinct composition-campaign same-law reference, not a degree replicate or pooled primary cohort.'})
    result['same_law_composition_references']=references
    result['reference_scope']='The chi1 degree construction is exactly iidBernoulli. Hollow composition-cohort references use the identical model and primary estimator, but are not included in the degree cohort count.'
    (HERE/"comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    return result

def make_figure(result, dense=None):
    sys.path.insert(0, str(HERE.parent / "figures"))
    from paper_style import (CROSS, E, I, JOINT, MUTED, SLATE, SIZE_MARKERS,
                             apply_style, panel_label, save_figure)
    from matplotlib.lines import Line2D
    from sweep_records import cohort_handles, dense_records, draw_group, grouped_networks, point_offsets

    apply_style()
    if dense is None:
        dense = dense_records()
    original = [{"config": {"n": row["N"], "degree_dispersion": row["chi"]},
                 "job": {"experiment": "degree"}, "primary": row["primary"],
                 "local": row["local_statistics"]} for row in result["networks"]]
    groups = grouped_networks(original, [r for r in dense if r["job"]["experiment"] == "degree"])
    references = result.get("same_law_composition_references", [])
    curve = json.loads((HERE / "theory_curve.json").read_text())["rows"]
    chi = np.array([r["chi"] for r in curve])
    # The square-root coordinate resolves the five settings between 0 and 0.2.
    # Offsets are display offsets only; theory and data use nominal chi.
    xcurve = np.sqrt(chi)
    plotted = {(np.sqrt(x), n, cohort): rows for (x, n, cohort), rows in groups.items()}
    displayed = dict(plotted)
    for ref in references:
        displayed[1., ref["N"], "reference"] = []
    offsets = point_offsets(displayed, .026)
    fig, axes = plt.subplots(2, 2, figsize=(7.7, 5.6), layout="constrained")

    def observations(ax, quantity, field, component=(), factor=1., color=JOINT):
        for key, rows in plotted.items():
            x, n, cohort = key
            if quantity == "local":
                values = [r["local"][component[0]][field] for r in rows]
            else:
                values = [np.asarray(r["primary"][field])[component] for r in rows]
            draw_group(ax, x+offsets[key], factor*np.asarray(values), SIZE_MARKERS[n],
                       color, cohort, jitter=.004, show_networks=field=="d")
        # The Bernoulli composition cohorts supply same-law reference points at chi=1.
        for ref in references:
            n = ref["N"]
            ref_field = {"d":"d", "Delta_input_means":"Delta_raw_input_means",
                         "q_temporal_input":"q_temporal_input", "population_energy":"block_energy"}[field]
            metric = ref["metrics"][ref_field]
            ax.errorbar(1 + offsets[1.,n,"reference"], factor*np.asarray(metric["mean"])[component],
                        yerr=factor*np.asarray(metric["se"])[component], fmt=SIZE_MARKERS[n],
                        color=color, mfc="white", mew=.85, capsize=1.7, ms=4.6, zorder=5)

    a, b, c, d = axes.flat
    a.plot(xcurve, [r["prediction"]["dimension_fraction"] for r in curve], color=JOINT)
    observations(a, "primary", "d")
    a.set(yscale="log", ylabel=r"Activity dimension $D/N$")
    a.set_yticks([.01,.02,.05,.10,.20], ["0.01","0.02","0.05","0.10","0.20"])
    a.minorticks_off()
    for ax, theory_key, measured_key, ylabel in [
        (b, "static_variances", "Delta_input_means", r"Input mean variance $\widehat{\Delta}_a$"),
        (c, "dynamic_variances", "q_temporal_input", r"Temporal input variance $q_a$"),
    ]:
        for pop, label, ink, ls in [(0, "E", E, "-"), (1, "I", I, "--")]:
            ax.plot(xcurve, [r[theory_key][pop] for r in curve], color=ink, ls=ls, label=label)
            observations(ax, "local", measured_key, (pop,), color=ink)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=-.04*ax.get_ylim()[1])
    b.legend(loc="upper left", ncol=2)
    for row, col, factor, ink, ls, name in [(0,0,1,E,"-","E–E"), (0,1,2,CROSS,"--","2 × E–I"),
                                            (1,1,1,I,":","I–I")]:
        d.plot(xcurve, [factor*r["block_energies"][row][col] for r in curve], color=ink, ls=ls, label=name)
        observations(d, "primary", "population_energy", (row,col), factor, ink)
    d.set_ylabel("Block energy per neuron")
    d.legend(loc="upper right", labelspacing=.25)

    ticks = [0, .05, .2, .5, 1, 1.5, 2]
    for ax, letter in zip(axes.flat, "ABCD"):
        ax.set(xlabel=r"Degree dispersion $\chi$ (square-root scale)", xlim=(-.065,np.sqrt(2)+.065))
        ax.set_xticks(np.sqrt(ticks), [f"{value:g}" for value in ticks])
        panel_label(ax, letter)
    handles = [Line2D([], [], color=JOINT, label="Theory")]
    handles += cohort_handles({n for _, n, _ in groups}, has_dense=bool(dense), reference=bool(references))
    fig.legend(handles=handles, loc="outside lower center", ncol=4,
               handlelength=1.5, columnspacing=1.15, labelspacing=.25)
    save_figure(fig, HERE / "degree_validation")

    fig, ax = plt.subplots(figsize=(4.6,3.2),layout="constrained")
    ax.plot(chi,[r["decomposition"]["uncorrelated_dimension_fraction"] for r in curve],
            label="Variance-distribution factor",color=MUTED)
    ax.plot(chi,[r["decomposition"]["correlation_reduction_factor"] for r in curve],
            label="Correlation factor",color=SLATE,ls="--")
    ax.plot(chi,[r["prediction"]["dimension_fraction"] for r in curve],
            label="Product, PR/N",color=JOINT)
    ax.set(xlabel=r"Degree dispersion $\chi$",ylabel="Dimension factors",yscale="log")
    ax.legend()
    save_figure(fig, HERE / "degree_mechanism_decomposition")


def main():
    result=aggregate()
    print(json.dumps({"complete_cohort":result["complete_cohort"],
                      "complete_networks":result["complete_networks"],"expected_networks":result["expected_networks"],
                      "groups":[{"N":r["N"],"chi":r["chi"],"d":r["metrics"]["d"],
                                 "relative_PR_error":r["relative_PR_error"]} for r in result["groups"]]},indent=2))
    if result["complete_cohort"]:make_figure(result)
    else:print("Figure withheld until every prespecified degree result is complete.")

if __name__=="__main__":main()
