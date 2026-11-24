"""Compute the numbers quoted in the campaign README from the summary tables and arrays.

    python experiments/paper_campaign/report.py   ->   summary.json, summary.md

Uncertainty unit: independently generated networks. Means carry one across-network
standard error; paired composition contrasts carry two-sided 95% Student-t intervals
over matched seeds; size-series entries are across-network medians with bootstrap
standard errors (see plots/common.py).
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "plots")]
import theory
from common import column, fluctuating, groups, load, mean_se, median_se, select


def rel(rows, name):
    return column(rows, name) / rows[0]["th_" + name] - 1


def ms(values):
    m, se, n = mean_se(values)
    return dict(mean=float(m), se=None if not np.isfinite(se) else float(se), n=int(n))


def main():
    rows = load()
    out = {"networks": len(rows), "fixed_point_networks": {}, "settled_during_recording": {}}
    for r in rows:
        if r["fluctuating"] == 0:
            key = f"{r['family']} N={int(r['n'])} chi={r['chi']:g} fE={r['fraction_e']:g} s={r['scale']:g} m={r['rate']:g}"
            bucket = out["fixed_point_networks" if r["state"] == "fixed_point" else "settled_during_recording"]
            bucket[key] = bucket.get(key, 0) + 1
    live = fluctuating(rows)

    # degree sweep
    deg = select(live, family="degree")
    out["degree"] = {}
    for chi, cell in groups(deg, "chi").items():
        out["degree"][f"{chi:g}"] = dict(theory_d=cell[0]["th_d"], d=ms(column(cell, "d")), rel_d=ms(rel(cell, "d")),
                                         rel_nu=ms(rel(cell, "nu")), rel_energy=ms(rel(cell, "energy")),
                                         rel_P_EE=ms(rel(cell, "P_EE")), rel_A_E=ms(rel(cell, "A_E")),
                                         rel_q_E=ms(rel(cell, "q_E")), overlap=ms(column(cell, "overlap")),
                                         theory_overlap=cell[0]["th_overlap"],
                                         lyapunov=ms(column(cell, "lyapunov")))
    d0, d2 = column(select(deg, chi=0.), "d"), column(select(deg, chi=2.), "d")
    out["degree_fold_change_0_to_2"] = dict(measured=float(d0.mean() / d2.mean()),
                                            theory=select(deg, chi=0.)[0]["th_d"] / select(deg, chi=2.)[0]["th_d"])
    # log-RMSE of the conditional curve and of the two averaged curves over all dispersions
    curves = json.loads((HERE / "theory_curves.json").read_text())["degree"]
    by_chi = {round(r["chi"], 4): r for r in curves}
    err = {k: [] for k in ("d", "d_spectra", "d_gains")}
    for chi, cell in groups(deg, "chi").items():
        mean_log = np.log(column(cell, "d")).mean()   # cell mean of log dimension
        for k in err:
            err[k].append(mean_log - np.log(by_chi[round(chi, 4)][k]))
    out["degree_log_rmse_of_cell_means"] = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in err.items()}

    # size series (medians)
    out["size"] = {}
    for label, chi, f in (("chi0", 0., .8), ("chi0.5", .5, .8), ("chi1", 1., .8), ("chi2", 2., .8), ("f0.6", 1., .6)):
        cell = [r for r in live if r["family"] in ("size", "degree", "composition") and np.isclose(r["chi"], chi)
                and np.isclose(r["fraction_e"], f) and (f != .8 or r["family"] != "composition")]
        out["size"][label] = {}
        for n, g in groups(cell, "n").items():
            item = {"networks": len(g)}
            for name in ("d", "nu", "energy", "P_EE", "A_E", "q_E"):
                m, se, _ = median_se(rel(g, name))
                item[f"rel_{name}"] = dict(median=float(m), se=None if not np.isfinite(se) else float(se))
            item["rel_d_mean"] = ms(rel(g, "d"))
            out["size"][label][str(int(n))] = item

    # composition: seed-paired contrasts against 0.80
    comp = select(live, family="composition")
    ref = {r["seed"]: r for r in select(comp, fraction_e=.8)}
    curve = {round(r["fraction_e"], 4): r for r in json.loads((HERE / "theory_curves.json").read_text())["composition"]}
    readout = {round(r["fraction_e"], 4): r for r in json.loads((HERE / "theory_curves.json").read_text())["readout"]}
    out["composition_paired_vs_0.80"] = {}
    for f, cell in groups(comp, "fraction_e").items():
        if np.isclose(f, .8):
            continue
        item = {}
        for name, th in (("PR_E100", readout[round(f, 4)]["R_E100"] - readout[.8]["R_E100"]),
                         ("PR_I100", None if readout[round(f, 4)]["R_I100"] is None else readout[round(f, 4)]["R_I100"] - readout[.8]["R_I100"]),
                         ("P_EE", curve[round(f, 4)]["P_EE"] - curve[.8]["P_EE"]),
                         ("P_II", curve[round(f, 4)]["P_II"] - curve[.8]["P_II"]),
                         ("dE_total", curve[round(f, 4)]["dE_total"] - curve[.8]["dE_total"]),
                         ("d", curve[round(f, 4)]["d"] - curve[.8]["d"])):
            diff = np.array([r[name] - ref[r["seed"]][name] for r in cell if r["seed"] in ref])
            diff = diff[np.isfinite(diff)]
            if len(diff) < 3:
                continue
            m, se, n = mean_se(diff)
            half = student.ppf(.975, n - 1) * se
            item[name] = dict(theory=th, mean=float(m), ci95=[float(m - half), float(m + half)], pairs=int(n))
        out["composition_paired_vs_0.80"][f"{f:g}"] = item
    out["composition_cells"] = {f"{f:g}": dict(rel_d=ms(rel(cell, "d")), rel_nu=ms(rel(cell, "nu")),
                                               rel_energy=ms(rel(cell, "energy")), dE_total=ms(column(cell, "dE_total")),
                                               theory_dE_total=cell[0]["th_dE_total"], overlap=ms(column(cell, "overlap")),
                                               theory_overlap=cell[0]["th_overlap"], PR_E100=ms(column(cell, "PR_E100")),
                                               PR_I100=ms(column(cell, "PR_I100")))
                               for f, cell in groups(comp, "fraction_e").items()}
    best = max(out["composition_cells"].items(), key=lambda kv: kv[1]["dE_total"]["mean"])
    out["composition_dE_total_peak"] = dict(measured_fraction=float(best[0]), measured=best[1]["dE_total"],
                                            theory_fraction=float(max(curve.values(), key=lambda r: r["dE_total"])["fraction_e"]))

    # recording protocol
    windows = [w for w in load("windows.csv") if w["family"] == "record" and w["duration"] == 32000.]
    out["record"] = {}
    for label, chi, f in (("chi0", 0., .8), ("chi1", 1., .8), ("chi2", 2., .8), ("f0.6", 1., .6)):
        cell = select(windows, chi=chi, fraction_e=f)
        item = {}
        for block in (500., 1000., 2000.):
            sub = select(cell, block=block, gap=0., window_duration=32000.)
            if sub:
                item[f"classic_over_meandiff_B{int(block)}"] = ms(column(sub, "classic_d") / column(sub, "d"))
        full = {w["name"]: w["d"] for w in select(cell, block=500., gap=0., window_duration=32000.)}
        for T in (2000., 4000., 8000., 16000.):
            sub = select(cell, block=500., gap=0., window_duration=T)
            if sub:
                item[f"d_T{int(T)}_over_T32000"] = ms(np.array([w["d"] / full[w["name"]] for w in sub]) )
        sub = select(cell, block=500., gap=0., window_duration=32000.)
        if sub:
            item["d_T32000"] = ms(column(sub, "d"))
            net = select(rows, family="record", chi=chi, fraction_e=f)
            item["theory_d"] = net[0]["th_d"]
        ref8 = {(w["seed"]): w for w in select(cell, block=500., gap=0., window_duration=8000.)}
        for dt in (.025, .125):
            nets = [r for r in select(live, family="record", chi=chi, fraction_e=f) if np.isclose(r["dt"], dt)]
            diff = np.array([r["d"] / ref8[r["seed"]]["d"] - 1 for r in nets if r["seed"] in ref8])
            if len(diff):
                item[f"d_step{dt:g}_vs_0.05"] = ms(diff)
        out["record"][label] = item

    # generality
    out["generality"] = {}
    for r_key, cell in groups(select(live, family="generality"), "name").items():
        pass
    gen = select(live, family="generality")
    for (s, m) in sorted({(r["scale"], r["rate"]) for r in gen}):
        cell = [r for r in gen if r["scale"] == s and r["rate"] == m]
        med, se, _ = median_se(rel(cell, "d"))
        out["generality"][f"s={s:g},m={m:g}"] = dict(theory_d=cell[0]["th_d"], d=ms(column(cell, "d")),
                                                     rel_d_median=dict(median=float(med), se=float(se)),
                                                     rel_nu=ms(rel(cell, "nu")), rel_energy=ms(rel(cell, "energy")),
                                                     lyapunov=ms(column(cell, "lyapunov")))
    # Largest Lyapunov exponent: its own short run (2,000 leak times), so judged on its eight block
    # values, not on the state of the 8,000-time recording. "Resolved positive" = every block positive.
    import gzip
    measured = [r for r in rows if np.isfinite(r["lyapunov"])]
    blocks = {}
    for r in measured:
        with gzip.open(HERE / "records" / f"{r['name']}.json.gz", "rt") as handle:
            blocks[r["name"]] = json.load(handle)["extras"]["tangent"]["eight_duration_blocks"]
    resolved = [r for r in measured if min(blocks[r["name"]]) > 0]
    other = [dict(name=r["name"], exponent=r["lyapunov"], state=r["state"], block_min=min(blocks[r["name"]]),
                  block_max=max(blocks[r["name"]])) for r in measured if min(blocks[r["name"]]) <= 0]
    out["lyapunov"] = dict(measured=len(measured), all_blocks_positive=len(resolved),
                           min_resolved=float(min(r["lyapunov"] for r in resolved)),
                           max_resolved=float(max(r["lyapunov"] for r in resolved)), not_resolved_positive=other,
                           families=sorted({r["family"] for r in measured}),
                           largest_n=int(max(r["n"] for r in measured)))

    # lag maps: mean cross-fitted map against the two-lag prediction
    out["lag_maps"] = {}
    for chi in (0., 1., 2.):
        maps = {k: [] for k in ("psi_EE", "psi_EI", "psi_II")}
        for r in select(live, family="degree", chi=chi):
            with np.load(HERE / "arrays" / f"{r['name']}.npz") as z:
                if "psi_EE" in z.files:
                    for k in maps:
                        maps[k].append(z[k])
        if not maps["psi_EE"]:
            continue
        th = theory.lag_map(1., .3, chi)
        item = {"networks": len(maps["psi_EE"])}
        for key, a, b in (("psi_EE", 0, 0), ("psi_EI", 0, 1), ("psi_II", 1, 1)):
            mean, pred = np.mean(maps[key], 0), th["psi"][a, b]
            item[key] = dict(relative_frobenius_error=float(np.linalg.norm(mean - pred) / np.linalg.norm(pred)),
                             correlation=float(np.corrcoef(mean.ravel(), pred.ravel())[0, 1]),
                             most_negative_over_peak=float(mean.min() / mean.max()),
                             zero_lag_ratio=float(mean[24, 24] / pred[24, 24]))
        out["lag_maps"][f"chi={chi:g}"] = item

    (HERE / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = ["# Campaign numbers", "", f"{len(rows)} networks summarized; {sum(out['fixed_point_networks'].values())} sat at a fixed point for the whole recording and {sum(out['settled_during_recording'].values())} settled to one during it.", "",
             "## Degree sweep (N = 2,000)", "", "| chi | theory D/N | measured D/N | relative error | networks |", "|---:|---:|---:|---:|---:|"]
    for chi, v in out["degree"].items():
        lines.append(f"| {chi} | {v['theory_d']:.4f} | {v['d']['mean']:.4f} ± {v['d']['se']:.4f} | {100*v['rel_d']['mean']:+.1f}% ± {100*v['rel_d']['se']:.1f} | {v['d']['n']} |")
    lines += ["", "## Size series: median relative error of D/N", "", "| setting | " + " | ".join(f"N={n:,}" for n in (500, 1000, 2000, 4000, 8000, 16000)) + " |", "|---|" + "---:|" * 6]
    for label, cells in out["size"].items():
        row = []
        for n in (500, 1000, 2000, 4000, 8000, 16000):
            c = cells.get(str(n))
            row.append("" if c is None else f"{100*c['rel_d']['median']:+.1f}% ± {100*(c['rel_d']['se'] or 0):.1f} ({c['networks']})")
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    lines += ["", "## Composition: seed-paired change from 0.80 (95% t interval)", "", "| f_E | quantity | theory | measured | interval | pairs |", "|---:|---|---:|---:|---:|---:|"]
    for f in ("0.6", "0.88", "0.94"):
        for name, v in out["composition_paired_vs_0.80"].get(f, {}).items():
            th = "" if v["theory"] is None else f"{v['theory']:+.4g}"
            lines.append(f"| {f} | {name} | {th} | {v['mean']:+.4g} | [{v['ci95'][0]:+.4g}, {v['ci95'][1]:+.4g}] | {v['pairs']} |")
    (HERE / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({k: out[k] for k in ("fixed_point_networks", "settled_during_recording", "degree_fold_change_0_to_2", "degree_log_rmse_of_cell_means",
                                           "composition_dE_total_peak", "record", "generality", "lyapunov", "lag_maps")}, indent=1))


if __name__ == "__main__":
    main()
