"""Analysis of the composition-optimum validation; writes results.json and the tables behind the figures.

    python experiments/composition_optimum/validation/analyze.py [--draws 4000]

Reads networks.csv (summarize.py) and predictions.json. A cohort is one (rate pair, N). For each
cohort: the structural fit (estimators.py), the optimum f* from fitted coefficients, the
model-free tilt tau = log d(0.94) - log d(0.70), the decomposition (measured T with predicted v
and vice versa) and per-cell summaries; uncertainty by a stratified bootstrap over networks
within cells. Shifts between rate pairs use independent bootstrap draws of the two cohorts.
Networks that ended at a fixed point are counted and excluded. The decision rules are those
of EXPERIMENT.md. Also repeats the fit on the estimator windows T4000 and gap1, and on the
existing paper-campaign networks (13 fractions x 48, N = 2,000, reference rates).

Writes results.json, cells.csv, structure.csv, optimum.csv, bootstrap_fstar.csv.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import estimators as est  # noqa: E402

PRED = json.loads((HERE / "predictions.json").read_text())
FLANKS = tuple(PRED["flanks"])
NAMES = dict(EEE=(0, 0, 0), EIE=(0, 1, 0), IIE=(1, 1, 0), EEI=(0, 0, 1), EII=(0, 1, 1), III=(1, 1, 1))
KEYS = ("E_EE", "E_EI", "E_II", "nu_E", "nu_I")
F_ALT = 0.887   # optimum of total E dimension at the reference


def pred_for(key, fractions):
    r = PRED["rate_pairs"][key]
    T, v = np.array(r["tensor_array"]), np.array(r["v"])
    mu = {}
    for f in fractions:
        fr = np.array([f, 1 - f])
        E = np.outer(fr, fr) * np.einsum("abc,c->ab", T, 1 / fr)
        mu[f] = np.array([E[0, 0], E[0, 1], E[1, 1]])
    return dict(mu=mu, T=T, v=v, f_star=r["f_star"])


def ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [np.nan, np.nan]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


def load(path, suffix=""):
    rows = list(csv.DictReader(open(path)))
    return rows


def cohort_cells(rows, key, n, suffix=""):
    cells, excluded = {}, {}
    for r in rows:
        if r["rates"] != key or int(r["n"]) != n:
            continue
        f = float(r["fraction_e"])
        if r["fluctuating"] != "1":
            excluded[f] = excluded.get(f, 0) + 1
            continue
        cells.setdefault(f, []).append([float(r[k + suffix]) for k in KEYS])
    return {f: np.array(v) for f, v in sorted(cells.items())}, excluded


def analyse_cohort(cells, pred, draws, seed):
    s = est.summarize(cells, pred, FLANKS)
    boot = est.bootstrap(cells, pred, FLANKS, draws=draws, seed=seed)
    fb = np.array([b["f_star"] for b in boot])
    Tb = np.array([b["T"] for b in boot])
    vb = np.array([b["v"] for b in boot])
    fit = s["fit"]
    out = dict(
        networks=int(sum(len(c) for c in cells.values())), per_cell={f"{f:g}": len(c) for f, c in cells.items()},
        f_star=s["f_star"], f_star_ci=ci(fb), f_star_se=float(np.nanstd(fb)), d_star=s["d_star"],
        unique_fraction=float(np.mean([b["nroots"] == 1 for b in boot])), nroots_point=s["nroots"],
        no_interior_root_fraction=float(np.mean(~np.isfinite(fb))),
        f_star_T_only=s["f_star_T_only"], f_star_T_only_ci=ci([b["f_star_T_only"] for b in boot]),
        f_star_v_only=s["f_star_v_only"], f_star_v_only_ci=ci([b["f_star_v_only"] for b in boot]),
        tilt=s["tilt"], tilt_ci=ci([b["tilt"] for b in boot]),
        tilt_mean_of_ratios=s["tilt_mean_of_ratios"], tilt_mean_of_ratios_ci=ci([b["tilt_mean_of_ratios"] for b in boot]),
        T={k: float(fit["T"][i]) for k, i in NAMES.items()},
        T_ratio={k: float(fit["T"][i] / pred["T"][i]) for k, i in NAMES.items()},
        T_ratio_ci={k: ci(Tb[(slice(None),) + i] / pred["T"][i]) for k, i in NAMES.items()},
        v=fit["v"].tolist(), v_ratio=(fit["v"] / pred["v"]).tolist(),
        v_ratio_ci=[ci(vb[:, 0] / pred["v"][0]), ci(vb[:, 1] / pred["v"][1])],
        v_slope=fit["v_slope"], v_slope_se=fit["v_slope_se"], v_chi2=fit["v_chi2"], dof_v=fit["dof_v"],
        block_chi2=fit["chi2"], dof_blocks=fit["dof_blocks"], block_residual_rel=fit["residual_rel"],
        fitted_curve=dict(f=np.round(np.linspace(.5, .97, 95), 6).tolist(), d=est.curve(fit["T"], fit["v"], np.linspace(.5, .97, 95)).tolist()),
        curve_band=None)
    # pointwise 95% band of the fitted curve
    grid = np.linspace(.5, .97, 95)
    curves = np.array([est.curve(b["T"], b["v"], grid) for b in boot[:1000]])
    out["curve_band"] = dict(lo=np.percentile(curves, 2.5, axis=0).tolist(), hi=np.percentile(curves, 97.5, axis=0).tolist())
    return out, fb, np.array([b["tilt"] for b in boot])


def cell_rows(cells, excluded, rows, key, n, pred):
    out = []
    for f, c in cells.items():
        sub = [r for r in rows if r["rates"] == key and int(r["n"]) == n and float(r["fraction_e"]) == f and r["fluctuating"] == "1"]
        rate = np.array([[float(r["rate_E"]), float(r["rate_I"])] for r in sub])
        d_net = np.array([float(r["d"]) for r in sub])
        m, se = c.mean(0), c.std(0, ddof=1) / np.sqrt(len(c))
        nu = f * c[:, 3] + (1 - f) * c[:, 4]
        en = c[:, :3] @ est.TOTAL
        thc = PRED["rate_pairs"][key]["cells"][f"{f:g}"]
        row = dict(rates=key, n=n, fraction_e=f, networks=len(c), excluded=excluded.get(f, 0),
                   d_ratio=float(nu.mean()**2 / en.mean()), d_mean=float(d_net.mean()), d_se=float(d_net.std(ddof=1) / np.sqrt(len(c))),
                   th_d=thc["d"], energy=float(en.mean()), energy_se=float(en.std(ddof=1) / np.sqrt(len(c))), th_energy=thc["energy"],
                   nu=float(nu.mean()), nu_se=float(nu.std(ddof=1) / np.sqrt(len(c))), th_nu=thc["nu"],
                   rate_E=float(rate[:, 0].mean()), rate_E_sd=float(rate[:, 0].std(ddof=1)),
                   rate_I=float(rate[:, 1].mean()), rate_I_sd=float(rate[:, 1].std(ddof=1)),
                   target_E=PRED["rate_pairs"][key]["rates"][0], target_I=PRED["rate_pairs"][key]["rates"][1])
        for k, name in enumerate(KEYS):
            row[name], row[name + "_se"] = float(m[k]), float(se[k])
            row["th_" + name] = float(pred["mu"][f][k]) if k < 3 else float(pred["v"][k - 3])
            row[name + "_rel_sd"] = float(c[:, k].std(ddof=1) / m[k])
        out.append(row)
    return out


def verdicts(R):
    v = {}
    c = PRED["contrasts"]
    for N in ("2000", "4000", "8000"):
        if f"shift_lo_N{N}" not in R["shifts"] or f"shift_hi_N{N}" not in R["shifts"]:
            continue
        res = {}
        for side, p in (("lo", c["shift_lo"]), ("hi", c["shift_hi"])):
            s = R["shifts"][f"shift_{side}_N{N}"]
            lo, hi = s["ci"]
            sign_ok = (lo > 0 and hi > 0) if p > 0 else (lo < 0 and hi < 0)
            res[side] = dict(excludes_zero_with_sign=bool(sign_ok), contains_prediction=bool(lo <= p <= hi))
        if all(r["excludes_zero_with_sign"] and r["contains_prediction"] for r in res.values()):
            word = "held"
        elif all(r["excludes_zero_with_sign"] for r in res.values()):
            word = "direction held, magnitude not"
        else:
            word = "failed"
        v[f"P-iv_N{N}"] = dict(verdict=word, **res)
    ref = PRED["rate_pairs"]["ref"]["f_star"]
    for N in ("8000", "16000"):
        k = f"ref_N{N}"
        if k in R["cohorts"]:
            lo, hi = R["cohorts"][k]["f_star_ci"]
            v[f"P-iii_N{N}"] = dict(contains_0851=bool(lo <= ref <= hi), excludes_0887=bool(not lo <= F_ALT <= hi), ci=[lo, hi])
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=4000)
    args = ap.parse_args()
    rows = load(HERE / "networks.csv")
    sizes = sorted({int(r["n"]) for r in rows})
    R = dict(draws=args.draws, cohorts={}, shifts={}, windows={}, excluded={}, campaign_replication=None)
    cells_out, boots, tilts = [], {}, {}
    seed = 1
    for n in sizes:
        for key in ("ref", "lo", "hi"):
            cells, excl = cohort_cells(rows, key, n)
            if len(cells) < 3:
                continue
            pred = pred_for(key, list(cells))
            res, fb, tb = analyse_cohort(cells, pred, args.draws, seed)
            seed += 1
            res["excluded"] = {f"{f:g}": c for f, c in excl.items()}
            R["cohorts"][f"{key}_N{n}"] = res
            boots[(key, n)], tilts[(key, n)] = fb, tb
            cells_out += cell_rows(cells, excl, rows, key, n, pred)
            for suffix, tag in (("_T4000", "T4000"), ("_gap1", "gap1")):
                cw, _ = cohort_cells(rows, key, n, suffix)
                rw, fbw, _ = analyse_cohort(cw, pred, max(500, args.draws // 4), seed)
                seed += 1
                R["windows"][f"{key}_N{n}_{tag}"] = dict(f_star=rw["f_star"], f_star_ci=rw["f_star_ci"], tilt=rw["tilt"],
                                                         tilt_ci=rw["tilt_ci"], T_ratio=rw["T_ratio"], v_ratio=rw["v_ratio"])
            print(f"{key} N={n}: f*={res['f_star']:.4f} {np.round(res['f_star_ci'], 4)} tilt={res['tilt']:.4f} "
                  f"T/Tp={ {k: round(x, 3) for k, x in res['T_ratio'].items()} } v/vp={np.round(res['v_ratio'], 3)} "
                  f"chi2={np.round(res['block_chi2'], 1)} unique={res['unique_fraction']:.3f}", flush=True)
        for side in ("lo", "hi"):
            if (side, n) in boots and ("ref", n) in boots:
                d = boots[(side, n)] - boots[("ref", n)]
                dt = tilts[(side, n)] - tilts[("ref", n)]
                R["shifts"][f"shift_{side}_N{n}"] = dict(
                    estimate=R["cohorts"][f"{side}_N{n}"]["f_star"] - R["cohorts"][f"ref_N{n}"]["f_star"], ci=ci(d), se=float(np.nanstd(d)),
                    predicted=PRED["contrasts"][f"shift_{side}"])
                R["shifts"][f"tilt_{side}_minus_ref_N{n}"] = dict(
                    estimate=R["cohorts"][f"{side}_N{n}"]["tilt"] - R["cohorts"][f"ref_N{n}"]["tilt"], ci=ci(dt),
                    predicted=PRED["contrasts"][f"tilt_{side}_minus_ref"])
    # size comparison of the reference optimum
    for a, b in ((2000, 8000), (8000, 16000), (2000, 16000)):
        for key in ("ref", "lo", "hi"):
            if (key, a) in boots and (key, b) in boots:
                d = boots[(key, b)] - boots[(key, a)]
                R["shifts"][f"size_{key}_N{b}_minus_N{a}"] = dict(estimate=R["cohorts"][f"{key}_N{b}"]["f_star"] - R["cohorts"][f"{key}_N{a}"]["f_star"],
                                                                  ci=ci(d))
    R["verdicts"] = verdicts(R)
    # existing campaign networks, fitted the same way
    camp = [x for x in csv.DictReader(open(HERE.parents[1] / "paper_campaign/networks.csv"))
            if x["family"] == "composition" and x["fluctuating"] == "1"]
    cc = {}
    for x in camp:
        cc.setdefault(float(x["fraction_e"]), []).append([float(x[k]) for k in KEYS])
    cc = {f: np.array(v) for f, v in sorted(cc.items())}
    pc = pred_for("ref", list(cc))
    rc, fbc, _ = analyse_cohort(cc, pc, args.draws, 999)
    R["campaign_replication"] = {k: rc[k] for k in ("networks", "f_star", "f_star_ci", "T_ratio", "T_ratio_ci", "v_ratio", "v_ratio_ci",
                                                    "block_chi2", "dof_blocks", "unique_fraction", "v_slope", "v_slope_se")}
    (HERE / "results.json").write_text(json.dumps(R, indent=1, default=float) + "\n")

    def write(name, rws):
        fields = []
        for r in rws:
            fields += [k for k in r if k not in fields]
        with open(HERE / name, "w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=fields)
            w.writeheader()
            w.writerows(rws)
    write("cells.csv", cells_out)
    struct, opt = [], []
    for k, c in R["cohorts"].items():
        key, n = k.split("_N")
        for name in NAMES:
            struct.append(dict(rates=key, n=int(n), quantity="T_" + name, fitted=c["T"][name], predicted=float(pred_for(key, [.8])["T"][NAMES[name]]),
                               ratio=c["T_ratio"][name], ratio_lo=c["T_ratio_ci"][name][0], ratio_hi=c["T_ratio_ci"][name][1]))
        for i, s in enumerate("EI"):
            struct.append(dict(rates=key, n=int(n), quantity="v_" + s, fitted=c["v"][i], predicted=PRED["rate_pairs"][key]["v"][i],
                               ratio=c["v_ratio"][i], ratio_lo=c["v_ratio_ci"][i][0], ratio_hi=c["v_ratio_ci"][i][1]))
        opt.append(dict(rates=key, n=int(n), networks=c["networks"], f_star=c["f_star"], lo=c["f_star_ci"][0], hi=c["f_star_ci"][1],
                        se=c["f_star_se"], predicted=PRED["rate_pairs"][key]["f_star"], unique_fraction=c["unique_fraction"],
                        f_star_T_only=c["f_star_T_only"], f_star_v_only=c["f_star_v_only"], tilt=c["tilt"], tilt_lo=c["tilt_ci"][0],
                        tilt_hi=c["tilt_ci"][1], tilt_predicted=PRED["rate_pairs"][key]["tilt_log_d_94_minus_70"]))
    write("structure.csv", struct)
    write("optimum.csv", opt)
    with open(HERE / "bootstrap_fstar.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["rates", "n", "draw", "f_star", "tilt"])
        for (key, n), fb in boots.items():
            for i, (x, t) in enumerate(zip(fb, tilts[(key, n)])):
                w.writerow([key, n, i, f"{x:.6f}", f"{t:.6f}"])
    print(json.dumps(R["verdicts"], indent=1))
    print(json.dumps({k: (round(v["estimate"], 4), np.round(v["ci"], 4).tolist(), v.get("predicted")) for k, v in R["shifts"].items()}, indent=0))


if __name__ == "__main__":
    main()
