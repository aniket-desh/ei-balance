"""Fixed-point modes and recorded covariance across network size and degree dispersion.

    python experiments/paper_campaign/plots/spectral.py

Reads experiments/paper_campaign/spectral/{measurements,spectra}. All statistics are those of
experiments/spectral_replication (first verified root, 64 within-population shuffles and 64
gain-decile shuffles per network); the unit is a network.
"""
import json

import numpy as np

from common import CAMPAIGN, CROSS, E, GOLD, GRAY, I, JOINT, LIGHT, SEQ, letters, mean_se, panels, plt, save, style

SPEC = CAMPAIGN / "spectral"
GROUPS = [("bernoulli", 250, "250"), ("bernoulli", 500, "500"), ("bernoulli", 1000, "1,000"), ("bernoulli", 2000, "2,000")]
DISPERSION = [("chi0", 1000, r"$\chi=0$"), ("bernoulli", 1000, r"$\chi=1$"), ("chi2", 1000, r"$\chi=2$")]


def load():
    cases = {}
    for path in sorted((SPEC / "measurements").glob("spectral_*.json")):
        c = json.loads(path.read_text())
        cell = path.stem.split("_")[1]
        row = dict(name=c["name"], cell=cell, n=c["N"], fluctuating=c["state_fluctuating"], roots=len(c["roots"]))
        if c["roots"] and c["state_fluctuating"]:
            r = c["roots"][0]
            row.update(rho=r["rho_input"][0], rho_null=float(np.median(r["population_null_rho"])),
                       unstable=r["unstable_mode_count"], max_real=r["max_real_coupling_eigenvalue"],
                       trace=r["rate_projection"][0][0], energy=r["rate_projection"][0][1], pr=r["rate_projection"][0][2],
                       energy_gain=float(np.median(np.array(r["gain_decile_null_rate"])[:, 1])),
                       pr_gain=float(np.median(np.array(r["gain_decile_null_rate"])[:, 2])),
                       energy_pop=float(np.median(np.array(r["population_null_rate"])[:, 1])),
                       half_agreement=r["half_rank_agreement"])
        elif c["roots"]:
            row.update(max_real=c["roots"][0]["max_real_coupling_eigenvalue"], unstable=c["roots"][0]["unstable_mode_count"])
        cases.setdefault((cell, c["N"]), []).append(row)
    return cases


def paired(ax, cases, groups, observed, null, color, ylabel, reference=None):
    for k, (cell, n, label) in enumerate(groups):
        rows = [r for r in cases.get((cell, n), []) if observed in r]
        if not rows:
            continue
        obs = np.array([r[observed] for r in rows])
        nul = np.array([r[null] for r in rows])
        for o, u in zip(obs, nul):
            ax.plot([k - .16, k + .16], [u, o], "-", color=LIGHT, lw=.5, zorder=1)
        ax.plot(np.full(len(nul), k - .16), nul, ".", color=GRAY, ms=3, alpha=.6, mew=0, zorder=2)
        ax.plot(np.full(len(obs), k + .16), obs, ".", color=color, ms=3, alpha=.6, mew=0, zorder=2)
        for x, v, c in ((k - .16, nul, GRAY), (k + .16, obs, color)):
            ax.plot([x - .11, x + .11], [np.median(v)] * 2, "-", color=c, lw=1.6, zorder=4)
    if reference is not None:
        ax.axhline(reference, color=GRAY, lw=.7, ls=(0, (4, 2)))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[2] for g in groups])
    ax.set_ylabel(ylabel)


def main():
    style()
    cases = load()
    fig, ax = panels(2, 4, width=7.2, height=4.3)
    # (a) one example spectrum at the largest size with a verified root
    example = None
    for n in (2000, 1000, 500):
        rows = [r for r in cases.get(("bernoulli", n), []) if "rho" in r]
        if rows:
            example = sorted(rows, key=lambda r: r["name"])[0]
            break
    if example:
        with np.load(SPEC / "spectra" / f"{example['name']}.npz") as z:
            lam, var = z["eigenvalues"], z["input_variance"][0]
        order = np.argsort(var)
        sc = ax[0, 0].scatter(lam.real[order], lam.imag[order], c=np.log10(np.maximum(var[order], 1e-12)), s=3,
                              cmap=SEQ, linewidths=0, rasterized=True)
        ax[0, 0].axvline(1, color=GRAY, lw=.7, ls=(0, (4, 2)))
        ax[0, 0].set_xlabel(r"$\mathrm{Re}\,\lambda$")
        ax[0, 0].set_ylabel(r"$\mathrm{Im}\,\lambda$")
        bar = fig.colorbar(sc, ax=ax[0, 0], shrink=.8, pad=.02, aspect=18)
        bar.set_label(r"$\log_{10}$ input variance", fontsize=7)
    paired(ax[0, 1], cases, GROUPS, "rho", "rho_null", JOINT, r"rank correlation $\rho$", 0)
    paired(ax[0, 2], cases, GROUPS, "energy", "energy_gain", E, "energy in leading tenth")
    paired(ax[0, 3], cases, GROUPS, "pr", "pr_gain", GOLD, "complement PR / full PR", 1)
    for a in ax[0, 1:]:
        a.set_xlabel(r"network size $N$")
    # bottom row: states, instability counts, and the dispersion comparison
    xs, frac_fluct, frac_root = [], [], []
    for cell, n, label in GROUPS:
        rows = cases.get((cell, n), [])
        if rows:
            xs.append(label), frac_fluct.append(np.mean([r["fluctuating"] for r in rows]))
            frac_root.append(np.mean([r["roots"] > 0 for r in rows]))
    ax[1, 0].plot(xs, frac_fluct, "o-", color=JOINT, ms=3.6, label="fluctuating")
    ax[1, 0].plot(xs, frac_root, "s--", color=GRAY, ms=3.4, mfc="white", label="verified fixed point found")
    ax[1, 0].set_ylim(0, 1.05)
    ax[1, 0].set_ylabel("fraction of networks")
    ax[1, 0].set_xlabel(r"network size $N$")
    ax[1, 0].legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(.14, .04))
    for k, (cell, n, label) in enumerate(GROUPS):
        rows = [r for r in cases.get((cell, n), []) if "unstable" in r and r["fluctuating"]]
        if rows:
            v = np.array([r["unstable"] for r in rows]) / n
            ax[1, 1].plot(np.full(len(v), k) + np.linspace(-.15, .15, len(v)), v, ".", color=JOINT, ms=3, alpha=.5, mew=0)
            ax[1, 1].plot([k - .2, k + .2], [np.median(v)] * 2, "-", color=JOINT, lw=1.6)
    ax[1, 1].set_xticks(range(len(GROUPS)))
    ax[1, 1].set_xticklabels([g[2] for g in GROUPS])
    ax[1, 1].set_ylabel(r"unstable modes / $N$")
    ax[1, 1].set_xlabel(r"network size $N$")
    paired(ax[1, 2], cases, DISPERSION, "rho", "rho_null", JOINT, r"rank correlation $\rho$", 0)
    paired(ax[1, 3], cases, DISPERSION, "energy", "energy_gain", E, "energy in leading tenth")
    for a in ax[1, 2:]:
        a.set_xlabel(r"dispersion ($N=1{,}000$)")
    letters(ax)
    counts = ", ".join(f"{len(cases.get((c, n), []))} at {lab}" for c, n, lab in GROUPS)
    save(fig, "spectral_cohort",
         "Fixed-point modes organize recorded fluctuations across network size and dispersion. (a) Eigenvalues of "
         f"J diag(phi'(x*)) at the first verified fixed point of one network of {example['n']:,} cells; color is the recorded input variance "
         "along each unit right eigenvector, and the dashed line is the instability threshold. (b-d) For each "
         "fluctuating Bernoulli network, the observed statistic (colored) beside its shuffle median (gray), joined by "
         "a line; horizontal bars are medians across networks. (b) Spearman correlation rho between a mode's real "
         "eigenvalue and its input variance, against within-population coordinate shuffles. (c) Fraction of squared "
         "rate-covariance energy inside the leading tenth of fixed-point directions, against gain-decile-matched "
         "shuffles. (d) PR of the recording after projecting out that subspace, relative to the full PR. (e) "
         "Fraction of networks that fluctuate and fraction with a verified fixed point. (f) Number of unstable "
         "fixed-point modes per neuron. (g,h) The comparisons of (b,c) at N = 1,000 for degree dispersion 0, 1 "
         f"and 2. Networks: {counts} cells (Bernoulli); 16 each at dispersion 0 and 2.")
    # numbers for the write-up
    out = {}
    for (cell, n), rows in sorted(cases.items()):
        fl = [r for r in rows if "rho" in r]
        item = dict(networks=len(rows), fluctuating=int(sum(r["fluctuating"] for r in rows)), with_root=int(sum(r["roots"] > 0 for r in rows)))
        if fl:
            d_rho = np.array([r["rho"] - r["rho_null"] for r in fl])
            d_en = np.array([r["energy"] - r["energy_gain"] for r in fl])
            item.update(analysed=len(fl), median_rho=float(np.median([r["rho"] for r in fl])),
                        median_rho_minus_shuffle=float(np.median(d_rho)), rho_positive=int(np.sum(d_rho > 0)),
                        median_energy_fraction=float(np.median([r["energy"] for r in fl])),
                        median_gain_shuffle_energy=float(np.median([r["energy_gain"] for r in fl])),
                        energy_above_shuffle=int(np.sum(d_en > 0)),
                        median_complement_pr_ratio=float(np.median([r["pr"] for r in fl])),
                        complement_pr_above_one=int(np.sum(np.array([r["pr"] for r in fl]) > 1)),
                        median_unstable_per_neuron=float(np.median([r["unstable"] / n for r in fl])))
        out[f"{cell}_N{n}"] = item
    (SPEC / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
