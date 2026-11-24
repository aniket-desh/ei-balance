"""Single-neuron laws, temporal kernels, two-lag pair moments and covariance spectra.

    python experiments/paper_campaign/plots/local_temporal.py

These figures read the per-network arrays (arrays/<name>.npz): each neuron's
time-averaged input, temporal input variance, rate variance, mean gain and realized
in-degree; kernels and cross-fitted lag maps from 192 sampled neurons per
population; and eigenvalues of the whole-record rate covariance.
"""
import sys

import numpy as np

from common import (CAMPAIGN, CROSS, E, GOLD, GRAY, I, JOINT, LIGHT, SEQ, SIGNED, fluctuating, letters, load, panels,
                    plt, save, select, style)

sys.path.insert(0, str(CAMPAIGN))
import theory

POPS = (("E", E), ("I", I))
DEGREE = [(0., r"$\chi=0$"), (1., r"$\chi=1$"), (2., r"$\chi=2$")]


def arrays(row):
    return np.load(CAMPAIGN / "arrays" / f"{row['name']}.npz")


def pool(rows, keys, limit=12):
    """Concatenate per-neuron arrays of the first `limit` networks, split by population."""
    out = {k: [[], []] for k in keys}
    for r in rows[:limit]:
        with arrays(r) as a:
            ne = int(a["n_e"])
            for k in keys:
                v = a[k]
                out[k][0].append(v[:ne])
                out[k][1].append(v[ne:])
    return {k: [np.concatenate(v[0]), np.concatenate(v[1])] for k, v in out.items()}


def local_laws(networks):
    """Operating-point distribution and conditional variance and gain: the content of manuscript Fig. 2, measured."""
    fig, ax = panels(3, 3, height=6.2)
    for j, (chi, label) in enumerate(DEGREE):
        rows = fluctuating(select(networks, family="degree", chi=chi))
        if not rows:
            continue
        summary, th = theory.solve(1., .3, chi)
        data = pool(rows, ("mean_x", "variance_rate", "mean_gain"))
        v_th = th["covariance"][..., 0]
        for a, (pop, color) in enumerate(POPS):
            mu = data["mean_x"][a]
            h, delta = summary["mean_inputs"][a], summary["static_variances"][a]
            bins = np.linspace(mu.min() - .3, mu.max() + .3, 70)
            ax[0, j].hist(mu, bins=bins, density=True, color=color, alpha=.35, lw=0)
            if delta > 0:
                x = np.linspace(bins[0], bins[-1], 400)
                ax[0, j].plot(x, np.exp(-(x - h)**2 / (2 * delta)) / np.sqrt(2 * np.pi * delta), color=color, lw=1.3)
            else:
                ax[0, j].axvline(h, color=color, lw=1.3)
            pick = np.random.default_rng(0).choice(len(mu), min(len(mu), 5000), replace=False)
            for row, key, curve in ((1, "variance_rate", v_th[a]), (2, "mean_gain", th["alpha"][a])):
                ax[row, j].plot(mu[pick], data[key][a][pick], ".", color=color, ms=1.6, alpha=.18, mew=0, rasterized=True)
                # binned mean of the measurements, to compare with the conditional curve
                edges = np.quantile(mu, np.linspace(0, 1, 16))
                mid = [mu[(mu >= lo) & (mu <= hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
                val = [data[key][a][(mu >= lo) & (mu <= hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
                ax[row, j].plot(mid, val, "o", color=color, ms=2.8, mec="white", mew=.4, zorder=5)
                if delta > 0:
                    ax[row, j].plot(th["mu"][a], curve, "-", color=color, lw=1.3, zorder=4)
                else:
                    ax[row, j].plot([h], [curve[0]], "D", mfc="white", mec=color, ms=5, zorder=6)
        ax[0, j].set_title(label)
        ax[2, j].set_xlabel(r"time-averaged input $\mu_i$")
        for row in range(3):
            ax[row, j].set_xlim(*np.quantile(np.concatenate(data["mean_x"]), [.001, .999]) + np.array([-.5, .5]))
    ax[0, 0].set_ylabel("density of operating points")
    ax[1, 0].set_ylabel(r"temporal rate variance $v_i$")
    ax[2, 0].set_ylabel(r"time-averaged gain $\alpha_i$")
    letters(ax)
    save(fig, "local_laws",
         "The same operating point controls a neuron's rate variance and gain. Columns: degree dispersion 0, 1 and 2 "
         "at N = 2,000 (12 networks pooled; E blue, I red). Top: distribution of time-averaged inputs with the "
         "predicted Gaussian (a vertical line at zero dispersion, where the leading theory has no static spread). "
         "Middle: each neuron's temporal rate variance against its mean input (dots, 5,000 neurons per population), "
         "binned means (circles) and the conditional prediction v_a(z) (line). Bottom: the same for the "
         "time-averaged gain and alpha_a(z). At zero dispersion the diamond marks the single predicted point.")


def degree_inputs(networks):
    fig, ax = panels(2, 3, height=4.2)
    for j, (chi, label) in enumerate(DEGREE):
        rows = fluctuating(select(networks, family="degree", chi=chi))
        if not rows:
            continue
        data = pool(rows, ("mean_x", "q_x", "indegree"), limit=6)
        for a, (pop, color) in enumerate(POPS):
            mu, deg = data["mean_x"][a], data["indegree"][a].astype(float)
            # net count drive: excess E inputs minus excess I inputs, each in units of its own SD
            ze = (deg[:, 0] - deg[:, 0].mean()) / (deg[:, 0].std() + 1e-12)
            zi = (deg[:, 1] - deg[:, 1].mean()) / (deg[:, 1].std() + 1e-12)
            pick = np.random.default_rng(1).choice(len(mu), min(len(mu), 3000), replace=False)
            ax[0, j].plot(deg[pick, 1] - deg[:, 1].mean(), mu[pick], ".", color=color, ms=1.8, alpha=.2, mew=0, rasterized=True)
            ax[1, j].plot(mu[pick], data["q_x"][a][pick], ".", color=color, ms=1.8, alpha=.2, mew=0, rasterized=True)
            r_i = np.corrcoef(deg[:, 1], mu)[0, 1] if deg[:, 1].std() > 0 else np.nan
            r_e = np.corrcoef(deg[:, 0], mu)[0, 1] if deg[:, 0].std() > 0 else np.nan
            ax[0, j].text(.97, .97 - .1 * a, rf"{pop}: $r_I={r_i:+.2f}$, $r_E={r_e:+.2f}$", transform=ax[0, j].transAxes,
                          ha="right", va="top", color=color, fontsize=6.8)
        ax[0, j].set_title(label)
        ax[0, j].set_xlabel("inhibitory in-degree minus class mean")
        ax[1, j].set_xlabel(r"time-averaged input $\mu_i$")
    ax[0, 0].set_ylabel(r"time-averaged input $\mu_i$")
    ax[1, 0].set_ylabel(r"temporal input variance $q_i$")
    letters(ax)
    save(fig, "local_degree_and_inputs",
         "Where the operating points come from. Top: each neuron's time-averaged input against its realized "
         "inhibitory in-degree (r_I and r_E are the correlations of mean input with inhibitory and excitatory "
         "in-degree). With fixed in-degree the remaining spread comes from sampling unequal presynaptic rates. "
         "Bottom: each neuron's temporal input variance against its mean input; the closure assigns one temporal "
         "variance per population. Six networks pooled per dispersion, 3,000 neurons shown per population, N = 2,000.")


def kernels(networks):
    cases = [("degree", dict(chi=0.), (1., .3, 0.), r"$\chi=0$"), ("degree", dict(chi=1.), (1., .3, 1.), r"$\chi=1$"),
             ("degree", dict(chi=2.), (1., .3, 2.), r"$\chi=2$"), ("generality", dict(scale=3.), (3., .3, 1.), r"$s=3$")]
    fig, ax = panels(2, 4, width=7.2, height=3.9, sharex=True)
    for j, (family, where, args, label) in enumerate(cases):
        rows = fluctuating(select(networks, family=family, **where))
        if not rows:
            continue
        _, th = theory.solve(*args)
        t = th["time"]
        rate_th = np.einsum("azt,z->at", th["covariance"].astype(float), th["weights"])
        stack = {k: [] for k in ("input_kernel_E", "input_kernel_I", "rate_kernel_E", "rate_kernel_I")}
        for r in rows:
            with arrays(r) as a:
                lags = a["kernel_lags"]
                for k in stack:
                    stack[k].append(a[k])
        for a_, (pop, color) in enumerate(POPS):
            for row, key, curve in ((0, f"input_kernel_{pop}", th["kernels"][a_]), (1, f"rate_kernel_{pop}", rate_th[a_])):
                m = np.mean(stack[key], 0)
                sd = np.std(stack[key], 0, ddof=1) / np.sqrt(len(stack[key]))
                ax[row, j].plot(t, curve, "-", color=color, lw=1.3)
                ax[row, j].errorbar(lags[::6], m[::6], yerr=sd[::6], fmt="o", color=color, ms=2.6, elinewidth=.7,
                                    mec="white", mew=.3)
        ax[0, j].set_title(label)
        ax[0, j].set_xlim(0, 30)
        ax[1, j].set_xlabel(r"lag $\tau$ (leak times)")
    ax[0, 0].set_ylabel(r"input autocovariance $K_a(\tau)$")
    ax[1, 0].set_ylabel(r"rate autocovariance $\langle c_a(z,\tau)\rangle_z$")
    letters(ax)
    save(fig, "temporal_kernels",
         "Single-neuron temporal kernels against the one-site prediction. Top: autocovariance of the input "
         "(population-shared dynamic kernel K_a). Bottom: autocovariance of the rate, averaged over operating "
         "points. Lines are parameter-only; markers are means over networks (192 sampled neurons per population per "
         "network, whole-record centering) with one across-network standard error. Columns: degree dispersion 0, 1, "
         "2 at the reference coupling, and coupling variance scaled by 3. N = 2,000.")


def lag_maps(networks):
    cases = [("degree", dict(chi=0.), (1., .3, 0.), "chi0", r"$\chi=0$"), ("degree", dict(chi=1.), (1., .3, 1.), "chi1", r"$\chi=1$"),
             ("degree", dict(chi=2.), (1., .3, 2.), "chi2", r"$\chi=2$")]
    for family, where, args, tag, label in cases:
        rows = fluctuating(select(networks, family=family, **where))
        maps = {k: [] for k in ("psi_EE", "psi_EI", "psi_II", "source_E", "source_I")}
        for r in rows:
            with arrays(r) as a:
                if "psi_EE" in a.files:
                    for k in maps:
                        maps[k].append(a[k])
        if not maps["psi_EE"]:
            continue
        th = theory.lag_map(*args)
        lags = th["lags"]
        fig, ax = panels(3, 3, height=6.3)
        zero = len(lags) // 2
        for j, (key, a, b, name) in enumerate((("psi_EE", 0, 0, "E-E"), ("psi_EI", 0, 1, "E-I"), ("psi_II", 1, 1, "I-I"))):
            mean = np.mean(maps[key], 0)
            se = np.std(maps[key], 0, ddof=1) / np.sqrt(len(maps[key]))
            pred = th["psi"][a, b]
            vmax = max(np.abs(th["psi"]).max(), max(np.abs(np.mean(maps[k], 0)).max() for k in ("psi_EE", "psi_EI", "psi_II")))
            for row, m, title in ((0, pred, f"theory, {name}"), (1, mean, f"networks, {name}")):
                im = ax[row, j].imshow(m, origin="lower", extent=[lags[0], lags[-1], lags[0], lags[-1]], cmap=SIGNED,
                                       vmin=-vmax, vmax=vmax, aspect="equal", interpolation="nearest")
                ax[row, j].contour(lags, lags, m, levels=[-.02 * vmax], colors=[JOINT], linewidths=.5, linestyles="--")
                ax[row, j].set_title(title)
                ax[row, j].set_xlabel(r"$\tau_2$")
                ax[row, j].set_ylabel(r"$\tau_1$")
            if j == 2:
                bar = fig.colorbar(im, ax=ax[:2, :], shrink=.55, pad=.015, aspect=26)
                bar.set_label(r"$\psi_{ab}(\tau_1,\tau_2)$")
            color = (E, CROSS, I)[j]
            ax[2, j].plot(lags, np.diag(pred), "-", color=color, lw=1.3, label=r"$\tau_1=\tau_2$")
            ax[2, j].errorbar(lags[::2], np.diag(mean)[::2], yerr=np.diag(se)[::2], fmt="o", color=color, ms=2.8,
                              elinewidth=.7)
            ax[2, j].plot(lags, pred[zero], "-", color=GRAY, lw=1.1, label=r"$\tau_2=0$")
            ax[2, j].errorbar(lags[::2], mean[zero][::2], yerr=se[zero][::2], fmt="s", color=GRAY, ms=2.6,
                              elinewidth=.7, mfc="white")
            ax[2, j].axhline(0, color=LIGHT, lw=.6)
            ax[2, j].set_xlabel(r"lag $\tau_1$")
            ax[2, j].set_ylabel(r"$\psi_{ab}$")
        ax[2, 0].legend()
        letters(ax)
        save(fig, f"lag_maps_{tag}",
             f"Two-lag pair moments at {label.strip('$')}, N = 2,000 ({len(maps['psi_EE'])} networks). "
             "psi_ab(tau1,tau2) is N times the mean over distinct pairs (i in a, j in b) of C_ij(tau1) C_ij(tau2). "
             "Top: parameter-only prediction. Middle: mean of cross-fitted network estimates, in which the two "
             "covariance factors come from disjoint halves of the recording so that estimation noise does not "
             "multiply itself. Each column shares one symmetric color scale; dashed contours mark negative values. "
             "Bottom: equal-lag slice (colored) and tau2 = 0 slice (gray) with one across-network standard error. "
             "No amplitude or timescale is fitted.")
    # same-neuron sources Q_a(t1,t2) at chi = 1
    rows = fluctuating(select(networks, family="degree", chi=1.))
    src = {k: [] for k in ("source_E", "source_I")}
    for r in rows:
        with arrays(r) as a:
            if "source_E" in a.files:
                for k in src:
                    src[k].append(a[k])
    if src["source_E"]:
        th = theory.lag_map(1., .3, 1.)
        lags = th["lags"]
        zero = len(lags) // 2
        fig, ax = panels(1, 3, height=2.5)
        for a_, (pop, color) in enumerate(POPS):
            mean = np.mean(src[f"source_{pop}"], 0)
            se = np.std(src[f"source_{pop}"], 0, ddof=1) / np.sqrt(len(src[f"source_{pop}"]))
            for a, (slice_m, slice_t, title) in zip(ax[:2], ((np.diag(mean), np.diag(th["source"][a_]), r"$Q_a(\tau,\tau)$"),
                                                            (mean[zero], th["source"][a_][zero], r"$Q_a(\tau,0)$"))):
                a.plot(lags, slice_t, "-", color=color, lw=1.3)
                err = np.diag(se) if title.endswith(r"\tau)$") else se[zero]
                a.errorbar(lags[::2], slice_m[::2], yerr=err[::2], fmt="o", color=color, ms=2.8, elinewidth=.7)
                a.set_ylabel(title)
                a.set_xlabel(r"lag $\tau$")
            # what premature averaging discards: Q(t,t) versus the squared mean kernel
            ax[2].plot(lags, th["source"][a_].diagonal() / th["mean_kernel"][a_]**2, "-", color=color, lw=1.3,
                       label=f"{pop}")
        ax[2].axhline(1, color=GRAY, lw=.7, ls=(0, (4, 2)))
        ax[2].set_ylabel(r"$Q_a(\tau,\tau)\,/\,\langle c_a(\tau)\rangle^2$ (theory)")
        ax[2].set_xlabel(r"lag $\tau$")
        ax[2].set_xlim(-8, 8)
        ax[2].set_ylim(0, None)
        ax[2].legend()
        letters(ax)
        save(fig, "lag_sources",
             "The same-neuron source of covariance energy at Bernoulli dispersion, N = 2,000. Q_a(tau1,tau2) is the "
             "mean over neurons of the product of a neuron's own rate autocovariance at two lags, cross-fitted across "
             "recording halves. (a) Equal-lag slice and (b) tau2 = 0 slice, with predictions. (c) Predicted ratio of "
             "the source to the squared population-mean kernel; averaging spectra before taking the product replaces "
             "this ratio by one.")


def spectra(networks):
    fig, ax = panels(1, 3, height=2.6)
    sets = [("degree", "chi", (0., .1, .5, 1., 2., 3.), lambda v: v / 3, r"$\chi$", dict()),
            ("composition", "fraction_e", (.5, .6, .7, .8, .88, .94), lambda v: (v - .5) / .45, r"$f_E$", dict()),
            (("size", "degree"), "n", (500., 1000., 2000.), lambda v: np.log2(v / 500) / 2, r"$N$", dict(chi=1., fraction_e=.8))]
    for a, (family, key, values, cmap, label, where) in zip(ax, sets):
        for v in values:
            rows = fluctuating(select(networks, family=family, **{key: v}, **where))
            eig = []
            for r in rows:
                with arrays(r) as z:
                    if "eig_joint" in z.files:
                        e = z["eig_joint"].astype(float)
                        eig.append(e / e.sum() * len(e))
            if not eig:
                continue
            m = np.exp(np.mean(np.log(np.clip(eig, 1e-12, None)), 0))
            rank = (np.arange(len(m)) + 1) / len(m)
            a.plot(rank, m, "-", color=SEQ(.1 + .85 * cmap(v)), lw=1.1, label=f"{v:g}" if key != "n" else f"{int(v):,}")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_ylim(1e-3, None)
        a.set_xlabel("rank / N")
        a.legend(title=label, fontsize=6.5, title_fontsize=7.5, loc="lower left")
    ax[0].set_ylabel(r"eigenvalue / mean eigenvalue")
    letters(ax)
    save(fig, "covariance_spectra",
         "Eigenvalue spectra of the whole-record rate covariance (ordinary centering, 8,000 leak times), normalized "
         "to unit mean and plotted against normalized rank; geometric mean over up to eight networks. (a) Degree "
         "dispersion at N = 2,000. (b) Excitatory fraction at N = 2,000. (c) Network size at dispersion 1, 80:20. "
         "These spectra include the finite-record noise floor and are shown for "
         "shape only; the manuscript's dimension estimates use the cross-window estimator.")


def main():
    style()
    networks = load()
    for figure in (local_laws, degree_inputs, kernels, lag_maps, spectra):
        figure(networks)


if __name__ == "__main__":
    main()
