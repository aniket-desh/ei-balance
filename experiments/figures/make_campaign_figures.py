"""Reproduce the focused campaign figures from completed network records.

Each point is one independently generated network. Error bars are network SE,
not the spread of neurons, temporal blocks, or overlapping quadruples.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from paper_style import (COMPOSITION, CROSS, E, I, JOINT, LIGHT, MUTED, SIZE_MARKERS,
                         apply_style, panel_label, save_figure)

apply_style()


def records():
    from sweep_records import figure_record

    out = []
    for path in sorted((ROOT / 'next_round/runtime/runs').glob('*.json')):
        row = json.loads(path.read_text())
        if row.get('status') != 'complete':
            continue
        if row['job'].get('experiment') not in {'composition', 'degree'}:
            continue
        if any(e['block_duration'] == 500 for e in row['estimators']):
            out.append(figure_record(row))
    return out


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=LIGHT, linestyle=':', linewidth=.55)
    ax.set_axisbelow(True)


def point(ax, x, values, color, label=None, marker='o', offset=0.):
    values = np.asarray(values, float)
    if not len(values):
        return
    jit = np.linspace(-.025, .025, len(values)) if len(values) > 1 else [0.]
    ax.scatter(x + offset + np.asarray(jit), values, s=12,
               marker=marker, color=color, alpha=.32, edgecolors='none', zorder=3)
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.
    ax.errorbar(x + offset, values.mean(), yerr=se, fmt=marker,
                color=color, capsize=2.6, markersize=5.2, elinewidth=1.2,
                markeredgewidth=.8, label=label, zorder=4)


def degree(rows):
    theory = json.loads((ROOT / 'next_round/degree/predictions.json').read_text())['rows']
    rows = [r for r in rows if r['config']['ensemble'] == 'degree']
    if not rows:
        return
    chis = [r['chi'] for r in theory]
    sizes = sorted({r['config']['n'] for r in rows})
    fig, axs = plt.subplots(2, 3, figsize=(10.8, 6.5), layout='constrained')
    axes = axs.ravel()
    axes[0].plot(chis, [r['prediction']['dimension_fraction'] for r in theory],
                 color=JOINT, label='Conditional theory')
    for a in range(2):
        style = '-' if a == 0 else '--'
        axes[1].plot(chis, [r['static_variances'][a] for r in theory],
                     color=(E,I)[a], ls=style, label='E' if a == 0 else 'I')
        axes[2].plot(chis, [r['dynamic_variances'][a] for r in theory], color=(E,I)[a], ls=style)
    for a, b, ls, lab in [(0, 0, '-', 'EE'), (0, 1, '--', '2EI'), (1, 1, ':', 'II')]:
        factor = 2 if a != b else 1
        axes[3].plot(chis, [factor*r['block_energies'][a][b] for r in theory],
                     color={(0,0):E,(0,1):CROSS,(1,1):I}[a,b], ls=ls, label=lab)
    for j, key in [(4, 'uncorrelated_dimension_fraction'), (5, 'correlation_reduction_factor')]:
        axes[j].plot(chis, [r['decomposition'][key] for r in theory], color=JOINT)
    for ni, n in enumerate(sizes):
        marker = SIZE_MARKERS[n]
        offset = (ni-(len(sizes)-1)/2)*.06
        for chi in chis:
            rr = [r for r in rows if r['config']['n'] == n
                  and r['config']['degree_dispersion'] == chi]
            if not rr:
                continue
            point(axes[0], chi, [r['primary']['d'] for r in rr], MUTED,
                  f'N={n}' if chi == chis[0] else None, marker=marker, offset=offset)
            for a in range(2):
                point(axes[1], chi, [r['local'][a]['Delta_input_means'] for r in rr],
                      (E,I)[a], marker=marker, offset=offset)
                point(axes[2], chi, [r['local'][a]['q_temporal_input'] for r in rr],
                      (E,I)[a], marker=marker, offset=offset)
            for a, b, color in [(0, 0, E), (0, 1, CROSS), (1, 1, I)]:
                factor = 2 if a != b else 1
                point(axes[3], chi,
                      [factor*r['primary']['population_energy'][a][b] for r in rr],
                      color, marker=marker, offset=offset)
            unc, corr = [], []
            for r in rr:
                f = np.array([r['config']['fraction_e'], 1-r['config']['fraction_e']])
                diag = f @ [s['mean_v_squared'] for s in r['local']]
                unc.append(r['primary']['nu']**2/diag)
                corr.append(diag/r['primary']['energy'])
            point(axes[4], chi, unc, MUTED, marker=marker, offset=offset)
            point(axes[5], chi, corr, MUTED, marker=marker, offset=offset)
    labels = [r'$d=D/N$', r'$\Delta_a$', r'$q_a$', 'Block energy per neuron',
              r'$d_{\rm uncorr}$', r'$\mathcal{E}_{\rm diag}/\mathcal{E}$']
    for ax, letter, ylabel in zip(axes, 'ABCDEF', labels):
        clean(ax)
        ax.set(xlabel=r'Degree dispersion $\chi$', ylabel=ylabel)
        panel_label(ax, letter)
        ax.set_xticks(chis)
    axes[0].legend(fontsize=8);axes[1].legend(fontsize=8);axes[3].legend(fontsize=8)
    save_figure(fig,HERE/'nonuniform_degree')


def composition_figure(rows, *, appendix):
    reference = json.loads((ROOT / 'next_round/composition/composition_theory_targets.json').read_text())['one_site_reference']
    rows = [r for r in rows if r['config']['ensemble'] == 'bernoulli']
    if not rows:
        return
    labels = [r'Mean rate variance $\nu_N/\nu_0$', r'Covariance energy $\mathcal{E}_N/\mathcal{E}_0$',
              r'Dimension $d_N/d_0$']
    if appendix:
        labels += [r'E gain $A_{E,N}/A_{E,0}$', r'I gain $A_{I,N}/A_{I,0}$',
                   r'Population-axis fraction $\mathcal{E}_{\mathrm{pop}}/\mathcal{E}$']
    else:
        labels += ['E–I temporal alignment']
    ncols = 3 if appendix else 2
    fig, axs = plt.subplots(2, ncols, figsize=(3.45*ncols, 5.0), layout='constrained')
    axes = axs.ravel()
    sizes = sorted({r['config']['n'] for r in rows})
    for fe, offset, marker in [(.6, -.065, 'o'), (.8, .065, 'D')]:
        color = COMPOSITION[fe]
        f = np.array([fe, 1-fe])
        theory = reference['predictions'][str(fe)]
        nu0 = f @ theory['mean_variances']
        for n in sizes:
            selected = [r for r in rows if r['config']['n'] == n and r['config']['fraction_e'] == fe]
            if not selected:
                continue
            x = np.log2(n/1000) + offset
            for ax, key, denominator in zip(axes, ['nu','energy','d'],
                    [nu0,theory['covariance_energy'],theory['dimension_fraction']]):
                point(ax, x, [r['primary'][key]/denominator for r in selected], color, marker=marker)
            if appendix:
                for pop in (0,1):
                    point(axes[3+pop], x,
                          [r['local'][pop]['A_timeaveraged_gain_squared']/reference['alpha_second'][pop]
                           for r in selected], color, marker=marker)
                point(axes[5], x, [r['primary']['coherent_projection_fraction'] for r in selected],
                      color, marker=marker)
            else:
                blocks = np.array([r['primary']['population_energy'] for r in selected])
                overlap = blocks[:,0,1] / np.sqrt(blocks[:,0,0]*blocks[:,1,1])
                point(axes[3], x, overlap, color, marker=marker)
        if not appendix:
            block = np.array(theory['pair_covariance_energy']) + np.diag(theory['variance_seconds']/f)
            axes[3].axhline(block[0,1]/np.sqrt(block[0,0]*block[1,1]), color=color, ls='--', lw=1.25)
    for j, (ax,label) in enumerate(zip(axes,labels)):
        if j < 3 or (appendix and j < 5):
            ax.axhline(1, color=MUTED, ls='--', lw=1.15)
        ax.set(xlabel=r'Network size $N$', ylabel=label, xlim=(-.20, np.log2(max(sizes)/1000)+.20))
        ax.set_xticks([np.log2(n/1000) for n in sizes], [f'{n:,}' for n in sizes])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        panel_label(ax, 'ABCDEF'[j])
    handles = [Line2D([], [], color=COMPOSITION[f], marker=m, ls='none', markersize=5,
                      label=l) for f,m,l in [(.6,'o','60:40 E:I'),(.8,'D','80:20 E:I')]]
    handles.append(Line2D([],[],color=MUTED,ls='--',label='Conditional theory'))
    fig.legend(handles=handles, loc='outside lower center', ncol=3)
    save_figure(fig,HERE / ('composition_validation' if appendix else 'composition_main'))


def composition(rows):
    composition_figure(rows, appendix=True)


def composition_main(rows, dense=None):
    from sweep_records import (cohort_handles, composition_moments, dense_records,
                               draw_group, grouped_networks, point_offsets)

    if dense is None:
        dense = dense_records()
    if not dense:
        return composition_figure(rows, appendix=False)
    archived = [r for r in rows if r['job']['experiment'] == 'composition']
    fresh = [r for r in dense if r['job']['experiment'] == 'composition']
    groups = grouped_networks(archived, fresh)
    offsets = point_offsets(groups, .009)
    tensor, means, _ = composition_moments()

    def prediction(fe):
        fractions = np.array([fe, 1-fe])
        blocks = np.einsum("abc,c->ab", tensor, 1/fractions)
        nu = fractions @ means
        energy = fractions @ blocks @ fractions
        return np.array([nu, energy, nu**2/energy,
                         blocks[0,1]/np.sqrt(blocks[0,0]*blocks[1,1])])

    reference = json.loads((ROOT / 'next_round/composition/composition_theory_targets.json').read_text())
    for fraction, theory in reference['one_site_reference']['predictions'].items():
        expected = [theory['covariance_energy'], theory['dimension_fraction']]
        np.testing.assert_allclose(prediction(float(fraction))[1:3], expected, rtol=2e-6)
    fractions = sorted({key[0] for key in groups})
    curve_f = np.linspace(min(fractions), max(fractions), 201)
    fig, axes = plt.subplots(2, 2, figsize=(7.7, 5.25), layout="constrained")
    axes = axes.ravel()
    for ax in axes[:3]:
        ax.axhline(1., color=MUTED, linestyle="--", lw=1.2)
    axes[3].plot(curve_f, [prediction(x)[3] for x in curve_f], color=JOINT)
    for key, networks in groups.items():
        fe, n, cohort = key
        pred = prediction(fe)
        values = []
        for record in networks:
            primary = record['primary']
            energy = np.asarray(primary['population_energy'])
            overlap = energy[0,1] / np.sqrt(energy[0,0]*energy[1,1])
            values.append([primary['nu']/pred[0], primary['energy']/pred[1],
                           primary['d']/pred[2], overlap])
        for ax, column in zip(axes, np.asarray(values).T):
            draw_group(ax, fe+offsets[key], column, SIZE_MARKERS[n], JOINT, cohort, jitter=.0013)
    labels = [r'Mean rate variance $\nu_N/\nu_0$', r'Covariance energy $\mathcal{E}_N/\mathcal{E}_0$',
              r'Dimension $d_N/d_0$', 'E–I temporal alignment']
    for ax, letter, label in zip(axes, 'ABCD', labels):
        ax.set(xlabel=r'Excitatory fraction $f_E$', ylabel=label,
               xlim=(min(fractions)-.018, max(fractions)+.014))
        ax.set_xticks([.6,.7,.8,.88,.94])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        panel_label(ax, letter)
    handles = [Line2D([], [], color=JOINT, label="Theory")]
    handles += cohort_handles({n for _, n, _ in groups})
    fig.legend(handles=handles, loc="outside lower center", ncol=3, handlelength=1.5,
               columnspacing=1.5, labelspacing=.25)
    save_figure(fig, HERE / 'composition_main')


if __name__ == '__main__':
    rows = records()
    degree(rows);composition(rows);composition_main(rows)
    (HERE / 'campaign_figure_values.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps({'complete_networks': len(rows), 'figures': str(HERE)}))
