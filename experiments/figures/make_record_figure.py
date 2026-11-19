"""Paired network-level centering and gap diagnostics; no covariance fitting."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from paper_style import JOINT, MUTED, apply_style, panel_label, save_figure

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / 'archive/research_round/record'
protocol = json.loads((SOURCE/'cohort_protocol.json').read_text())
rows = [json.loads((SOURCE/'cohort'/f'seed{seed}.json').read_text()) for seed in protocol['seeds']]
assert all(r['status'] == 'complete' for r in rows)


def get(record, block, gap=1):
    blocks = next(v for v in record['estimator_results'] if v['block_duration'] == block)
    prefix = next(v for v in blocks['prefixes'] if v['record_duration'] == 16000)
    return next(v for v in prefix['gaps'] if v['min_empty_blocks_between_all_windows'] == gap)


def mean_ci(values):
    values = np.asarray(values)
    return values.mean(), t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))


apply_style()
fig, (a,b) = plt.subplots(1, 2, figsize=(7.45,2.7), layout='constrained')
blocks = [500,1000,2000]
diffs = [mean_ci([get(r,B)['classic_d']-get(r,B)['d_mean_free_ratio'] for r in rows]) for B in blocks]
a.errorbar(blocks, *np.asarray(diffs).T, fmt='o-', capsize=3, color=JOINT, ms=4.5)
a.set(xscale='log', xlabel=r'Covariance block duration $B$', ylabel=r'$d_{\mathrm{own}}-d_{\mathrm{four}}$')
a.set_xticks(blocks, ['500','1,000','2,000'])
a.minorticks_off()
panel_label(a, 'A')
gaps = [0,1,2]
diffs = [mean_ci([get(r,500,g)['d_mean_free_ratio']-get(r,500,1)['d_mean_free_ratio']
                  for r in rows]) for g in gaps]
b.errorbar([500*g for g in gaps], *(1e6*np.asarray(diffs).T), fmt='s-', capsize=3, color=JOINT, ms=4.5)
b.set(xlabel='Gap between windows', ylabel=r'Change in $d_{\mathrm{four}}$ ($\times\,10^{-6}$)')
b.set_xticks([0,500,1000], ['0','500','1,000'])
panel_label(b, 'B')
for ax in (a,b):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.axhline(0, color=MUTED, ls='--', lw=1., zorder=0)
save_figure(fig,HERE/'finite_record_diagnostics')
(HERE/'figure_caption.txt').write_text('Finite-record estimator diagnostics on the prespecified sixteen Bernoulli60:40 networks, N1000, each extended toT16000. Error bars are95% Student-t intervals of paired differences across connectivity realizations. A: the standard separately centered covariance-block ratio exceeds the mean-free four-window ratio at the same block duration; increasing record length at fixed block duration does not remove that centering effect. B: after the mean-free repair, changing the minimum empty gap from0 to1000 atB500 changes meanPR by only a few parts in10^6 in this cohort. The scale differs strongly between panels. Overlapping quadruples are not treated as independent observations. These diagnostics support this coupling point only, not a uniform stationary-limit or critical-scaling theorem.\n')
