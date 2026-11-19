# Focused EI manuscript

`main.tex` is the extracted paper source. It centers the conditional balanced
EI local-law-to-geometry construction, population averaging inequalities, and
the composition and degree interventions. The scalar crossover remains
supporting analytical work in the archive rather than a prerequisite for
this paper.

The current draft preserves the historical 60:40 discrepancy and distinguishes
it from the fresh prospective campaign. `claim_evidence.md` records which
claims are algebraic, conditional, or supported by completed network tests.
Author order, affiliations, and a submission venue are intentionally unset.

Build from the repository root. With TeX Live and latexmk installed, the saved figures and tables are sufficient to compile the manuscript.

```bash
latexmk -pdf -cd paper/main.tex
```

To regenerate the local-law figure, tables and paper from their numerical inputs, run the following from the repository root.

```bash
python paper/make_local_figure.py
python paper/make_campaign_tables.py
python paper/make_auxiliary_tables.py
python paper/build_manuscript.py
```

The last command runs the usual pdflatex/bibtex sequence on a complete TeX
Live installation. In this minimal Linux runtime it can build a private
format, search database, and font map under `tmp/ei_manuscript_tex` without
changing system TeX files. These regenerable build files are not part of the
research archive. Compiling the paper requires the figures and the analytics
appendix included alongside the source; it does not require rerunning the
network simulations.

The table generator reads the completed composition, degree, and calibrated
gain summaries and records their hashes in `table_provenance.json`. Its
default completion guard requires all 40 composition and all 24 degree outcomes.
The development option `--allow-incomplete` is reserved for clearly labeled
in-progress drafts; the final paper uses the guarded output. The campaign is
staged: later allocations are recorded before their own outcomes, after the
earlier pilot results. It is not described as 64 fully preregistered networks.

The degree and composition plots also read the separate 81-network manifest
and its per-network measurements under `experiments/dense_sweeps/`. To redraw
those plots and refresh the paper storyboard, run these commands from the
repository root, then rebuild the manuscript.

```bash
python experiments/degree/aggregate_degree.py
python experiments/figures/make_campaign_figures.py
python figures/population_pr.py
python figures/pr_controls.py
python docs/refresh_figures.py
```

The parameter-only local-law and averaging figure is regenerated from the
archived conditional solver output. Its three approximations retain the
original local-law numerator; they are not new self-consistent network
ensembles. `figures/conditional_averaging_values.json` contains the full
population energy matrices and numerical checks used by the plot.

`literature_verification.md` gives the primary sources checked for the
specific claims used by this draft. It does not establish an exhaustive
priority claim. `build_validation.json` records typesetting checks;
scientific validation is provided separately by the analytic and campaign
outputs.
