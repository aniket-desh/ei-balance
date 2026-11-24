"""Write the deterministic manifest for the larger paper campaign.

Families (all strict-Dale positive-erf networks, midpoint step 0.05 unless stated,
500 burn-in, sampled every 0.25, four-window mean-difference estimator, B = 500):

  degree       chi sweep at 80:20, N = 2000                     15 chi x 24 seeds
  size         N in {500, 1000, 4000, 8000} at chi in {0, .5, 1, 2} (80:20)
               and at 60:40 Bernoulli; N = 2000 comes from degree/composition
  composition  f_E sweep at fixed J0 F and G F, chi = 1, N = 2000   13 f x 48 matched seeds,
               with K-cell readouts, K in {25, 50, 100, 200}
  record       T = 32000 recordings (B = 500, 1000, 2000) plus step 0.025 and 0.125
               replays of the same graphs, at four parameter points   8 seeds each
  generality   coupling variance scaled by s, or target rate changed, at chi = 1, 80:20

Seed labels are new (41001-45016) and matched across settings within a family.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from theory import F0, G0, J0, solve

HERE = Path(__file__).resolve().parent
CHIS = (0, .025, .05, .1, .2, .3, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3)
FRACTIONS = (.50, .55, .60, .65, .70, .75, .80, .84, .86, .88, .90, .92, .94)
SIZE_SEEDS = {500: 48, 1000: 32, 4000: 12, 8000: 6}
SIZE_CHIS = (0, .5, 1, 2)
SCALES = (.5, .75, 1.5, 2., 3.)
RATES = (.15, .5)
BASE = dict(g=4.0, dt=.05, burn=500., duration=8000., sample_dt=.25)


def degree_config(n, seed, chi, scale=1., rate=.3):
    """Fixed G, J0 and f; the O(1) drive is the parameter-only h(chi)."""
    summary, _ = solve(scale, rate, float(chi))
    config = dict(BASE, n=n, seed=seed, fraction_e=.8, mean_rate=rate, ensemble="degree",
                  degree_dispersion=chi, variance_profile=(scale * G0).tolist(), mean_connectivity=J0.tolist())
    return config, summary["mean_inputs"]


def composition_config(n, seed, f):
    """Bernoulli edges; presynaptic columns rescaled so G diag(f) and J0 diag(f) stay fixed."""
    summary, _ = solve(1., .3, 1.)
    scale = F0 / np.array([f, 1 - f])
    config = dict(BASE, n=n, seed=seed, fraction_e=f, mean_rate=.3, ensemble="bernoulli",
                  variance_profile=(G0 * scale).tolist(), mean_connectivity=(J0 * scale).tolist())
    return config, summary["mean_inputs"]


def cost(config, lag=False, tangent=False):
    """Rough single-core seconds, calibrated on N = 2000, T = 8000 (about 100 s simulate, 45 s analyze)."""
    n, t = config["n"] / 2000, config["duration"]
    c = 100 * n**2 * (t + 500) / 8500 * .05 / config["dt"] + 45 * n**2.5 * t / 8000
    return c + (20 if lag else 0) + (47 * n**2 if tangent else 0)


def main():
    jobs = []

    def add(family, cell, index, config, mean_input, *, readouts=(100,), lag=False, spectrum=False,
            tangent=False, blocks=(500.,), prefixes=(4000., 8000.), tag=""):
        name = f"pc_{family}_{tag}N{config['n']}_seed{config['seed']}"
        extras = dict(readout_sizes=list(readouts), lag=lag, spectrum=spectrum and config["n"] <= 2000,
                      tangent=tangent)
        jobs.append(dict(name=name, family=family, cell=cell, seed_index=index, config=config,
                         mean_input=list(mean_input), blocks=list(blocks), prefixes=list(prefixes),
                         extras=extras, cost=cost(config, lag, tangent)))

    for chi in CHIS:
        for i in range(24):
            config, h = degree_config(2000, 41001 + i, chi)
            add("degree", dict(chi=chi, fraction_e=.8, scale=1., rate=.3), i, config, h,
                lag=i < 8, spectrum=i < 8, tangent=i < 6, tag=f"chi{chi:g}_")

    for n, seeds in SIZE_SEEDS.items():
        deep = {500: 6, 1000: 6, 4000: 2, 8000: 0}[n]
        for i in range(seeds):
            for chi in SIZE_CHIS:
                config, h = degree_config(n, 42001 + i, chi)
                add("size", dict(chi=chi, fraction_e=.8, scale=1., rate=.3), i, config, h,
                    readouts=(100,) if n >= 1000 else (), lag=i < 4, spectrum=i < 6,
                    tangent=i < deep, tag=f"chi{chi:g}_")
            config, h = composition_config(n, 42001 + i, .6)
            add("size", dict(chi=1, fraction_e=.6, scale=1., rate=.3, bernoulli=True), i, config, h,
                readouts=(100,) if n >= 1000 else (), lag=i < 4, spectrum=i < 6, tangent=i < deep,
                tag="f0.6_")

    for f in FRACTIONS:
        for i in range(48):
            config, h = composition_config(2000, 43001 + i, f)
            add("composition", dict(chi=1, fraction_e=f, scale=1., rate=.3, bernoulli=True), i, config, h,
                readouts=(25, 50, 100, 200), lag=i < 8, spectrum=i < 8, tangent=i < 6, tag=f"f{f:g}_")

    record_cells = [("chi0", lambda s: degree_config(2000, s, 0), dict(chi=0, fraction_e=.8)),
                    ("chi1", lambda s: degree_config(2000, s, 1), dict(chi=1, fraction_e=.8)),
                    ("chi2", lambda s: degree_config(2000, s, 2), dict(chi=2, fraction_e=.8)),
                    ("f0.6", lambda s: composition_config(2000, s, .6), dict(chi=1, fraction_e=.6, bernoulli=True))]
    for tag, make, cell in record_cells:
        for i in range(8):
            config, h = make(44001 + i)
            cell = dict(cell, scale=1., rate=.3)
            add("record", dict(cell, dt=.05, duration=32000.), i, dict(config, duration=32000.), h,
                readouts=(), lag=True, spectrum=True, blocks=(500., 1000., 2000.),
                prefixes=(2000., 4000., 8000., 16000., 32000.), tag=f"{tag}_T32000_")
            for dt in (.025, .125):
                add("record", dict(cell, dt=dt, duration=8000.), i, dict(config, dt=dt), h,
                    readouts=(), tag=f"{tag}_dt{dt:g}_")

    for i in range(16):
        for scale in SCALES:
            config, h = degree_config(2000, 45001 + i, 1, scale=scale)
            add("generality", dict(chi=1, fraction_e=.8, scale=scale, rate=.3), i, config, h,
                lag=i < 4, spectrum=i < 6, tangent=i < 6, tag=f"s{scale:g}_")
        for rate in RATES:
            config, h = degree_config(2000, 45001 + i, 1, rate=rate)
            add("generality", dict(chi=1, fraction_e=.8, scale=1., rate=rate), i, config, h,
                lag=i < 4, spectrum=i < 6, tangent=i < 6, tag=f"m{rate:g}_")

    assert len({j["name"] for j in jobs}) == len(jobs)
    data = json.dumps(jobs, indent=1) + "\n"
    (HERE / "jobs.json").write_text(data)
    by = {}
    for j in jobs:
        key = (j["family"], j["config"]["n"])
        by[key] = by.get(key, [0, 0.])
        by[key][0] += 1
        by[key][1] += j["cost"] / 3600
    for key, (count, hours) in sorted(by.items()):
        print(f"{key[0]:12s} N={key[1]:5d}  {count:4d} networks  ~{hours:6.1f} core-hours")
    print(f"{len(jobs)} networks, ~{sum(j['cost'] for j in jobs)/3600:.0f} core-hours, "
          f"manifest SHA-256 {hashlib.sha256(data.encode()).hexdigest()}")


def extension():
    """Added after the N <= 8000 size series showed the fixed in-degree error shrinking with N:
    one more octave, N = 16000, at dispersions 0, 1 and 2 (four networks each)."""
    jobs = []
    for chi in (0, 1, 2):
        for i in range(4):
            config, h = degree_config(16000, 42001 + i, chi)
            jobs.append(dict(name=f"pc_size_chi{chi:g}_N16000_seed{42001 + i}", family="size",
                             cell=dict(chi=chi, fraction_e=.8, scale=1., rate=.3), seed_index=i, config=config,
                             mean_input=list(h), blocks=[500.], prefixes=[4000., 8000.],
                             extras=dict(readout_sizes=[100], lag=True, spectrum=False, tangent=False),
                             cost=cost(config, True, False)))
    (HERE / "jobs_extension.json").write_text(json.dumps(jobs, indent=1) + "\n")
    print(f"{len(jobs)} extension networks, ~{sum(j['cost'] for j in jobs)/3600:.0f} core-hours")


if __name__ == "__main__":
    import sys
    extension() if "--extension" in sys.argv else main()
