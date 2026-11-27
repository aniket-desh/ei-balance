"""Write the deterministic manifest for the composition-optimum validation (jobs.json, replay.json).

    python experiments/composition_optimum/validation/prepare.py

Cohort: three target-rate pairs x ten excitatory fractions at N = 2,000, 4,000 and 8,000, plus the
reference pair at N = 16,000, on the
composition path of the paper campaign (presynaptic columns of G0 and J0 rescaled by f0_b / f_b,
Bernoulli edges, midpoint step 0.05, 500 burn-in, 8,000 recorded, sampled every 0.25,
four-window mean-difference estimator with window 500). The order-one drive of each rate
pair is the parameter-only mean input from predictions.json. Every network has its own seed
label, 50001 upward, so networks in different cells are independent.

The reference pair uses prepare.composition_config of the campaign unchanged (scalar target
rate 0.3); the two unequal pairs pass mean_rate as [m_E, m_I], which make_network uses
elementwise in the balance drive -sqrt(N) J0 (f * m) + h (validate.py checks this).

replay.json holds one archived campaign job, rerun on a pod to check this copy of the runner.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1] / "paper_campaign"
sys.path.insert(0, str(CAMPAIGN))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("campaign_prepare", CAMPAIGN / "prepare.py")
campaign = importlib.util.module_from_spec(_spec)   # loaded by path: this file has the same module name
_spec.loader.exec_module(campaign)

PRED = json.loads((HERE / "predictions.json").read_text())
FRACTIONS = PRED["fractions"]
PER_CELL = {2000: 64, 4000: 16, 8000: 16, 16000: 4}
RATES_AT = {16000: ("ref",)}          # the largest size extends the reference pair only
REPLAY = "pc_composition_f0.86_N2000_seed43020"


def config_for(key, n, seed, f):
    if key == "ref":
        config, h = campaign.composition_config(n, seed, f)
        assert np.allclose(h, PRED["rate_pairs"]["ref"]["mean_inputs"], rtol=0, atol=1e-9)
        return config, list(h)
    rates = PRED["rate_pairs"][key]["rates"]
    config, _ = campaign.composition_config(n, seed, f)
    return dict(config, mean_rate=list(rates)), list(PRED["rate_pairs"][key]["mean_inputs"])


def main():
    jobs = []
    seed = 50001
    for n, per in PER_CELL.items():
        for key in RATES_AT.get(n, ("ref", "lo", "hi")):
            for f in FRACTIONS:
                for i in range(per):
                    config, h = config_for(key, n, seed, f)
                    jobs.append(dict(name=f"cv_{key}_f{f:g}_N{n}_seed{seed}", family="validation",
                                     cell=dict(rates=key, mean_rates=PRED["rate_pairs"][key]["rates"], fraction_e=f, n=n,
                                               chi=1, bernoulli=True),
                                     seed_index=i, config=config, mean_input=h, blocks=[500.], prefixes=[4000., 8000.],
                                     extras=dict(readout_sizes=[], lag=False, spectrum=False, tangent=False),
                                     cost=campaign.cost(config)))
                    seed += 1
    assert len({j["name"] for j in jobs}) == len(jobs) and len({j["config"]["seed"] for j in jobs}) == len(jobs)
    data = json.dumps(jobs, indent=1) + "\n"
    (HERE / "jobs.json").write_text(data)
    replay = [j for j in json.loads((CAMPAIGN / "jobs.json").read_text()) if j["name"] == REPLAY]
    (HERE / "replay.json").write_text(json.dumps(replay, indent=1) + "\n")
    by = {}
    for j in jobs:
        k = j["config"]["n"]
        by.setdefault(k, [0, 0.])
        by[k][0] += 1
        by[k][1] += j["cost"] / 3600
    for k, (c, h) in sorted(by.items()):
        print(f"N={k:5d}  {c:4d} networks  ~{h:6.1f} core-hours (campaign cost model)")
    print(f"{len(jobs)} networks, seeds {jobs[0]['config']['seed']}-{jobs[-1]['config']['seed']}, "
          f"manifest SHA-256 {hashlib.sha256(data.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
