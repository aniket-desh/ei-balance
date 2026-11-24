"""Fine parameter-only curves for the campaign figures (no network data enters).

    python experiments/paper_campaign/theory_curves.py   ->   theory_curves.json
"""
import json
from pathlib import Path

import numpy as np

import theory
from summarize import theory_row

HERE = Path(__file__).resolve().parent
CHI = [0, .0125, .025, .05, .075, .1, .15, .2, .25, .3, .4, .5, .625, .75, .875, 1, 1.125, 1.25, 1.375, 1.5,
       1.625, 1.75, 1.875, 2, 2.25, 2.5, 2.75, 3]
FRACTION = np.round(np.arange(.45, .9601, .0025), 4).tolist()
SCALE = [.4, .45, .5, .6, .7, .75, .85, 1, 1.15, 1.3, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25]
RATE = [.12, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7]


def rows(cells, controls=False):
    out = []
    for cell in cells:
        try:
            row = dict(cell, **theory_row(cell))
        except Exception as exc:                      # no resolved chaotic branch at this point
            print("skipped", cell, repr(exc)[:80])
            continue
        if controls:
            c = theory.averaging_controls(cell["scale"], cell["rate"], float(cell["chi"]),
                                          cell["fraction_e"] if cell.get("bernoulli") else None)
            for name in ("spectra", "gains"):
                row[f"d_{name}"] = c[name]["d"]
                row[f"overlap_{name}"] = c[name]["overlap"]
                row[f"dE_{name}"], row[f"dI_{name}"] = c[name]["d_within"]
        out.append(row)
    return out


def main():
    base = dict(scale=1., rate=.3)
    curves = {
        "degree": rows([dict(base, chi=c, fraction_e=.8) for c in CHI], controls=True),
        "composition": rows([dict(base, chi=1, fraction_e=f, bernoulli=True) for f in FRACTION]),
        "composition_controls": rows([dict(base, chi=1, fraction_e=f, bernoulli=True)
                                      for f in np.round(np.arange(.45, .961, .03), 3)], controls=True),
        "scale": rows([dict(chi=1, fraction_e=.8, scale=s, rate=.3) for s in SCALE], controls=True),
        "rate": rows([dict(chi=1, fraction_e=.8, scale=1., rate=m) for m in RATE], controls=True),
        "readout": [dict(fraction_e=f, **{f"R_{'EI'[a]}{k}": theory.readout_ratio(f, k)[a]
                                          for k in (25, 50, 100, 200) for a in range(2)}) for f in FRACTION],
    }
    (HERE / "theory_curves.json").write_text(json.dumps(curves) + "\n")
    print({k: len(v) for k, v in curves.items()})


if __name__ == "__main__":
    main()
