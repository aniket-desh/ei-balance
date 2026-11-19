"""Sensitivity to degree rounding, reimposing the same leading population means.

The recomputed h differs from the primary protocol, so this is a diagnostic,
not an exact finite-N prediction for the already specified drives.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from predict_degree import G,F,OUT,chi_effective
from eicavity.multitype import solve_populations,population_dimension

def main():
    primary=json.loads((OUT/"predictions.json").read_text())
    rows=[]
    for chi in [0.,.5]:
        for n in [1000,2000,4000]:
            eff=np.array(chi_effective(n,chi))
            one=solve_populations(G,fractions=F,degree_dispersion=eff)
            pair=population_dimension(one)
            base=next(r for r in primary["rows"] if r["chi"]==chi)
            rows.append({"n":n,"chi":chi,"effective":eff.tolist(),
                         "h_retargeted_mean":one.mean_inputs.tolist(),
                         "h_difference":(one.mean_inputs-np.array(base["mean_inputs"])).tolist(),
                         "static_variances":one.static_variances.tolist(),
                         "dimension_retargeted_mean":pair["dimension_fraction"],
                         "dimension_relative_difference":pair["dimension_fraction"]/base["prediction"]["dimension_fraction"]-1})
    record={"scope":"Diagnostic conditional theory using exact expected degree dispersion but reimposing target populationmeans. This changes h slightly and is not the parameter-only prediction for the already specified finite-N drives. The O(1/M) Var(m) factor is not included.",
            "rows":rows}
    (OUT/"rounding_diagnostic.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))

if __name__=="__main__":main()
