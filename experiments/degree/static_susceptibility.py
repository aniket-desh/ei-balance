"""Check the regular nonuniform static susceptibility at zero degree disorder."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
import json,numpy as np
from predict_degree import G,F,OUT
from eicavity.multitype import solve_populations

def main():
    base=json.loads((OUT/"predictions.json").read_text())["rows"][0]
    W=G@np.diag(F)
    local=np.diag(base["alpha_second"])
    source=W@np.array([.3**2,.3**2])
    susceptibility=np.linalg.solve(np.eye(2)-W@local,source)
    checks=[]
    for chi in [1e-3,1e-4,1e-5]:
        one=solve_populations(G,fractions=F,degree_dispersion=chi)
        checks.append({"chi":chi,"Delta":one.static_variances.tolist(),
                       "Delta_over_chi":(one.static_variances/chi).tolist(),
                       "relative_error":(one.static_variances/chi/susceptibility-1).tolist()})
    record={"formula":"(I-W diag(A0))^-1 W mbar^2",
            "source":source.tolist(),"static_response_matrix":(W@local).tolist(),
            "response_eigenvalues":np.linalg.eigvals(W@local).tolist(),
            "susceptibility":susceptibility.tolist(),"checks":checks}
    (OUT/"static_susceptibility.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))

if __name__=="__main__":main()
