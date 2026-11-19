"""Calibrated local-law explanation of finite-network static heterogeneity."""
from pathlib import Path
import json,numpy as np

HERE=Path(__file__).resolve().parent

def main():
    primary=json.loads((HERE/"predictions.json").read_text())["rows"][0]
    records=[]
    for name in ["degree_chi0_N1000_seed4001","degree_chi0_N2000_seed4001","degree_chi0_N4000_seed4001"]:
        path=HERE/"ingredient_diagnostics"/(name+".json")
        if not path.exists():continue
        r=json.loads(path.read_text());row=r["rows"][0]
        D=np.array(row["Delta_crossblock_input_means"])
        Vq=np.array(row["q_variance_across_neurons_fourwindow"])
        cov=np.array(row["input_mean_temporal_q_covariance_threewindow"])
        actual=np.array(row["static_rate_mean_variance_crossblock"])
        comparisons=[]
        backgrounds=[
            ("leading_chi0",np.array(primary["mean_inputs"]),np.array(primary["dynamic_variances"])),
            ("measured_population_background",np.array([x["h"] for x in r["raw_full_record_population_local_stats"]]),np.array(row["q_crossblock_temporal_input"]))
        ]
        for label,h,q in backgrounds:
            alpha=np.sqrt(2/(np.pi*(1+q)))*np.exp(-h*h/(2*(1+q)))
            beta=-h/(1+q)*alpha
            parts={"input_mean_disorder":alpha**2*D,
                   "mu_q_covariance":alpha*beta*cov,
                   "q_heterogeneity":beta**2/4*Vq}
            total=sum(parts.values())
            comparisons.append({"background":label,"h":h.tolist(),"q":q.tolist(),
                                "alpha":alpha.tolist(),"beta":beta.tolist(),
                                "parts":{k:v.tolist() for k,v in parts.items()},
                                "predicted_m_variance":total.tolist(),
                                "measured_m_variance":actual.tolist(),
                                "relative_residual":(total/actual-1).tolist()})
        records.append({"network":name,"comparison":comparisons})
    result={"scope":"Calibrated small-spread Gaussian local-rate expansion with measured Delta, Varq and Covmuq. It does not predict those missing finite-N quantities from parameters, nor supply a controlled correction.",
            "records":records}
    (HERE/"local_variance_feedback_diagnostic.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
