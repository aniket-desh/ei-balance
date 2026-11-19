"""Compute a smooth parameter-only theory curve for the nonuniform degree plot."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time
from predict_degree import predict,OUT,G,J0,F


def main():
    primary=json.loads((OUT/'predictions.json').read_text())
    rows=[]
    for chi in [0.,.025,.05,.1,.2,.3,.5,.75,1.,1.25,1.5,1.75,2.]:
        existing=next((r for r in primary['rows'] if r['chi']==chi),None)
        row=existing if existing is not None else predict(chi)[0]
        rows.append(row)
        (OUT/'theory_curve.json').write_text(json.dumps({'G':G.tolist(),'J0':J0.tolist(),
            'fractions':F.tolist(),'rows':rows,'source':'Parameter-only conditional DMFT and two-site closure, no network fit.'},indent=2)+'\n')
        print(json.dumps({'chi':chi,'d':row['prediction']['dimension_fraction']}),flush=True)


if __name__=='__main__':main()
