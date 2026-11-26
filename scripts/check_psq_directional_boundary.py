"""Validate compensated CUDA boundary caps against 80-digit arithmetic."""
import argparse
from decimal import Decimal,localcontext
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.check_psq_step_cap import evaluate_recorded,evaluate


def singular(F,D,t):
    z=[f+t*d for f,d in zip(F,D)]
    a=sum(z[i]*z[i] for i in range(3));c=sum(z[i]*z[i] for i in range(3,6));b=sum(z[i]*z[i+3] for i in range(3))
    disc=((a-c)**2+4*b*b).sqrt()
    return ((a+c+disc)/2).sqrt(),((a+c-disc)/2).sqrt()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,nargs='+',required=True)
    p.add_argument('--before-dll',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rng=np.random.default_rng(20261116)
    random_rows=[];rejected_inputs=0
    with localcontext() as ctx:
        ctx.prec=80;limit=Decimal.from_float(1.1)
        for i in range(600):
            u=np.linalg.qr(rng.normal(size=(3,2)))[0];v=np.linalg.qr(rng.normal(size=(2,2)))[0]
            high=1.1-rng.integers(1,64)*np.spacing(1.1);low=rng.uniform(.1,.9)
            F=(u*np.array([high,low]))@v.T
            D=rng.normal(size=(3,2))*10**rng.uniform(-8,1) if i%3 else -.01*F
            flat=F.T.ravel().tolist();df=D.T.ravel().tolist();ddf=list(map(Decimal.from_float,flat));ddd=list(map(Decimal.from_float,df))
            s=np.linalg.svd(F,compute_uv=False);norm=np.linalg.norm(D)
            if s[0]>=1.1 or singular(ddf,ddd,Decimal(0))[0]>=limit:
                rejected_inputs+=1;continue
            legacy=min(1.,.9*min(1.1-s[0],s[1]-1e-12)/norm)
            random_rows.append([legacy,0,s[0],s[1],norm]+flat+df)
        # Explicit inward/outward, rank-crossing, zero and curved spectral path.
        f=np.array([[1.1-8*np.finfo(float).eps,0],[0,.2],[0,0.]])
        special=[-.01*f,.01*f,-2*f,np.zeros_like(f),np.array([[-.001,0],[0,0],[.1,0]])]
        for d in special:
            s=np.linalg.svd(f,compute_uv=False);norm=np.linalg.norm(d)
            legacy=min(1.,.9*min(1.1-s[0],s[1]-1e-12)/norm) if norm else 1.
            random_rows.append([legacy,0,s[0],s[1],norm]+f.T.ravel().tolist()+d.T.ravel().tolist())
        actual={}
        all_rows=[]
        for path in a.audit:
            for record in map(json.loads,path.read_text().splitlines()):
                for face in record['faces']:
                    row=face['values'];all_rows.append(row)
                    if row[2]>=1.1-64*np.finfo(float).eps*1.1:
                        actual[str(path),record['frame'],record['iteration'],face['face']]=row
        rows=np.array(random_rows+list(actual.values()))
        caps=evaluate_recorded(rows);assert np.isfinite(caps).all() and (caps[:,1]>0).all() and (caps[:,1]<=1).all()
        min_gap=Decimal(1);min_rank=Decimal(1)
        for row,cap in zip(rows,caps[:,1]):
            F=list(map(Decimal.from_float,row[5:11]));D=list(map(Decimal.from_float,row[11:17]));t=Decimal.from_float(float(cap))
            start=singular(F,D,Decimal(0));end=singular(F,D,t)
            rank=start[1]-t*sum(d*d for d in D).sqrt()
            assert rank>Decimal('1e-12')
            assert start[0]<limit and end[0]<limit
            min_gap=min(min_gap,limit-start[0],limit-end[0]);min_rank=min(min_rank,rank)
            for k in range(1,20):
                ss=singular(F,D,t*Decimal(k)/20)
                assert ss[0]<limit and ss[1]>Decimal('1e-12')
        all_rows=np.array(all_rows)
        nonboundary=all_rows[all_rows[:,2]<1.1-64*np.finfo(float).eps*1.1]
        np.testing.assert_array_equal(evaluate_recorded(nonboundary),evaluate_recorded(nonboundary,a.before_dll.resolve()))
        invalid=np.array([[[1.1,0],[0,.5],[0,0]],[[1.,0],[0,0],[0,0]]])
        assert np.all(evaluate(invalid,np.zeros_like(invalid))==0)
        # Even a falsely rounded-down native smax must not admit an exact
        # boundary/over-boundary start in the compensated branch.
        bad=rows[:1].copy();bad[0,2]=np.nextafter(1.1,0);bad[0,3]=.5;bad[0,4]=.01
        bad[0,5:11]=[1.1,0,0,0,.5,0];bad[0,11:17]=[-.01,0,0,0,0,0]
        assert evaluate_recorded(bad)[0,1]==0
        result=dict(decimal_precision=80,random_and_special_cases=len(random_rows),excluded_infeasible_random_inputs=rejected_inputs,
            actual_unique_near_directions=len(actual),total_cases=len(rows),path_samples=21,
            strict_endpoint_and_sample_checks_passed=True,entire_segment_upper_bound_by_convexity=True,
            entire_segment_rank_bound_by_weyl=True,minimum_gap=str(min_gap),minimum_rank_bound=str(min_rank),
            nonboundary_recorded_exact_regressions=len(nonboundary),invalid_initial_states_rejected=True,
            limitation='Checks the exact linear F+tD path of recorded double inputs. Native vertex-coordinate rounding, all other faces, energy descent and contacts require simulation validation.')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
