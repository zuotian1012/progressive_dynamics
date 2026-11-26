"""High-precision local path check for an actual remaining cap bottleneck."""
import argparse
from decimal import Decimal,localcontext
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('audit',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    records=[json.loads(line) for line in a.audit.read_text().splitlines()]
    record=records[-1];face=record['faces'][1];v=face['values']
    with localcontext() as ctx:
        ctx.prec=80
        F=[Decimal.from_float(x) for x in v[5:11]];D=[Decimal.from_float(x) for x in v[11:17]]
        limit=Decimal.from_float(1.1);alpha=Decimal('.01')
        def singular(t):
            z=[f+t*d for f,d in zip(F,D)]
            aa=sum(z[i]*z[i] for i in range(3));cc=sum(z[i]*z[i] for i in range(3,6));bb=sum(z[i]*z[i+3] for i in range(3))
            discriminant=((aa-cc)**2+4*bb*bb).sqrt()
            return ((aa+cc+discriminant)/2).sqrt(),((aa+cc-discriminant)/2).sqrt()
        states=[singular(alpha*Decimal(k)/200) for k in range(201)]
        assert all(s<limit and t>Decimal('1e-12') for s,t in states)
        s0,t0=states[0];s1,t1=states[-1]
        norm=sum(d*d for d in D).sqrt()
        rank_lower_bound=t0-alpha*norm
        assert rank_lower_bound>Decimal('1e-12')
        # Convexity of the spectral norm bounds the whole segment using its
        # endpoints. Weyl gives an independent whole-segment rank bound.
        result=dict(frame=record['frame'],iteration=record['iteration'],face=face['face'],
            native_cap=v[1],checked_alpha=float(alpha),alpha_ratio=float(alpha/Decimal.from_float(v[1])),
            decimal_precision=ctx.prec,represented_double_limit=str(limit),
            initial_gap=str(limit-s0),endpoint_gap=str(limit-s1),
            original_margin_gap=str(Decimal(64)*Decimal.from_float(2.220446049250313e-16)*limit),
            endpoint_stretch_increased=bool(s1>s0),path_samples=201,
            entire_segment_upper_stretch_by_convexity=True,
            entire_segment_rank_lower_bound=str(rank_lower_bound),
            F=v[5:11],D=v[11:17],
            limitation='Local face only, using recorded F and dF and high-precision linear interpolation. Not a global cap, native position-update replay, contact/CCD test, or energy line-search acceptance. No larger step was applied to the simulation.')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
