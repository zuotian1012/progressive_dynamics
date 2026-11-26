"""Independently verify each refined probe using logged full-energy samples."""
import argparse,json,re
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    pending={};events=[]
    for line in a.run.with_suffix('.log').read_text(encoding='utf-8-sig',errors='replace').splitlines():
        if 'FLAG_PROBE_REFINE ' not in line and 'FLAG_PROBE_REFINED ' not in line:continue
        d=dict(re.findall(r'(\w+)=([^\s]+)',line));key=(int(d['frame']),int(d['newton']),int(d['refinement']))
        assert 1<=key[2]<=3
        if 'FLAG_PROBE_REFINE ' in line:
            slopes=np.array([float(d[f'old_slope{i}']) for i in (1,2,3)])
            assert np.all(slopes>0) and np.ptp(slopes)>.25*slopes.max()
            pending[key]=d;continue
        before=pending.pop(key);h=np.array([float(d[f'h{i}']) for i in (1,2,3)])
        e=np.array([float(d[f'e{i}']) for i in (1,2,3)]);base=float(d['E0'])
        np.testing.assert_allclose(h,float(before['old_h1'])*np.array([.1,.01,.001]),rtol=1e-14,atol=0)
        slopes=(e-base)/h;noise=64*np.finfo(float).eps*max(abs(base),abs(e).max(),1e-30)
        resolved=bool(np.all(e-base>noise));consistent=resolved and np.ptp(slopes)<=.25*slopes.max()
        assert resolved==(d['resolved']=='true') and consistent==(d['consistent']=='true')
        np.testing.assert_allclose(slopes[:2],[float(before['old_slope2']),float(before['old_slope3'])],rtol=1e-12,atol=0)
        events.append(dict(frame=key[0],newton=key[1],refinement=key[2],h=h.tolist(),slopes=slopes.tolist(),energy_noise=noise,resolved=resolved,consistent=bool(consistent)))
    assert not pending and events
    result=dict(passed=True,refinement_events=len(events),consistent_events=sum(e['consistent'] for e in events),
        unresolved_events=sum(not e['resolved'] for e in events),unchanged_noise_multiplier=64,unchanged_relative_slope_limit=.25,events=events)
    (a.run/'refined_probe_audit.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='events'})

if __name__=='__main__':main()
