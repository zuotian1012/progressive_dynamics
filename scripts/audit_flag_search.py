"""Recompute interpolation proposals and check actual full-energy trial records."""
import argparse,json,re
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    trials={};shrinks=[]
    for line in a.run.with_suffix('.log').read_text(encoding='utf-8-sig',errors='replace').splitlines():
        if 'FLAG_SEARCH_TRIAL ' not in line and 'FLAG_SEARCH_SHRINK ' not in line:continue
        raw=dict(re.findall(r'(\w+)=([^\s]+)',line));d={k:(v=='true' if v in ('true','false') else float(v)) for k,v in raw.items()}
        key=(int(d['frame']),int(d['newton']),int(d['search']))
        if 'FLAG_SEARCH_TRIAL ' in line:assert key not in trials;trials[key]=d
        else:shrinks.append((key,d))
    interpolated=[]
    for key,d in shrinks:
        t=trials[key];assert t['E']>t['E0'];np.testing.assert_allclose(d['delta'],t['E']-t['E0'],rtol=1e-12,atol=0)
        alpha=d['alpha'];expected=.5*alpha
        if d['previous_alpha']>alpha:
            previous=trials[(key[0],key[1],key[2]-1)]
            assert d['previous_alpha']==previous['alpha']
            np.testing.assert_allclose(d['previous_delta'],previous['E']-previous['E0'],rtol=1e-12,atol=0)
            q=(d['previous_delta']/d['previous_alpha']-d['delta']/alpha)/(d['previous_alpha']-alpha)
            g=d['delta']/alpha-q*alpha
            if np.isfinite(q) and np.isfinite(g) and q>0 and g<0:expected=np.clip(-g/(2*q),.1*alpha,.5*alpha)
        np.testing.assert_allclose(d['next'],expected,rtol=1e-12,atol=0)
        assert .1*alpha<=d['next']<=.5*alpha
        following=trials.get((key[0],key[1],key[2]+1))
        if following:
            assert following['alpha']==d['next']
            if d['next']<.5*alpha and following['E']<=following['E0']:
                interpolated.append(dict(frame=key[0],newton=key[1],search=key[2]+1,alpha=following['alpha'],energy_drop=following['E0']-following['E']))
    groups={}
    for key,t in trials.items():
        assert 0<=key[2]<8 and 0<t['alpha']<=t['cap']
        groups.setdefault(key[:2],[]).append(key[2])
    for searches in groups.values():assert sorted(searches)==list(range(len(searches))) and len(searches)<=8
    assert interpolated
    result=dict(passed=True,newton_searches=len(groups),maximum_trials=max(map(len,groups.values())),
        total_trials=len(trials),proposals_verified=len(shrinks),interpolated_following_trials_with_energy_decrease=len(interpolated),examples=interpolated[:10],
        scope='Diagnostic non-descent energy probes are separate; native convergence semantics are unchanged')
    (a.run/'line_search_audit.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
