"""Recompute three-scale evidence for shirt-aligned coarse preview stops."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from paper_solver_settings import validate_shirt_solver_history


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    m=json.loads((a.run/'run.json').read_text());validate_shirt_solver_history(m)
    assert m['status']!='running'
    rows=json.loads((a.run/'steps.json').read_text());events={};rejected=0
    derivative_comparisons=[]
    fallback_events={}
    with a.run.with_suffix('.log').open(encoding='utf-8-sig',errors='replace') as log:
        for line in log:
            if 'BUNNY_EVENT_TRIGGER ' in line:
                assert m.get('event_fallback_enabled')
                event=dict(re.findall(r'(\w+)=([^\s]+)',line));frame=int(event['frame'])
                assert frame not in fallback_events
                base,retained=float(event['E0']),float(event['retained'])
                assert np.isfinite([base,retained]).all()
                assert abs(base-retained)<=64*np.finfo(float).eps*max(abs(base),1e-30)
                fallback_events[frame]=event
                events.pop(frame,None)
                continue
            assert 'PSQ_FORCE_ACCEPT frame=' not in line
            assert 'PSQ_PREVIEW_SMALL_STEP frame=' not in line
            rejected+=int('PSQ_PREVIEW_REJECT frame=' in line)
            if m.get('frozen_direction_derivative'):
                if 'FLAG_SEARCH_TRIAL ' in line:
                    probe_step=float(re.search(r'\bcap=(\S+)',line)[1])*1e-5
                elif 'FLAG_PROBE_REFINED ' in line:
                    probe_step=float(re.search(r'\bh3=(\S+)',line)[1])
                elif 'PSQ_PREVIEW_REJECT ' in line:
                    probe=dict(re.findall(r'(\w+)=(\S+)',line))
            if 'BUNNY_PREVIEW_STAGNATION frame=' in line:
                assert m.get('representable_preview') and m.get('frozen_offset_only') and m['level']<7
                raw=dict(re.findall(r'(\w+)=([^\s]+)',line))
                frame=int(raw['frame'])
                base,retained=float(raw['E0']),float(raw['retained'])
                assert frame not in events and int(raw['vertices'])>=m['vertices'] and float(raw['alpha'])>0
                assert np.isfinite([base,retained]).all() and abs(retained-base)<=64*np.finfo(float).eps*max(abs(base),1e-30)
                events[frame]=dict(reason='representable_exhaustion',newton=int(raw['newton']))
                continue
            if 'BUNNY_PREVIEW_DERIVATIVE frame=' in line:
                assert m.get('frozen_direction_derivative') and (m.get('frozen_offset_only') or m.get('event_fallback_enabled'))
                raw=dict(re.findall(r'(\w+)=([^\s]+)',line))
                frame=int(raw['frame'])
                slope,magnitude,bound,base,retained=map(float,(raw['slope'],raw['magnitude'],raw['bound'],raw['E0'],raw['retained']))
                assert np.isfinite([slope,magnitude,bound,base,retained]).all() and magnitude>=abs(slope)*(1-1e-12)
                assert bound==128*np.finfo(float).eps*magnitude
                assert abs(retained-base)<=64*np.finfo(float).eps*max(abs(base),1e-30)
                assert raw['frame']==probe['frame'] and raw['newton']==probe['newton']
                d1,d2,d3=(float(probe[f'slope{i}']) for i in (1,2,3))
                # Extrapolate two forward secants to zero step. The variation
                # across scales is an empirical indicator of truncation error.
                extrapolated=(10*d3-d2)/9
                variation=abs(extrapolated-(10*d2-d1)/9)
                energy_resolution=64*np.finfo(float).eps*max(abs(base),1e-30)/probe_step
                derivative_comparisons.append(dict(frame=frame,newton=int(raw['newton']),analytic=slope,
                    extrapolated_secant=extrapolated,scale_variation=variation,energy_resolution=energy_resolution))
                if slope>bound:
                    assert frame not in events
                    events[frame]=dict(reason='analytic_non_descent',newton=int(raw['newton']))
                continue
            if 'PSQ_PREVIEW_STOP frame=' not in line:continue
            raw=dict(re.findall(r'(\w+)=([^\s]+)',line));frame=int(raw['frame'])
            assert frame not in events
            events[frame]={k:v if k=='reason' else float(v) for k,v in raw.items()}
    accepted_events=[]
    for row in rows:
        n=row['native'];step=row['step'];preview=bool(n.get('preview_non_descent'))
        assert not n.get('preview_small_step')
        assert preview==(step in events)
        if not preview:continue
        assert m['level']<(9 if m['scene']=='flag' else 7)
        assert not n['converged'] and not n['hit_newton_limit'] and n['hit_line_search_limit']
        assert n['termination_reason']=='preview_non_descent' and n['last_line_search_alpha']==0
        e=events[step]
        if e['reason'] in ('analytic_non_descent','representable_exhaustion'):
            assert e['newton']==n['newton_iterations']-1
            accepted_events.append(step)
            continue
        assert e['reason']=='preview_non_descent' and e['alpha']==0 and e['newton']==n['newton_iterations']-1
        hs=np.array([e[f'h{i}'] for i in (1,2,3)]);es=np.array([e[f'e{i}'] for i in (1,2,3)])
        assert np.isfinite(hs).all() and np.isfinite(es).all() and 0<hs[2]<hs[1]<hs[0]<=.001
        noise=64*np.finfo(float).eps*max(abs(e['E_base']),abs(es).max(),1e-30)
        np.testing.assert_allclose(e['energy_noise'],noise,rtol=1e-14,atol=0)
        slopes=(es-e['E_base'])/hs
        assert np.all(es-e['E_base']>noise) and np.ptp(slopes)<=.25*slopes.max()
        np.testing.assert_allclose(slopes,[e[f'slope{i}'] for i in (1,2,3)],rtol=1e-12,atol=0)
        assert abs(e['E_retained']-e['E_base'])<=noise and e['E_last']>=e['E_base']-noise
        accepted_events.append(step)
    assert len(rows)==m['completed_steps'] and len(accepted_events)==m['preview_steps']
    for row in rows:
        if row['step'] in fallback_events:
            assert row['native']['converged'] and not row['native'].get('preview_non_descent')
    result=dict(accepted_steps=len(rows),verified_non_descent_stops=len(accepted_events),
        analytic_direction_stops=sum(events[k]['reason']=='analytic_non_descent' for k in accepted_events),
        representable_search_stops=sum(events[k]['reason']=='representable_exhaustion' for k in accepted_events),
        all_accepted_preview_evidence_verified=True,retained_current_energy_verified=True,
        preview_at_first_newton_frames=[k for k in accepted_events if events[k]['newton']==0],
        rejected_probe_groups=rejected,unaccepted_event_frames=sorted(set(events)-set(accepted_events)),
        limitation='Numerical evidence for these search directions, not proof of equilibrium or full-gradient stationarity.')
    out=a.run/'report';out.mkdir(exist_ok=True)
    if fallback_events:result['event_fallback_frames']=sorted(fallback_events)
    if derivative_comparisons:
        (out/'directional_derivatives.json').write_text(json.dumps(derivative_comparisons,indent=2))
    (out/'non_descent_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:len(v) if isinstance(v,list) else v for k,v in result.items()},indent=2))


if __name__=='__main__':main()
