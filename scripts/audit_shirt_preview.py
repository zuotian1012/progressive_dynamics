"""Independently verify every coarse exit from numerical evidence and native status."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shirt_demo.hierarchy import validate_level

def audit(root):
    meta=json.loads((root/'run.json').read_text());assert meta['status']!='running'
    records=json.loads((root/'steps.json').read_text());assert len(records)==meta['completed_steps']
    events={};rejects=0;forced=False
    with root.with_suffix('.log').open(encoding='utf-8-sig',errors='replace') as f:
        for line in f:
            forced=forced or 'PSQ_FORCE_ACCEPT frame=' in line
            rejects+=int('PSQ_PREVIEW_REJECT frame=' in line)
            if 'PSQ_PREVIEW_STOP frame=' not in line:continue
            row=dict(re.findall(r'(\w+)=([^\s]+)',line))
            frame=int(row['frame']);assert frame not in events
            row={k:(v if k=='reason' else float(v)) for k,v in row.items()};events[frame]=row
    assert not forced
    enabled=bool(meta.get('coarse_preview',False))
    if enabled:
        last=validate_level(meta['parameters']['assets'],meta['level'])
        assert meta['level']<last and meta['config']['newton']['semi_implicit']['enable']==0
        if meta['level']>0:assert meta.get('lower_run') and meta['method']==f"psq_velpro_L{meta['level']}"
        bundle=json.loads(Path(meta['psq_bundle']).with_suffix('.json').read_text())
        assert bundle['counts']['proxy_vertices']>bundle['counts']['coarse_vertices']
        assert meta['strain_step_cap']=='directional'
    else:assert not events
    for r in records:
        v=r['native'];is_preview=bool(v.get('preview_non_descent'))
        assert is_preview==(r['step'] in events)
        if not is_preview:continue
        e=events[r['step']]
        assert not v['converged'] and not v['hit_newton_limit']
        assert v['termination_reason']=='preview_non_descent' and v['last_line_search_alpha']==0
        assert v['hit_line_search_limit'] and e['newton']==v['newton_iterations']-1
        assert e['reason']=='preview_non_descent' and e['alpha']==0
        hs=np.array([e[f'h{i}'] for i in (1,2,3)]);es=np.array([e[f'e{i}'] for i in (1,2,3)])
        assert np.isfinite(hs).all() and np.isfinite(es).all() and 0<hs[2]<hs[1]<hs[0]<=.001
        noise=64*np.finfo(float).eps*max(abs(e['E_base']),abs(es).max(),1e-30)
        np.testing.assert_allclose(e['energy_noise'],noise,rtol=1e-14,atol=0)
        slopes=(es-e['E_base'])/hs
        assert np.all(es-e['E_base']>noise) and np.ptp(slopes)<=.25*slopes.max()
        np.testing.assert_allclose(slopes,[e[f'slope{i}'] for i in (1,2,3)],rtol=1e-12,atol=0)
        assert abs(e['E_retained']-e['E_base'])<=noise
        assert e['E_last']>=e['E_base']-noise
    if enabled:assert len(events)==meta['preview_stopped_steps']
    unclassified=[r['step'] for r in records if not r['native']['converged'] and not r['native'].get('preview_non_descent')]
    return dict(completed=meta['completed'],steps=len(records),preview_exit_frames=sorted(events),verified_preview_exits=len(events),
        rejected_probe_groups=rejects,unclassified_nonconverged_frames=unclassified,
        non_preview_line_search_limit_frames=[r['step'] for r in records if r['native']['hit_line_search_limit'] and not r['native'].get('preview_non_descent')],
        preview_at_first_newton_frames=[k for k,e in events.items() if e['newton']==0],
        native_converged_steps=sum(r['native']['converged'] for r in records),all_preview_evidence_verified=True,
        preview_not_counted_as_convergence=True,retained_current_energy_verified=True,
        note='Numerical three-scale non-descent evidence, not a proof of full-gradient stationarity. Non-preview native failure policy is unchanged.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();result=audit(a.run)
    (a.run/'report').mkdir(exist_ok=True);(a.run/'report/preview_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
