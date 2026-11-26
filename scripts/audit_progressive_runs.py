"""Audit saved NEW solver runs; exit nonzero for incomplete or infeasible data."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def audit(root):
    config=json.loads((root/'experiment.json').read_text())
    if not config['completed']: raise ValueError(f'{root}: experiment incomplete')
    summary={'root':str(root.resolve()),'runs':[]}
    for meta in config['runs']:
        if 'reused_from' in meta: continue
        p=root/meta['folder']
        report=json.loads((p/'solver.json').read_text())
        metrics=json.loads((p/'metrics.json').read_text())
        state=np.load(p/'positions.npy',mmap_mode='r')
        times=np.load(p/'timing.npz')['step_seconds']
        assert meta['completed'] and meta['completed_steps']==config['steps']
        assert len(report)==len(metrics)==len(times)==config['steps']
        assert np.isfinite(state).all() and np.isfinite(times).all() and np.all(times>=0)
        assert all(r.get('accepted',r['converged']) for r in report)
        preview_stops=[r for r in report if not r['converged']]
        if preview_stops:
            assert meta['method']!='direct' and meta['level']<config['parameters']['levels']-1
            assert not config['parameters'].get('strict_preview',True)
            assert all(r['preview_stop'] and r['reason'] in ('non_descent_direction','line_search_failed','preview_step_too_small') for r in preview_stops)
        stretch=max(m['max_stretch'] for m in metrics)
        assert stretch < config['parameters']['strain_limit']
        item={'method':meta['method'],'level':meta['level'],'accepted_steps':len(report),
              'newton_converged_steps':len(report)-len(preview_stops),'preview_stopped_steps':len(preview_stops),
              'physics_seconds':float(times.sum()),'max_stretch':stretch,
              'max_newton_iterations':max(r['iterations'] for r in report)}
        if meta['level']:
            item['mean_coarse_consistency_rms_mm']=float(1000*np.mean([m['coarse_rms_m'] for m in metrics]))
        summary['runs'].append(item)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();result=[audit(root) for root in args.roots]
    if args.output:args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
