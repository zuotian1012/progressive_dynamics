"""Extract an explicitly labeled, fully accepted temporal window; never invent states.

The source can be longer or interrupted. Its original requested duration and
completion status are copied verbatim into the derivation provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--steps',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=(a.source/'run.json').read_bytes();m=json.loads(raw);records=json.loads((a.source/'steps.json').read_text())
    if not 0<a.steps<=m['completed_steps'] or len(records)<a.steps:raise ValueError('Requested prefix is not fully accepted')
    rows=records[:a.steps]
    for row in rows:
        n=row['native']
        if not n['completed'] or n['hit_newton_limit']:raise ValueError('Invalid accepted prefix')
        if not(n['converged'] or n.get('preview_small_step') or n.get('preview_non_descent')):raise ValueError('Missing acceptance status')
    x=np.load(a.source/'positions.npy',mmap_mode='r')[:a.steps+1]
    if not np.isfinite(x).all():raise ValueError('Nonfinite prefix')
    a.output.mkdir(parents=True,exist_ok=False);np.save(a.output/'positions.npy',x)
    m['derivation']=dict(kind='accepted_temporal_prefix',source_run=str(a.source.resolve()),
        source_requested_steps=m['steps'],source_completed_steps=m['completed_steps'],source_status=m['status'],
        source_run_json_sha256=hashlib.sha256(raw).hexdigest(),
        source_positions_sha256=hashlib.sha256((a.source/'positions.npy').read_bytes()).hexdigest(),
        note='Only completion of this shorter window is claimed; the original longer run is not relabeled as completed')
    m.update(steps=a.steps,completed_steps=a.steps,completed=True,status='completed',
        scope=f'Accepted {a.steps*m["dt"]:g}s temporal window from explicitly linked source; no recomputed or interpolated positions',
        native_steps=sum(bool(r['native']['converged']) for r in rows),
        preview_steps=sum(bool(r['native'].get('preview_small_step') or r['native'].get('preview_non_descent')) for r in rows),
        small_step_preview_steps=sum(bool(r['native'].get('preview_small_step')) for r in rows),wall_seconds=sum(r['seconds'] for r in rows))
    m.pop('error',None);m.pop('rejected_step',None)
    m['reconstruction_choices']=[r for r in m['reconstruction_choices'] if not r.startswith('bunny:')]
    m['reconstruction_choices'].append(f'bunny: local comparison window {a.steps*m["dt"]:g}s; paper total duration not recovered')
    (a.output/'run.json').write_text(json.dumps(m,indent=2));(a.output/'steps.json').write_text(json.dumps(rows,indent=2))
    (a.output/'source_run_snapshot.json').write_bytes(raw)
    if (a.source/'source_snapshot.py').exists():(a.output/'source_snapshot.py').write_bytes((a.source/'source_snapshot.py').read_bytes())
    print(json.dumps(m['derivation'],indent=2))

if __name__=='__main__':main()
