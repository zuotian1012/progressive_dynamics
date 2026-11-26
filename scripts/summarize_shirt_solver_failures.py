"""Extract failure evidence without loading an entire native solver log into RAM."""
import argparse
import json
from pathlib import Path
import re

p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--caps',type=Path,required=True);a=p.parse_args()
meta=json.loads((a.run/'run.json').read_text());assert meta['status']!='running'
records=json.loads((a.run/'steps.json').read_text())
rows={r['step']:dict(frame=r['step'],native=r['native'],contact_caps=0,ccd_caps=0,cfl_caps=0,
      line_search_exhaustions=0,first_residual=None,last_residual=None,tolerance=None,last_line_search_failure=None)
      for r in records if r['native']['hit_newton_limit'] or r['native']['hit_line_search_limit']}
current=None
with a.run.with_suffix('.log').open(encoding='utf-8-sig',errors='replace') as stream:
    for line in stream:
        match=re.search(r'>>> Begin Frame: (\d+)',line)
        if match:current=rows.get(int(match[1]))
        if current is None:continue
        for key,token in [('contact_caps','Feasible Step Filter:'),('ccd_caps','CCD Filter:'),('cfl_caps','CFL Filter:')]:
            current[key]+=int(token in line)
        if 'Line Search Exits with Max Iteration' in line:
            current['line_search_exhaustions']+=1;current['last_line_search_failure']=line.strip()
        match=re.search(r'MaxTranslationChecker> Residual/AbsTol: ([^/]+)/([^\s]+)',line)
        if match:
            residual,tolerance=map(float,match.groups())
            if current['first_residual'] is None:current['first_residual']=residual
            current['last_residual']=residual;current['tolerance']=tolerance
with a.caps.open() as stream:
    for line in stream:
        record=json.loads(line)
        if record['frame'] in rows:
            rows[record['frame']]['last_sampled_strain_cap']={k:record[k] for k in ['iteration','legacy_min','directional_min']}
for row in rows.values():
    row['extra_backtracking_trials']=row['native']['line_search_trials']-row['native']['newton_iterations']
    row['last_residual_over_tolerance']=row['last_residual']/row['tolerance'] if row['tolerance'] else None
result=dict(frames=list(rows.values()),capped_frames=[k for k,r in rows.items() if r['native']['hit_newton_limit']],
            note='Only recorded failure flags and sampled cap directions; this does not diagnose gradient/Hessian consistency.')
(a.run/'report').mkdir(exist_ok=True);(a.run/'report/solver_failures.json').write_text(json.dumps(result,indent=2))
print(json.dumps([r for r in result['frames'] if r['native']['hit_newton_limit']],indent=2))
