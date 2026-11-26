"""Summarize same-build legacy/directional cap controls and actual direction samples."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
old=root/'legacy_settle';new=root/'directional_grasp'
om=json.loads((old/'run.json').read_text());nm=json.loads((new/'run.json').read_text())
assert om['completed'] and nm['completed']
for key in ['mesh_sha256','robot_q_sha256','material_parameters','config','psq_bundle_sha256','runtime_hashes']:
    assert om[key]==nm[key],key
n=om['completed_steps']
def metrics(path,count):
    rows=json.loads((path/'steps.json').read_text())[:count]
    times=np.load(path/'timing.npz')['step_seconds'][:count]
    return dict(steps=len(rows),physics_seconds=float(times.sum()),
                newton_iterations=sum(r['native']['newton_iterations'] for r in rows),
                newton_limit_steps=sum(r['native']['hit_newton_limit'] for r in rows),
                line_search_limit_steps=sum(r['native']['hit_line_search_limit'] for r in rows),
                native_converged_steps=sum(r['native']['converged'] for r in rows))
sample=[json.loads(line) for line in (root/'legacy_caps.jsonl').read_text().splitlines()]
tiny=[r for r in sample if r['legacy_min']<.001]
result=dict(same_build_and_physical_inputs=True,matched_steps=n,
            legacy=metrics(old,n),directional=metrics(new,n),directional_full=metrics(new,nm['completed_steps']),
            actual_legacy_audit=dict(samples=len(sample),tiny_samples=len(tiny),
                 tiny_binding_faces=dict(Counter(r['faces'][0]['face'] for r in tiny)),
                 tiny_samples_with_directional_cap_at_least_point_one=sum(r['directional_min']>=.1 for r in tiny)),
            caveats=['One run per mode; sampled audit adds overhead; not a statistical speed benchmark.',
                     'Native defaults include semi-implicit termination; native_converged is not strict Newton convergence.',
                     'This changes the custom PSQ feasibility cap, not upstream solver tolerances.'])
result['matched_prefix_wall_ratio']=result['legacy']['physics_seconds']/result['directional']['physics_seconds']
if (new/'report/audit.json').exists():result['motion_audit']=json.loads((new/'report/audit.json').read_text())
(root/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='motion_audit'},indent=2))
