"""Sequential paper-scale levels, stopping at the first unaccepted trajectory.

Full flag duration is published (20 s); bunny 4 s is a local observation window.
No dependent level is run from a failed lower trajectory.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True,choices=['flag','bunny'])
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int)
    p.add_argument('--last-level',type=int)
    p.add_argument('--tiny-probe',action='store_true')
    p.add_argument('--preview-min-alpha',type=float,default=0.)
    a=p.parse_args()
    if a.tiny_probe or a.preview_min_alpha!=0:
        p.error('Shirt-aligned stopping requires --preview-min-alpha 0 and no --tiny-probe')
    a.output.mkdir(parents=True,exist_ok=False)
    count=10 if a.scene=='flag' else 8
    last=count-1 if a.last_level is None else a.last_level
    if not 0<=last<count:p.error('Invalid last level')
    plan=dict(scene=a.scene,assets=str(a.assets.resolve()),levels=count,last_requested_level=last,
        requested_steps=a.steps or (500 if a.scene=='flag' else 400),status='running',completed_levels=[],
        full_duration_paper_verified=a.scene=='flag',semi_implicit=False,forced_updates=False)
    path=a.output/'hierarchy.json'
    def save():path.write_text(json.dumps(plan,indent=2))
    save()
    for k in range(last+1):
        cmd=[sys.executable,str(ROOT/'scripts/run_paper_l0.py'),'--scene',a.scene,'--assets',str(a.assets.resolve()),
            '--output',str((a.output/f'L{k}').resolve()),'--level',str(k),'--steps',str(plan['requested_steps'])]
        if k:cmd+=['--lower',str((a.output/f'L{k-1}').resolve())]
        if a.tiny_probe:cmd+=['--tiny-probe']
        if a.preview_min_alpha:cmd+=['--preview-min-alpha',str(a.preview_min_alpha)]
        with (a.output/f'L{k}.log').open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        mpath=a.output/f'L{k}/run.json';m=json.loads(mpath.read_text()) if mpath.exists() else {}
        if result.returncode or not m.get('completed'):
            plan.update(status='blocked_by_failed_level',failed_level=k,exit_code=result.returncode,
                completed_steps_in_failed_level=m.get('completed_steps',0),failure=m.get('error','See level log'))
            save();print(json.dumps(plan,indent=2));return 1
        plan['completed_levels'].append(k);save();print(f'COMPLETED L{k}',flush=True)
    plan['status']='completed_requested_levels';save();print(json.dumps(plan,indent=2));return 0

if __name__=='__main__':sys.exit(main())
