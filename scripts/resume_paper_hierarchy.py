"""Continue from a completed, provenance-linked level; preserve all previous runs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from paper_solver_settings import validate_shirt_solver_history
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--from-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    previous=a.from_run.resolve();m=json.loads((previous/'run.json').read_text())
    if not m['completed']:raise ValueError('Only completed trajectories can seed finer levels')
    validate_shirt_solver_history(m)
    chain=[];cursor=previous
    while True:
        cm=json.loads((cursor/'run.json').read_text());validate_shirt_solver_history(cm)
        chain.insert(0,dict(level=cm['level'],run=str(cursor)))
        if not cm.get('lower_run'):break
        cursor=Path(cm['lower_run'])
    a.output.mkdir(parents=True,exist_ok=True);last=9 if m['scene']=='flag' else 7
    summary=dict(status='running',chain=chain,scene=m['scene'],requested_steps=m['steps'],last_level=last)
    def save():(a.output/'chain.json').write_text(json.dumps(summary,indent=2))
    save()
    for k in range(m['level']+1,last+1):
        target=a.output/f'L{k}'
        if target.exists():raise FileExistsError(target)
        cmd=[sys.executable,str(ROOT/'scripts/run_paper_l0.py'),'--scene',m['scene'],'--assets',m['assets'],'--output',str(target.resolve()),
            '--level',str(k),'--lower',str(previous),'--steps',str(m['steps']),
            '--max-newton',str(m['config']['newton']['max_iter']),'--preview-min-alpha',str(m['preview_min_alpha']),
            '--preconditioner',m['config']['linear_system']['fem_preconditioner']]
        if m.get('native_package'):
            cmd.extend(['--native-package',m['native_package']])
        with (a.output/f'L{k}.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        path=target/'run.json';current=json.loads(path.read_text()) if path.exists() else {}
        if r.returncode or not current.get('completed'):
            summary.update(status='blocked_by_failed_level',failed_level=k,failure=current.get('error','See log'),accepted_steps=current.get('completed_steps',0));save();print(json.dumps(summary,indent=2));return 1
        summary['chain'].append(dict(level=k,run=str(target.resolve())));save();previous=target.resolve();print(f'COMPLETED L{k}',flush=True)
    summary['status']='completed_hierarchy';save();return 0

if __name__=='__main__':sys.exit(main())
