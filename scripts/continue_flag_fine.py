"""Sequential full-window continuation; a failed run or audit blocks the next level."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--lower',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-package',type=Path);p.add_argument('--max-newton',type=int,default=300);p.add_argument('--preconditioner',choices=['diag','mas'],default='diag');p.add_argument('--last',type=int,default=9);a=p.parse_args()
    lower=a.lower.resolve();m=json.loads((lower/'run.json').read_text());assert m['completed'] and m['completed_steps']==500
    a.output.mkdir(exist_ok=False);chain=dict(status='running',seed=str(lower),levels=[],last=a.last)
    def save():(a.output/'chain.json').write_text(json.dumps(chain,indent=2))
    save()
    for k in range(m['level']+1,a.last+1):
        target=(a.output/f'L{k}').resolve();cmd=[sys.executable,str(ROOT/'scripts/run_flag_fine.py'),'--scene','flag','--level',str(k),'--lower',str(lower),'--assets',m['assets'],'--output',str(target),'--native-package',str(a.native_package.resolve()) if a.native_package else m['native_package'],'--steps','500','--rescue','--refined-probe','--adaptive-search','--max-newton',str(a.max_newton),'--preconditioner',a.preconditioner]
        chain.update(current_level=k,current_run=str(target));save()
        with target.with_suffix('.log').open('w') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        current=json.loads((target/'run.json').read_text()) if (target/'run.json').exists() else {}
        if r.returncode or not current.get('completed'):
            chain.update(status='failed_level',failure=current.get('error','See log'),accepted_steps=current.get('completed_steps',0));save();return 1
        with (target/'audit_commands.log').open('w') as log:
            checks=['audit_paper_run.py','audit_paper_non_descent.py'] if k<9 else ['audit_paper_run.py']
            checks.append('audit_flag_fine.py')
            for script in checks:
                r=subprocess.run([sys.executable,str(ROOT/'scripts'/script),str(target)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if r.returncode:chain.update(status='failed_audit',failure=script);save();return 1
        audit=json.loads((target/'audit.json').read_text())
        if not audit['completed_requested_run'] or audit['zero_motion_steps']:
            chain.update(status='failed_motion_audit');save();return 1
        chain['levels'].append(dict(level=k,run=str(target),wall_seconds=current['wall_seconds'],native_steps=current['native_steps'],preview_steps=current['preview_steps'],maximum_stretch=audit['maximum_stretch']))
        lower=target;save();print(f'VERIFIED L{k}: {current["wall_seconds"]:.1f}s',flush=True)
    chain['status']='completed';save();return 0

if __name__=='__main__':sys.exit(main())
