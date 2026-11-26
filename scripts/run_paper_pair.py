"""Run both reconstructed paper-scale scenes sequentially on one GPU.

One scene's failed level does not suppress the other scene's experiment.
Both retain explicit incomplete statuses, source snapshots and accepted states.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--flag-assets',type=Path,default=ROOT/'output/pdpp_flag_paper_assets_20261117_v2')
    p.add_argument('--bunny-assets',type=Path,default=ROOT/'output/bunny_paper_assets')
    p.add_argument('--short',action='store_true',help='2 s flag and 1 s bunny, first two levels only')
    p.add_argument('--preview-min-alpha',type=float,default=0.)
    a=p.parse_args()
    if a.preview_min_alpha!=0:
        p.error('Shirt-aligned stopping requires --preview-min-alpha 0')
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(scope='Short prerequisite' if a.short else 'Full requested hierarchy',scenes={},complete=False)
    target=a.output/'pair.json'
    if not a.short:
        specs=[('flag',a.flag_assets,500,10),('bunny',a.bunny_assets,400,8)]
        for scene,assets,steps,count in specs:
            (a.output/scene).mkdir()
            result['scenes'][scene]=dict(status='running',completed_levels=[],requested_steps=steps,levels=count)
        for level in range(10):
            for scene,assets,steps,count in specs:
                record=result['scenes'][scene]
                if level>=count or record['status']!='running':continue
                base=a.output/scene
                cmd=[sys.executable,str(ROOT/'scripts/run_paper_l0.py'),'--scene',scene,'--assets',str(assets),
                    '--output',str(base/f'L{level}'),'--steps',str(steps),'--level',str(level),
                    '--preview-min-alpha',str(a.preview_min_alpha)]
                if level:cmd+=['--lower',str(base/f'L{level-1}')]
                with (base/f'L{level}.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                path=base/f'L{level}/run.json';m=json.loads(path.read_text()) if path.exists() else {}
                if r.returncode or not m.get('completed'):
                    record.update(status='blocked_by_failed_level',failed_level=level,completed_steps_in_failed_level=m.get('completed_steps',0),failure=m.get('error','See level log'))
                else:
                    record['completed_levels'].append(level)
                    if level==count-1:record['status']='completed_requested_levels'
                (base/'hierarchy.json').write_text(json.dumps(record,indent=2))
                target.write_text(json.dumps(result,indent=2));print(f'{scene} L{level}: {record["status"]}',flush=True)
        result['complete']=all(s['status']=='completed_requested_levels' for s in result['scenes'].values())
        target.write_text(json.dumps(result,indent=2));return 0 if result['complete'] else 1
    for scene,assets,steps in [('flag',a.flag_assets,50 if a.short else 500),('bunny',a.bunny_assets,100 if a.short else 400)]:
        cmd=[sys.executable,str(ROOT/'scripts/run_paper_hierarchy.py'),'--scene',scene,'--assets',str(assets),
            '--output',str(a.output/scene),'--steps',str(steps),'--preview-min-alpha',str(a.preview_min_alpha)]
        if a.short:cmd+=['--last-level','1']
        with (a.output/f'{scene}.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        path=a.output/scene/'hierarchy.json'
        result['scenes'][scene]=json.loads(path.read_text()) if path.exists() else dict(exit_code=r.returncode,status='failed_before_manifest')
        target.write_text(json.dumps(result,indent=2));print(f'{scene}: {result["scenes"][scene]["status"]}',flush=True)
    result['complete']=all(s['status']=='completed_requested_levels' for s in result['scenes'].values())
    target.write_text(json.dumps(result,indent=2))
    return 0 if result['complete'] else 1
if __name__=='__main__':sys.exit(main())
