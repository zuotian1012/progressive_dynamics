"""Supervise one bounded native probe and preserve failure/accepted-prefix metadata."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import os

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--bundle',type=Path)
    source.add_argument('--native-only',action='store_true',help='Native coarse elasticity without PSQ or the custom strain barrier')
    p.add_argument('--steps',type=int,default=25)
    p.add_argument('--step-budget',type=int,default=40,help='Explicit frame budget, at most 2000; wall budget remains 180 seconds')
    p.add_argument('--psq-mode',choices=['proxy','reference'],default='proxy')
    p.add_argument('--initial-stretch',type=float,default=1.)
    p.add_argument('--direction-audit',action='store_true')
    p.add_argument('--force-updates',action='store_true')
    p.add_argument('--coarse-preview',action='store_true')
    p.add_argument('--level',type=int,choices=[0,1,2],default=0)
    p.add_argument('--lower',type=Path)
    a=p.parse_args();log=a.output.with_suffix('.log')
    if a.native_only and (a.coarse_preview or a.force_updates or a.direction_audit or a.lower or a.initial_stretch!=1):
        raise ValueError('Native-only comparison requires ordinary strict integration without PSQ-specific options')
    if a.coarse_preview and (a.force_updates or a.psq_mode!='proxy' or a.direction_audit):
        raise ValueError('Coarse preview requires proxy mode and is separate from force/failure audit')
    if a.force_updates and (a.direction_audit or a.psq_mode!='proxy'):
        raise ValueError('Forced updates require proxy mode and are separate from failure diagnostics')
    if a.output.exists() or log.exists():raise FileExistsError('Preserve previous results; choose a new output')
    log.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(Path(__file__).with_name('probe_libuipc_bunny.py')),
        '--native-source','--output',str(a.output),'--steps',str(a.steps),'--step-budget',str(a.step_budget),
        '--psq-mode',a.psq_mode,'--initial-stretch',str(a.initial_stretch)]
    if a.bundle:cmd+=['--bundle',str(a.bundle)]
    if a.force_updates:cmd.append('--force-updates')
    if a.coarse_preview:cmd.append('--coarse-preview')
    cmd+=['--level',str(a.level)]
    if a.lower:cmd+=['--lower',str(a.lower)]
    tick=time.perf_counter()
    env=os.environ.copy()
    env.pop('UIPC_PSQ_BUNDLE',None)
    env.pop('UIPC_PSQ_MODE',None)
    env.pop('UIPC_PSQ_DIRECTION_AUDIT',None)
    env.pop('UIPC_PSQ_FORCE_UPDATE',None)
    env.pop('UIPC_PSQ_PREVIEW_STOP',None)
    if a.direction_audit:env['UIPC_PSQ_DIRECTION_AUDIT']='1'
    with log.open('w',encoding='utf-8') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
    elapsed=time.perf_counter()-tick
    run_path=a.output/'run.json'
    meta=json.loads(run_path.read_text()) if run_path.exists() else dict(completed_steps=0,requested_steps=a.steps)
    meta.update(process_exit_code=result.returncode,process_wall_seconds=elapsed)
    if result.returncode or not meta.get('completed'):
        text=log.read_text(encoding='utf-8',errors='replace')
        errors=[line for line in text.splitlines() if re.search('Line Search Exits|Max Iteration|failure:|Error:',line)]
        if meta.get('status') != 'budget_stopped':
            meta.update(status='failed',completed=False,error=errors[-1] if errors else f'Native process exited {result.returncode}',
                        failed_step=meta['completed_steps']+1)
        records=json.loads((a.output/'steps.json').read_text()) if (a.output/'steps.json').exists() else []
        meta.setdefault('wall_seconds',None)
        meta.setdefault('physics_seconds',sum(r['seconds'] for r in records))
    a.output.mkdir(parents=True,exist_ok=True)
    run_path.write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:meta.get(k) for k in ['status','completed_steps','process_exit_code','process_wall_seconds','error']},indent=2))
    return 0 if meta.get('completed') else 1

if __name__=='__main__':sys.exit(main())
