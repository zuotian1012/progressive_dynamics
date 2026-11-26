"""Serial paper-scale bunny experiments; a failed method never cancels controls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(config):
    files=[Path(config),*sorted(Path(__file__).parent.glob('*.py'))]
    parameters=json.loads(Path(config).read_text())['parameters']
    files.extend(sorted(Path(parameters['assets']).glob('*.npz')))
    files.extend(sorted(Path(parameters['assets']).glob('*.npy')))
    files.append(Path(parameters['assets'])/'manifest.json')
    return {str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def execute(command,log):
    with Path(log).open('w',encoding='utf-8') as stream:
        return subprocess.call(command,stdout=stream,stderr=subprocess.STDOUT)


def run(config,output):
    config=Path(config).resolve();output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Refusing to overwrite nonempty suite output')
    p=json.loads(config.read_text())['parameters']
    if p['scene']!='bunny':raise ValueError('Bunny configuration required')
    if len(set(p['methods']))!=len(p['methods']):raise ValueError('Duplicate methods')
    inputs=fingerprint(config)
    output.mkdir(parents=True,exist_ok=True)
    state={'schema':1,'started_utc':now(),'config':str(config),'inputs_sha256':inputs,
           'completed':False,'finished':False,'status':'running','jobs':[],
           'scope':'Reconstructed paper-scale inputs; one serial run per method. No automatic parameter changes or retries.',
           'timing_scope':'Each job wall time includes scene setup and simulation I/O; physics timing remains in per-level run.json. Audit/report wall times are separate.'}
    for method in p['methods']:
        command=[sys.executable,'-u','-m','progressive.run','--config',str(config),'--methods',method,'--output',str(output/method)]
        state['jobs'].append({'method':method,'status':'pending','command':command,'output':str(output/method)})
    def save():
        temporary=output/'suite.json.tmp'
        temporary.write_text(json.dumps(state,indent=2),encoding='utf-8')
        temporary.replace(output/'suite.json')
    save()
    try:
        for job in state['jobs']:
            if fingerprint(config)!=inputs:
                raise ValueError('Inputs or implementation changed during suite; refusing mixed comparisons')
            if (output/'STOP').exists():raise InterruptedError('Suite STOP marker: no further method will start')
            method=job['method'];root=Path(job['output'])
            job.update(status='running',started_utc=now());save()
            print(f'Starting {method}: {root}',flush=True)
            start=time.perf_counter()
            job['exit_code']=execute(job['command'],output/f'{method}.log')
            job['simulation_wall_seconds']=time.perf_counter()-start
            try:
                manifest=json.loads((root/'experiment.json').read_text()) if (root/'experiment.json').exists() else {}
            except (OSError,json.JSONDecodeError) as error:
                manifest={};job['manifest_error']=repr(error)
            job['simulation_completed']=job['exit_code']==0 and manifest.get('completed',False)
            job['status']='auditing';save()
            accepted=[] if job['simulation_completed'] else ['--accepted-prefix']
            audit_start=time.perf_counter()
            job['geometry_audit_exit_code']=execute([sys.executable,'-m','progressive.audit_bunny',str(root),*accepted],output/f'{method}_audit.log')
            if job['simulation_completed']:
                job['solver_audit_exit_code']=execute([sys.executable,'scripts/audit_progressive_runs.py',str(root),'--output',str(root/'audit_solver.json')],output/f'{method}_solver_audit.log')
            job['audit_wall_seconds']=time.perf_counter()-audit_start
            job['status']='completed' if job['simulation_completed'] and job['geometry_audit_exit_code']==0 and job.get('solver_audit_exit_code')==0 else 'failed'
            job['finished_utc']=now();save()
            print(f'{method}: {job["status"]}; simulation wall {job["simulation_wall_seconds"]:.1f} s',flush=True)
        # Never substitute coarse or failed trajectories for the full comparison.
        state['completed']=all(j['status']=='completed' for j in state['jobs'])
        if state['completed']:
            state['status']='rendering';save()
            roots=[j['output'] for j in state['jobs']]
            start=time.perf_counter()
            state['report_exit_code']=execute([sys.executable,'-m','progressive.report',roots[0],'--compare-with',*roots[1:]],output/'report.log')
            state['report_wall_seconds']=time.perf_counter()-start
            state['completed']=state['report_exit_code']==0
        state.update(finished=True,status='completed' if state['completed'] else 'finished_with_failures',finished_utc=now())
        save()
    except BaseException as error:
        state.update(status='interrupted',error=repr(error),finished_utc=now())
        save()
        raise
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    state=run(args.config,args.output)
    if not state['completed']:raise SystemExit(1)


if __name__=='__main__':main()
