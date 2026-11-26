"""Retry the converged 75-step L7 with the original CUDA Graph setting, preserving the failed run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from run_bunny_velpro_chain import ROOT, ASSETS, EVENT_RUNTIME, validate, digest, runtime_provenance, audit, event_audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'output/bunny_newton_converged_mu03_velpro_75_20261125/L7_retry_graph_20261126')
    args=parser.parse_args()
    base=ROOT/'output/bunny_newton_converged_mu03_velpro_75_20261125'
    parent=base/'L6';run=args.output.resolve()
    settings=SimpleNamespace(steps=75,friction=.3,event_fallback=True,output=base)
    dll=runtime_provenance(EVENT_RUNTIME)['dll_sha256']
    parent_meta=validate(parent,6,settings,dll)
    assert parent_meta["config"]["linear_system"]["use_cuda_graph"]==1
    saved=json.loads((parent/'no_ground_audit.json').read_text())
    assert saved['passed'] and saved['saved_states_checked']==76 and not saved['intersecting_states']
    assert saved['positions_sha256']==digest(parent/'positions.npy')
    if run.exists() or run.with_suffix('.log').exists():
        raise RuntimeError(f'Output already exists; choose a fresh --output: {run}')
    command=[sys.executable,'-u',str(ROOT/'scripts/run_bunny_ground_fine.py'),
        '--scene','bunny','--assets',str(ASSETS),'--output',str(run),'--level','7','--lower',str(parent),
        '--steps','75','--dt','.01','--method','velpro','--velocity-map','intrinsic',
        '--frozen-offset','--event-fallback','--refined-probe','--adaptive-search',
        '--max-newton','1500','--newton-velocity-tol','.001','--preconditioner','diag',
        '--native-package',str(EVENT_RUNTIME),'--solver-trace']
    print('L7: 75 steps from time zero, mu=.3, no ground, VelPro, GPU PCG, CUDA Graph ON (mode 1); every step must converge.',flush=True)
    print('Parent:',parent,'\nOutput:',run,flush=True)
    if args.dry_run:
        print('Preflight passed. No simulation started.');return
    run.parent.mkdir(parents=True,exist_ok=True)
    run.with_name(run.name+'_command.json').write_text(json.dumps(command,indent=2))
    env=os.environ.copy();env['BUNNY_EVENT_FALLBACK']='1'
    with run.with_suffix('.log').open('xb') as log:
        with subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT) as process:
            for line in process.stdout:
                log.write(line)
                if line.startswith(b'ACCEPTED '):
                    log.flush();print(line.decode('utf-8',errors='replace').strip(),flush=True)
            code=process.wait()
    if code:raise RuntimeError(f'L7 failed, exit code {code}; see {run.with_suffix(".log")}')
    m=validate(run,7,settings,dll)
    assert m['config']['linear_system']['use_cuda_graph']==1
    assert m['config']==parent_meta['config']
    print('Simulation completed; auditing all 76 saved states...',flush=True)
    audit(run,75)
    event_audit(run,m,json.loads((run/'steps.json').read_text()))
    print('L7 completed and all audits passed.',flush=True)


if __name__=='__main__':main()
