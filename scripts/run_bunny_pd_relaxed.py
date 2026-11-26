"""Run dependent L0/L1/L2 VelPro controls without overriding native stop policy."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path('output/bunny_pd_relaxed_20261116')
bundles=[Path('output/libuipc_psq_native/L0_to_90k.bin'),
         Path('output/bunny_L1_velpro_20261116/L1_to_90k.bin'),root/'L2_to_90k.bin']
sources=[Path('scripts/probe_libuipc_bunny.py'),Path(__file__),Path('scripts/libuipc_native_runtime.py')]
sources+=list(Path('external/libuipc-psq-src/build/python/src/uipc/_native').glob('uipc*.dll'))
manifest={'status':'running','requested_steps_per_level':400,'jobs':[],
          'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources+bundles}}
def save(): (root/'suite.json').write_text(json.dumps(manifest,indent=2))
save()
for level,bundle in enumerate(bundles):
    folder=root/f'L{level}'
    args=[sys.executable,'scripts/probe_libuipc_bunny.py','--native-source','--relaxed-pd',
          '--bundle',str(bundle),'--level',str(level),'--output',str(folder),
          '--steps','400','--step-budget','400','--wall-budget','1800']
    if level:args+=['--lower',str(root/f'L{level-1}')]
    tick=time.perf_counter()
    with folder.with_suffix('.log').open('w') as log:
        result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
    job=dict(level=level,command=args,exit_code=result.returncode,process_seconds=time.perf_counter()-tick)
    if (folder/'run.json').exists():
        m=json.loads((folder/'run.json').read_text())
        m.update(process_exit_code=result.returncode,process_wall_seconds=job['process_seconds'])
        if m['status']=='running':
            m.update(status='native_aborted',completed=False,failed_step=m['completed_steps']+1)
            log=folder.with_suffix('.log').read_text(errors='replace')
            errors=[s for s in log.splitlines() if '[warning]' in s or '[error]' in s]
            m['error']=errors[-1] if errors else 'Native exit without Python finalization'
        (folder/'run.json').write_text(json.dumps(m,indent=2))
        job.update(completed=m.get('completed',False),completed_steps=m['completed_steps'])
    manifest['jobs'].append(job); save()
    print(json.dumps(job),flush=True)
    if result.returncode or not job.get('completed'):
        manifest.update(status='incomplete',blocked_levels=list(range(level+1,3)));save();break
else:
    manifest.update(status='completed');save()
