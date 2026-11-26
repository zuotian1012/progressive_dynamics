"""Same strict workers, methods concurrent within each level, serial dependencies across levels."""
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from run_flag_hierarchy_suite import ROOT,OUT,BASE,L1,write


def main():
    OUT.mkdir(exist_ok=True);lock=threading.Lock();failures=[];start=time.time()
    index={'L0':{'root':str(BASE),'run':str(BASE/'L0')},'concurrent_methods':3,'levels':{}}
    index['levels']['1']={m:dict(root=str(L1),run=str(L1/('L1_direct_local' if m=='direct' else f'L1_{m}')),status='completed') for m in ['semi','velpro','direct']}
    def update(entry,**kwargs):
        with lock:entry.update(kwargs);write(OUT/'index.json',index)
    for level in range(2,10):
        root=OUT/f'L{level}'
        if not root.exists():
            with (OUT/f'prepare_L{level}.log').open('w') as fp:
                subprocess.run([sys.executable,'scripts/run_flag_parallel_wind.py','--output',str(root),'--reuse-assets',str(L1),
                    '--seed','.001','--wind-scale','3','--pin-mode','corners','--level',str(level),'--method','semi',
                    '--lower',index['levels'][str(level-1)]['semi']['run'],'--export-only'],cwd=ROOT,stdout=fp,stderr=subprocess.STDOUT,check=True)
        entries={m:dict(root=str(root),run=str(root/(f'L{level}_{m}'+('_local' if m=='direct' else ''))),status='pending') for m in ['semi','velpro','direct']}
        index['levels'][str(level)]=entries;write(OUT/'index.json',index)
        def worker(method):
            entry=entries[method];run=Path(entry['run']);previous=index['levels'][str(level-1)][method]
            if method!='direct' and previous['status']!='completed':update(entry,status='parent_failed');return
            if not run.exists():
                update(entry,status='running');print('START',run.name,flush=True)
                args=[sys.executable,'scripts/run_flag_parallel_wind.py','--output',str(root),'--seed','.001','--wind-scale','3',
                    '--pin-mode','corners','--level',str(level),'--method',method]
                if method!='direct':args+=['--lower',previous['run']]
                with (root/f'{run.name}.log').open('w') as fp:rc=subprocess.call(args,cwd=ROOT,stdout=fp,stderr=subprocess.STDOUT)
                update(entry,returncode=rc)
                if rc:
                    update(entry,status='failed',details=f'Worker exited with code {rc}');return
            meta=json.loads((run/'run.json').read_text()) if (run/'run.json').exists() else {}
            while meta.get('status') in ('running','initializing'):
                time.sleep(1)
                try:meta=json.loads((run/'run.json').read_text())
                except (OSError,json.JSONDecodeError):continue
            if meta.get('status')!='completed':
                update(entry,status='failed',details=meta.get('error','Worker did not complete'));print('FAILED',run.name,flush=True);return
            audit=root/'report'/run.name/'audit.json'
            if not audit.exists():
                with (root/f'{run.name}_audit.log').open('w') as fp:
                    rc=subprocess.call([sys.executable,'scripts/report_flag_parallel_wind.py','--root',str(root),'--run-name',run.name,'--audit-only'],cwd=ROOT,stdout=fp,stderr=subprocess.STDOUT)
                if rc:update(entry,status='audit_failed');return
            update(entry,status='completed',accepted_frames=meta['accepted_frames'],elapsed_s=meta['elapsed_s'],audit=str(audit))
            print('COMPLETE',run.name,round(meta['elapsed_s'],1),flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(worker,['semi','velpro','direct']))
        failures.extend(e['run'] for e in entries.values() if e['status']!='completed')
    write(OUT/'summary.json',dict(failed=failures,wall_s=time.time()-start,complete=not failures,concurrent_methods=3))
    print('SUITE FINISHED',failures,flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
