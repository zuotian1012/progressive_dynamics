"""Resumeable level-by-level flag suite, strict acceptance, never overwrite a run."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/flag_to_L9_20261122'
BASE=ROOT/'output/flag_L0_two_pins_wind300_20261122'
L1=ROOT/'output/flag_L1_two_pins_wind300_20261122'


def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2),encoding='utf-8');temp.replace(path)


def main():
    OUT.mkdir(exist_ok=True)
    index={'L0':{'root':str(BASE),'run':str(BASE/'L0')},'levels':{}}
    index['levels']['1']={m:dict(root=str(L1),run=str(L1/('L1_direct_local' if m=='direct' else f'L1_{m}')),status='completed') for m in ['semi','velpro','direct']}
    failures=[];start=time.time()
    for level in range(2,10):
        root=OUT/f'L{level}';index['levels'][str(level)]={}
        for method in ['semi','velpro','direct']:
            name=f'L{level}_{method}'+('_local' if method=='direct' else '')
            run=root/name;entry=dict(root=str(root),run=str(run),status='pending');index['levels'][str(level)][method]=entry
            previous=index['levels'][str(level-1)][method]
            if method!='direct' and previous['status']!='completed':
                entry['status']='parent_failed';failures.append(name);continue
            args=[sys.executable,'scripts/run_flag_parallel_wind.py','--output',str(root),'--seed','.001','--wind-scale','3','--pin-mode','corners','--level',str(level),'--method',method]
            if method!='direct':args+=['--lower',previous['run']]
            if not root.exists():args+=['--reuse-assets',str(L1)]
            if not run.exists():
                # The worker creates root on its first invocation; keep log outside it until then.
                log=OUT/f'{name}.log'
                entry['status']='running';write(OUT/'index.json',index)
                print('START',name,flush=True)
                with log.open('w') as fp:rc=subprocess.call(args,cwd=ROOT,stdout=fp,stderr=subprocess.STDOUT)
                if root.exists():log.replace(root/f'{name}.log')
                entry['returncode']=rc
            meta=json.loads((run/'run.json').read_text()) if (run/'run.json').exists() else {}
            if meta.get('status')!='completed':
                entry.update(status='failed',details=meta.get('error','Worker did not complete'));failures.append(name)
                write(OUT/'index.json',index);print('FAILED',name,entry['details'],flush=True);continue
            audit=root/'report'/name/'audit.json'
            if not audit.exists():
                with (root/f'{name}_audit.log').open('w') as fp:
                    rc=subprocess.call([sys.executable,'scripts/report_flag_parallel_wind.py','--root',str(root),'--run-name',name,'--audit-only'],cwd=ROOT,stdout=fp,stderr=subprocess.STDOUT)
                if rc:
                    entry.update(status='audit_failed');failures.append(name);write(OUT/'index.json',index);continue
            entry.update(status='completed',accepted_frames=meta['accepted_frames'],elapsed_s=meta['elapsed_s'],audit=str(audit))
            write(OUT/'index.json',index);print('COMPLETE',name,round(meta['elapsed_s'],1),'seconds',flush=True)
        write(OUT/'index.json',index)
    write(OUT/'summary.json',dict(failed=failures,wall_s=time.time()-start,complete=not failures))
    print('SUITE FINISHED',failures,flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
