"""Run L1 Semi and VelPro from the audited one-second no-ground L0 pair."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/bunny_pcg_no_ground_20261124'
OUT=BASE/'L1'

def main():
    OUT.mkdir(exist_ok=False)
    suite={'status':'running','runs':{}}
    for label in ['mu0','mu03']:
        for method in ['semi','velpro']:
            name=f'{label}_{method}'
            run=OUT/name
            command=[sys.executable,str(ROOT/'scripts/run_bunny_ground_fine.py'),
                     '--scene','bunny','--assets',str(ROOT/'output/bunny_paper_assets'),
                     '--output',str(run),'--level','1','--lower',str(BASE/label),
                     '--steps','100','--method',method,'--frozen-offset',
                     '--refined-probe','--adaptive-search','--max-newton','1500',
                     '--preconditioner','diag','--newton-velocity-tol','.001']
            print('Starting',name,flush=True)
            with run.with_suffix('.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            suite['runs'][name]={'returncode':result.returncode,'command':command}
            (OUT/'suite.json').write_text(json.dumps(suite,indent=2))
            print('Finished',name,result.returncode,flush=True)
            if result.returncode:continue
            for script,extra in [('audit_paper_run.py',[]),('audit_paper_non_descent.py',[]),
                                 ('audit_flag_fine_search.py',['--check-converged-energy'])]:
                with (run/(script+'.log')).open('w',encoding='utf-8') as log:
                    result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),str(run),*extra],
                                          cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                suite['runs'][name][script]=result.returncode
                (OUT/'suite.json').write_text(json.dumps(suite,indent=2))
                print('Audit',name,script,result.returncode,flush=True)
    suite['status']='finished'
    (OUT/'suite.json').write_text(json.dumps(suite,indent=2))

if __name__=='__main__':main()
