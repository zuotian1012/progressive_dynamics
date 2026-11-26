"""Run and audit each remaining bunny level before allowing the next level."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
from bunny_runtime import continuation_package
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lower',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,help='Requested time window in steps, at most the length of the audited parent')
    p.add_argument('--last',type=int,default=7)
    p.add_argument('--method',choices=['semi','full','velpro'],default='velpro',
                   help='semi: original PD (2024); full and velpro: PD++ (2025)')
    p.add_argument('--native-package',type=Path,help='CUDA package; defaults to the corrected build at L0 and the parent build thereafter')
    p.add_argument('--preconditioner',choices=['diag','mas'],help='PCG preconditioner; defaults to the parent setting')
    p.add_argument('--no-cuda-graph',action='store_true',help='Disable CUDA graph replay for PCG')
    p.add_argument('--solver-trace',action='store_true')
    p.add_argument('--max-newton',type=int,
                   help='Newton iteration budget per frame; by default inherit the seed budget')
    a=p.parse_args()
    lower=a.lower.resolve()
    out=a.output.resolve()
    seed=json.loads((lower/'run.json').read_text())
    assert seed['completed'] and json.loads((lower/'continuation_audit.json').read_text())['passed']
    steps=seed['steps'] if a.steps is None else a.steps
    if not 1<=steps<=seed['completed_steps']:
        p.error('The requested window must be covered by the audited parent')
    if not seed['level'] < a.last <= 7:
        p.error('Last level must follow the seed and be at most 7')
    if (out/'chain.json').exists():
        p.error('Choose a new output directory for this trajectory')
    out.mkdir(parents=True,exist_ok=True)
    protected=out/'protected_manifest.json'
    if not protected.exists():
        files=[lower/name for name in ('run.json','positions.npy','continuation_audit.json')]
        hashes={}
        for path in files:
            with path.open('rb') as source:
                hashes[str(path)]=hashlib.file_digest(source,'sha256').hexdigest()
        protected.write_text(json.dumps(hashes,indent=2))
    budget=a.max_newton if a.max_newton is not None else seed['config']['newton']['max_iter']
    if budget<1:p.error('Newton iteration budget must be positive')
    if seed.get('frozen_offset_only'):
        gradient_mode='--frozen-offset'
    else:
        gradient_mode='--exact-offset' if seed['exact_offset_gradient'] else '--rescue'
    package=a.native_package or continuation_package(seed)
    preconditioner=a.preconditioner or seed['config']['linear_system']['fem_preconditioner']
    velocity_map=seed.get('velocity_map','intrinsic') if a.method=='velpro' else 'intrinsic'
    chain=dict(status='running',seed=str(lower),method=a.method,levels=[],last=a.last,steps=steps)
    def save():(out/'chain.json').write_text(json.dumps(chain,indent=2))
    save()
    for level in range(seed['level']+1,a.last+1):
        target=out/f'L{level}';chain.update(current_level=level,current_run=str(target),stage='simulation');save()
        cmd=[sys.executable,str(ROOT/'scripts/run_bunny_ground_fine.py'),'--scene','bunny','--method',a.method,'--level',str(level),'--lower',str(lower),'--assets',seed['assets'],'--output',str(target),'--native-package',str(package),'--steps',str(steps),'--max-newton',str(budget),gradient_mode,'--velocity-map',velocity_map,'--refined-probe','--adaptive-search']
        cmd.extend(['--preconditioner',preconditioner])
        if a.no_cuda_graph or not seed['config']['linear_system']['use_cuda_graph']:cmd.append('--no-cuda-graph')
        if a.solver_trace:cmd.append('--solver-trace')
        with target.with_suffix('.log').open('w') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        m=json.loads((target/'run.json').read_text()) if (target/'run.json').exists() else {}
        if r.returncode or not m.get('completed'):
            chain.update(status='failed_level',failure=m.get('error',f'Solver process exited with code {r.returncode}'),process_exit_code=r.returncode,accepted_steps=m.get('completed_steps',0));save();return 1
        chain.update(stage='audit');save()
        with (target/'audit_commands.log').open('w') as log:
            checks=['audit_paper_run.py']+(['audit_paper_non_descent.py'] if level<7 else [])+['audit_bunny_ground_fine.py']
            for script in checks:
                r=subprocess.run([sys.executable,str(ROOT/'scripts'/script),str(target)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if r.returncode:chain.update(status='failed_audit',failure=script);save();return 1
        audit=json.loads((target/'audit.json').read_text());ground=json.loads((target/'ground_audit.json').read_text())
        chain['levels'].append(dict(level=level,run=str(target),vertices=m['vertices'],wall_seconds=m['wall_seconds'],native_steps=m['native_steps'],preview_steps=m['preview_steps'],maximum_stretch=audit['maximum_stretch'],minimum_height_m=ground['minimum_vertex_height_m']))
        lower=target;save();print(f'VERIFIED L{level}: {m["wall_seconds"]:.1f}s',flush=True)
    chain.update(status='completed',stage='verified');save();return 0

if __name__=='__main__':sys.exit(main())
