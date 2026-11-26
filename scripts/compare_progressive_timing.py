"""Compare matched saved runs without hiding the cost of coarse simulation."""
import argparse
import json
from pathlib import Path


def compare(progressive, direct, method='velpro'):
    a=json.loads((progressive/'experiment.json').read_text())
    b=json.loads((direct/'experiment.json').read_text())
    if 'continuation' in a or 'continuation' in b:
        raise ValueError('Inherited timings are not a fresh benchmark; rerun the matched experiment')
    if not a['completed'] or not b['completed']:
        raise ValueError('Both experiments must be complete')
    ignore={'output','methods'}
    pa={k:v for k,v in a['parameters'].items() if k not in ignore}
    pb={k:v for k,v in b['parameters'].items() if k not in ignore}
    if pa!=pb:raise ValueError('Parameters differ; this is not a matched timing comparison')
    for key in ('mesh_hashes','dt','steps','backend','robot_q_sha256'):
        if a.get(key)!=b.get(key):raise ValueError(f'Mismatched {key}')
    kernels=('shell.py','solver.py','prolongation.py','integrators.py','contact.py','affine.py')
    if 'code_sha256' in a and 'code_sha256' in b:
        for name in kernels:
            if a['code_sha256'][name]!=b['code_sha256'][name]:
                raise ValueError(f'Mismatched physics kernel: {name}')
    def entries(config,name):
        selected=sorted((r for r in config['runs'] if r['method']==name),key=lambda r:r['level'])
        if len(selected)!=config['parameters']['levels']:raise ValueError('Incomplete hierarchy')
        return selected
    progressive_levels=entries(a,method)
    direct_levels=entries(b,'direct')
    cumulative=sum(r['physics_seconds'] for r in progressive_levels)
    fine=direct_levels[-1]['physics_seconds']
    seconds=a['steps']*a['dt']
    return {
        'progressive_source':str(progressive.resolve()),'direct_source':str(direct.resolve()),
        'method':method,'physical_duration_seconds':seconds,
        'progressive_level_seconds':[r['physics_seconds'] for r in progressive_levels],
        'progressive_total_seconds':cumulative,'direct_finest_seconds':fine,
        'direct_level_seconds':[r['physics_seconds'] for r in direct_levels],
        'direct_over_progressive_total':fine/cumulative,
        'progressive_simulated_seconds_per_compute_second':seconds/cumulative,
        'direct_simulated_seconds_per_compute_second':seconds/fine,
        'timing_scope':progressive_levels[-1]['timing_scope'],
        'interpretation':'One diagnostic run, not a repeated or resource-isolated performance benchmark. Ratio > 1 favors progressive; all coarse levels are charged. Initialization, hierarchy construction, metrics and rendering excluded.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('progressive',type=Path);p.add_argument('direct',type=Path)
    p.add_argument('--method',default='velpro',choices=('semi','full','velpro'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=compare(args.progressive,args.direct,args.method)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
